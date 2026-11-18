import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
from transformers import AutoModel

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


def _ends_with_3x3(module: nn.Module) -> bool:
    """Mengecek apakah layer Conv2d terakhir pada feature extractor memiliki kernel 3x3."""
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        return True
    return last_conv.kernel_size == (3, 3)


class VisionEncoderWrapper(nn.Module):
    """
    Wrapper backbone visual untuk M1-M8:
    - M1: vgg11 (exact baseline paper)
    - M2: mobilenet_v3_large
    - M3: shufflenet_v2_x1_0
    - M4: efficientnet_b0
    - M5: starnet_s3 (timm)
    - M6: resnet18
    - M7: densenet121
    - M8: ghostnet_100 (timm)
    Menyisipkan blok post3x3 otomatis apabila konvolusi terakhir bukan 3x3 (sesuai paper).
    """

    def __init__(self, backbone_name: str = "vgg11", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.is_timm = False

        if self.backbone_name == "vgg11":
            m = tv_models.vgg11(weights=tv_models.VGG11_Weights.DEFAULT if pretrained else None)
            self.features = m.features
            in_ch = 512
        elif self.backbone_name in ("mobilenet_v3_large", "mobilenetv3"):
            m = tv_models.mobilenet_v3_large(
                weights=tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            )
            self.features = m.features
            in_ch = 960
        elif self.backbone_name in ("shufflenet_v2_x1_0", "shufflenetv2"):
            m = tv_models.shufflenet_v2_x1_0(
                weights=tv_models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
            )
            self.features = nn.Sequential(
                m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4, m.conv5
            )
            in_ch = 1024
        elif self.backbone_name in ("efficientnet_b0", "efficientnet"):
            m = tv_models.efficientnet_b0(
                weights=tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            self.features = m.features
            in_ch = 1280
        elif self.backbone_name == "resnet18":
            m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.features = nn.Sequential(*list(m.children())[:-2])
            in_ch = 512
        elif self.backbone_name == "densenet121":
            m = tv_models.densenet121(
                weights=tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
            )
            self.features = m.features
            in_ch = 1024
        else:
            # Untuk backbone timm seperti starnet_s3 dan ghostnet_100
            if not HAS_TIMM:
                raise ValueError(f"Backbone '{backbone_name}' membutuhkan library timm.")
            timm_alias = {
                "starnet": "starnet_s3",
                "ghostnet": "ghostnet_100",
            }
            resolved_name = timm_alias.get(self.backbone_name, self.backbone_name)
            self.is_timm = True
            self.features = timm.create_model(
                resolved_name, pretrained=pretrained, num_classes=0, global_pool=""
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.features(dummy)
                in_ch = out.shape[1]

        if not _ends_with_3x3(self.features):
            self.post3x3 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.SiLU(inplace=True),
            )
        else:
            self.post3x3 = nn.Identity()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.post3x3(feat)
        pooled = self.gap(feat).flatten(1)
        return pooled


class PrivilegedTextEncoder(nn.Module):
    """
    Encoder teks BiomedVLP-CXR-BERT-general dengan kebijakan fine-tuning 'authors_default'
    (membekukan embeddings & pooler, membuka blok encoder terakhir layer[-1])
    serta menggunakan pooler_output / [CLS] sesuai paper rujukan.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedVLP-CXR-BERT-general",
        freeze_policy: str = "authors_default",
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.out_dim = self.bert.config.hidden_size
        self.apply_freeze_policy(freeze_policy)

    def apply_freeze_policy(self, policy: str):
        if policy == "freeze_all":
            for p in self.bert.parameters():
                p.requires_grad = False
        elif policy == "unfreeze_all":
            for p in self.bert.parameters():
                p.requires_grad = True
        elif policy == "authors_default":
            for p in self.bert.parameters():
                p.requires_grad = False
            if hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
                for p in self.bert.encoder.layer[-1].parameters():
                    p.requires_grad = True
            else:
                for name, p in self.bert.named_parameters():
                    if "encoder.layer.11." in name:
                        p.requires_grad = True
        else:
            raise ValueError(f"freeze_policy tidak dikenal: {policy}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        enc = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(enc, "pooler_output", None) is not None:
            return enc.pooler_output
        return enc.last_hidden_state[:, 0]


def measure_inference_efficiency(
    model: nn.Module,
    device: torch.device,
    img_size: int = 224,
    warmup: int = 20,
    runs: int = 100,
) -> dict:
    """
    Mengukur metrik efisiensi komputasi saat INFERENSI (hanya cabang visual + projection + cls):
    - infer_params_m : Total parameter inferensi (Juta)
    - infer_size_mb  : Ukuran model inferensi dalam FP32 (MB)
    - latency_ms     : Rata-rata waktu inferensi per citra (ms)
    """
    model.eval()
    infer_modules = [model.vision_encoder, model.img_proj, model.mid, model.cls]

    total_params = sum(p.numel() for mod in infer_modules for p in mod.parameters())
    total_bytes = sum(
        p.numel() * p.element_size() for mod in infer_modules for p in mod.parameters()
    ) + sum(
        b.numel() * b.element_size() for mod in infer_modules for b in mod.buffers()
    )

    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    latency_ms = ((t1 - t0) / runs) * 1000.0
    train_params = sum(p.numel() for p in model.parameters())

    return {
        "infer_params_m": round(total_params / 1e6, 3),
        "infer_size_mb": round(total_bytes / (1024 ** 2), 3),
        "latency_ms": round(latency_ms, 3),
        "train_params_m": round(train_params / 1e6, 3),
    }


class TBClassifierUnimodal(nn.Module):
    """
    Baseline Unimodal dengan projection head identik (Linear -> ReLU -> mid -> cls).
    """

    def __init__(
        self,
        backbone_name: str = "vgg11",
        proj_dim: int = 256,
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.vision_encoder = VisionEncoderWrapper(backbone_name, pretrained=pretrained)
        self.img_proj = nn.Linear(self.vision_encoder.out_dim, proj_dim)
        self.act = nn.ReLU(inplace=True)
        self.mid = nn.Linear(proj_dim, proj_dim)
        self.cls = nn.Linear(proj_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        f_img = self.vision_encoder(images)
        z_img = self.mid(self.act(self.img_proj(f_img)))
        logits = self.cls(z_img)
        return logits


class TBClassifierLUPI(nn.Module):
    """
    Model Multimodal LUPI sesuai arsitektur MultimodalNet pada paper rujukan.
    """

    def __init__(
        self,
        backbone_name: str = "vgg11",
        text_model_name: str = "microsoft/BiomedVLP-CXR-BERT-general",
        proj_dim: int = 256,
        num_classes: int = 2,
        pretrained: bool = True,
        text_freeze_policy: str = "authors_default",
    ):
        super().__init__()
        self.vision_encoder = VisionEncoderWrapper(backbone_name, pretrained=pretrained)
        self.text_encoder = PrivilegedTextEncoder(
            model_name=text_model_name, freeze_policy=text_freeze_policy
        )

        self.img_proj = nn.Linear(self.vision_encoder.out_dim, proj_dim)
        self.txt_proj = nn.Linear(self.text_encoder.out_dim, proj_dim)
        self.act = nn.ReLU(inplace=True)

        self.mid = nn.Linear(proj_dim, proj_dim)
        self.cls = nn.Linear(proj_dim, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        f_img = self.vision_encoder(images)
        z_img = self.mid(self.act(self.img_proj(f_img)))
        logits_img = self.cls(z_img)

        if input_ids is None or attention_mask is None:
            return logits_img

        f_txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        z_txt = self.mid(self.act(self.txt_proj(f_txt)))
        logits_txt = self.cls(z_txt)

        return logits_img, logits_txt, z_img, z_txt