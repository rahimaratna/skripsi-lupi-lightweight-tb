import time
import torch
import torch.nn as nn
from torchvision import models as tv_models
from transformers import AutoModel

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class VisionEncoderWrapper(nn.Module):
    """
    Wrapper backbone visual untuk M1-M8 sesuai Fig. 1 Rajaraman et al. (2026):
    Backbone -> Global Average Pooling (GAP) -> output vektor fitur (B, C)
    Tanpa injeksi lapisan konvolusi tambahan agar parameter tetap ringan & pretrained utuh.
    """

    def __init__(self, backbone_name: str = "vgg11_bn", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.is_timm = False

        if self.backbone_name == "vgg11_bn":
            m = tv_models.vgg11_bn(
                weights=tv_models.VGG11_BN_Weights.DEFAULT if pretrained else None
            )
            self.features = m.features
            in_ch = 512
        elif self.backbone_name == "vgg11":
            m = tv_models.vgg11(
                weights=tv_models.VGG11_Weights.DEFAULT if pretrained else None
            )
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
            m = tv_models.resnet18(
                weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            self.features = nn.Sequential(*list(m.children())[:-2])
            in_ch = 512
        elif self.backbone_name == "densenet121":
            m = tv_models.densenet121(
                weights=tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
            )
            self.features = m.features
            in_ch = 1024
        else:
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
                out = self.features.forward_features(dummy) if hasattr(self.features, "forward_features") else self.features(dummy)
                in_ch = out.shape[1]

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_timm and hasattr(self.features, "forward_features"):
            feat = self.features.forward_features(x)
        else:
            feat = self.features(x)

        if feat.ndim == 4:
            pooled = self.gap(feat).flatten(1)
        elif feat.ndim == 3:
            pooled = feat.mean(dim=1)
        else:
            pooled = feat
        return pooled


class PrivilegedTextEncoder(nn.Module):
    """
    Encoder teks BiomedVLP-CXR-BERT-general.
    Sesuai paper hlm. 5 & 6 ('authors_default'): seluruh bobot CXR-BERT dibekukan
    dan dijaga dalam eval() mode untuk mencegah overfitting pada dataset kecil.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedVLP-CXR-BERT-general",
        freeze_policy: str = "authors_default",
    ):
        super().__init__()
        self.freeze_policy = freeze_policy
        self.bert = AutoModel.from_pretrained(model_name)
        self.out_dim = self.bert.config.hidden_size
        self.apply_freeze_policy(freeze_policy)

    def apply_freeze_policy(self, policy: str):
        if policy in ("authors_default", "freeze_all"):
            for p in self.bert.parameters():
                p.requires_grad = False
            self.bert.eval()
        elif policy == "unfreeze_last1":
            for p in self.bert.parameters():
                p.requires_grad = False
            if hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
                for p in self.bert.encoder.layer[-1].parameters():
                    p.requires_grad = True
        elif policy == "unfreeze_all":
            for p in self.bert.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"freeze_policy tidak dikenal: {policy}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_policy in ("authors_default", "freeze_all"):
            self.bert.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_policy in ("authors_default", "freeze_all"):
            with torch.no_grad():
                enc = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
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
    Mengukur metrik efisiensi komputasi saat INFERENSI (hanya cabang visual + projection + cls).
    """
    model.eval()
    infer_modules = [model.vision_encoder, model.img_dropout, model.img_proj, model.cls]

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
    Baseline Unimodal sesuai Fig. 1 Rajaraman et al. (2026):
    VisionEncoder (GAP) -> Dropout(0.3) -> Linear(C -> 256) -> Classifier(256 -> 2)
    """

    def __init__(
        self,
        backbone_name: str = "vgg11_bn",
        proj_dim: int = 256,
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.vision_encoder = VisionEncoderWrapper(backbone_name, pretrained=pretrained)
        self.img_dropout = nn.Dropout(p=0.3)
        self.img_proj = nn.Linear(self.vision_encoder.out_dim, proj_dim)
        self.cls = nn.Linear(proj_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h_img = self.vision_encoder(images)
        z_img = self.img_proj(self.img_dropout(h_img))
        logits = self.cls(z_img)
        return logits


class TBClassifierLUPI(nn.Module):
    """
    Model Multimodal LUPI sesuai Fig. 1 Rajaraman et al. (2026):
    - Image branch: VisionEncoder (GAP) -> Dropout(0.3) -> Linear(C -> 256)
    - Text branch : Frozen CXR-BERT -> Dropout(0.1) -> Linear(768 -> 256)
    - Shared classifier: Linear(256 -> 2)
    """

    def __init__(
        self,
        backbone_name: str = "vgg11_bn",
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

        self.img_dropout = nn.Dropout(p=0.3)
        self.txt_dropout = nn.Dropout(p=0.1)

        self.img_proj = nn.Linear(self.vision_encoder.out_dim, proj_dim)
        self.txt_proj = nn.Linear(self.text_encoder.out_dim, proj_dim)

        self.cls = nn.Linear(proj_dim, num_classes)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        h_img = self.vision_encoder(images)
        z_img = self.img_proj(self.img_dropout(h_img))
        logits_img = self.cls(z_img)

        if input_ids is None or attention_mask is None:
            return logits_img

        h_txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        z_txt = self.txt_proj(self.txt_dropout(h_txt))
        logits_txt = self.cls(z_txt)

        return logits_img, logits_txt, z_img, z_txt