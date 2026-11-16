from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import timm
from transformers import AutoModel


class VisionEncoderWrapper(nn.Module):
    """
    Wrapper terstandarisasi untuk 5 arsitektur eksperimen M1-M5:
      - M1: vgg11
      - M2: mobilenetv3 (mobilenet_v3_large)
      - M3: shufflenetv2 (shufflenet_v2_x1_0)
      - M4: efficientnet_b0
      - M5: starnet (starnet_s3 via timm) atau vit (vit_tiny_patch16_224)
    """
    SUPPORTED = {
        "vgg11",
        "mobilenetv3",
        "shufflenetv2",
        "efficientnet_b0",
        "starnet",
        "vit",
    }

    def __init__(self, backbone_name: str = "vgg11", pretrained: bool = True):
        super().__init__()
        name = backbone_name.lower().strip()
        if name not in self.SUPPORTED:
            raise ValueError(f"Backbone '{backbone_name}' tidak didukung. Pilih dari: {sorted(self.SUPPORTED)}")

        self.backbone_name = name

        if name == "vgg11":
            weights = tvm.VGG11_BN_Weights.DEFAULT if pretrained else None
            m = tvm.vgg11_bn(weights=weights)
            self.features = m.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = 512
            self.is_timm = False

        elif name == "mobilenetv3":
            weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            m = tvm.mobilenet_v3_large(weights=weights)
            self.features = m.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = 960
            self.is_timm = False

        elif name == "shufflenetv2":
            weights = tvm.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
            m = tvm.shufflenet_v2_x1_0(weights=weights)
            self.features = nn.Sequential(
                m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4, m.conv5
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = 1024
            self.is_timm = False

        elif name == "efficientnet_b0":
            weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            m = tvm.efficientnet_b0(weights=weights)
            self.features = m.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = 1280
            self.is_timm = False

        elif name == "starnet":
            self.features = timm.create_model("starnet_s3", pretrained=pretrained, num_classes=0)
            self.pool = nn.Identity()
            self.out_dim = self.features.num_features
            self.is_timm = True

        elif name == "vit":
            self.features = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
            self.pool = nn.Identity()
            self.out_dim = self.features.num_features
            self.is_timm = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        if not self.is_timm:
            feat = self.pool(feat).flatten(1)
        return feat


class ProjectionHead(nn.Module):
    """Memproyeksikan fitur modalitas ke ruang laten bersama berdimensi proj_dim (256)."""
    def __init__(self, in_dim: int, proj_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrivilegedTextEncoder(nn.Module):
    """Encoder teks klinis untuk LUPI (hanya digunakan saat fase training)."""
    def __init__(
        self,
        model_name: str = "microsoft/BiomedVLP-CXR-BERT-general",
        proj_dim: int = 256,
        freeze_policy: str = "last_n",
        unfreeze_last_n: int = 2,
    ):
        super().__init__()
        try:
            self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
        except Exception:
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        hidden_dim = self.encoder.config.hidden_size
        self.proj = ProjectionHead(hidden_dim, proj_dim=proj_dim)
        self._apply_freeze_policy(freeze_policy, unfreeze_last_n)

    def _apply_freeze_policy(self, policy: str, last_n: int):
        for p in self.encoder.parameters():
            p.requires_grad = False

        if policy == "train_all":
            for p in self.encoder.parameters():
                p.requires_grad = True
        elif policy == "last_n":
            layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
            if layers is not None and last_n > 0:
                for layer in layers[-last_n:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        pooled = torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        z_txt = self.proj(pooled)
        return F.normalize(z_txt, p=2, dim=-1)


class TBClassifierLUPI(nn.Module):
    """
    Model utama untuk skema Unimodal maupun Multimodal LUPI.
    Saat inferensi (input_ids=None), cabang teks tidak dieksekusi sama sekali.
    """
    def __init__(
        self,
        backbone_name: str = "vgg11",
        mode: str = "unimodal",
        proj_dim: int = 256,
        num_classes: int = 2,
        pretrained: bool = True,
        text_model_name: str = "microsoft/BiomedVLP-CXR-BERT-general",
        text_freeze_policy: str = "last_n",
        unfreeze_last_n: int = 2,
    ):
        super().__init__()
        assert mode in {"unimodal", "lupi"}, "mode harus 'unimodal' atau 'lupi'"
        self.mode = mode

        self.vision_encoder = VisionEncoderWrapper(backbone_name=backbone_name, pretrained=pretrained)
        self.vision_proj = ProjectionHead(self.vision_encoder.out_dim, proj_dim=proj_dim)
        self.classifier = nn.Linear(proj_dim, num_classes)

        self.text_encoder: Optional[PrivilegedTextEncoder] = None
        if self.mode == "lupi":
            self.text_encoder = PrivilegedTextEncoder(
                model_name=text_model_name,
                proj_dim=proj_dim,
                freeze_policy=text_freeze_policy,
                unfreeze_last_n=unfreeze_last_n,
            )

    def forward(
        self,
        image: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        v_feat = self.vision_encoder(image)
        z_img_raw = self.vision_proj(v_feat)
        logits = self.classifier(z_img_raw)
        z_img = F.normalize(z_img_raw, p=2, dim=-1)

        outputs = {
            "logits": logits,
            "z_img": z_img,
        }

        if self.mode == "lupi" and self.training and (input_ids is not None) and (attention_mask is not None):
            z_txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            outputs["z_txt"] = z_txt

        return outputs
