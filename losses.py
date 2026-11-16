from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricNTXentLoss(nn.Module):
    """Symmetric NT-Xent contrastive loss untuk pasangan image-text (Persamaan 6 paper)."""
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
        batch_size = z_img.size(0)
        device = z_img.device
        targets = torch.arange(batch_size, device=device)

        logits_i2t = torch.matmul(z_img, z_txt.T) / self.temperature
        logits_t2i = torch.matmul(z_txt, z_img.T) / self.temperature

        loss_i2t = F.cross_entropy(logits_i2t, targets)
        loss_t2i = F.cross_entropy(logits_t2i, targets)
        return 0.5 * (loss_i2t + loss_t2i)


class TBMultimodalLoss(nn.Module):
    """
    Persamaan (7): L_total = L_c,img + L_c,txt + L_cos + lambda * L_NTX(tau)
    """
    def __init__(
        self,
        mode: str = "unimodal",
        lambda_align: float = 0.6,
        temperature: float = 0.05,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.lambda_align = lambda_align
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.cos_loss = nn.CosineEmbeddingLoss(margin=0.0)
        self.ntx_loss = SymmetricNTXentLoss(temperature=temperature)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        z_img: Optional[torch.Tensor] = None,
        z_txt: Optional[torch.Tensor] = None,
        logits_txt: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        l_c_img = self.ce_loss(logits, labels)

        if self.mode == "lupi" and (z_img is not None) and (z_txt is not None):
            l_c_txt = self.ce_loss(logits_txt, labels) if logits_txt is not None else torch.tensor(0.0, device=logits.device)
            target_pos = torch.ones(z_img.size(0), device=z_img.device)
            l_cos = self.cos_loss(z_img, z_txt, target_pos)
            l_ntx = self.ntx_loss(z_img, z_txt)

            l_align = l_cos + self.lambda_align * l_ntx
            l_total = l_c_img + l_c_txt + l_align
            return {
                "loss": l_total,
                "loss_ce": (l_c_img + l_c_txt).detach(),
                "loss_align": l_align.detach(),
            }

        return {
            "loss": l_c_img,
            "loss_ce": l_c_img.detach(),
            "loss_align": torch.tensor(0.0, device=logits.device),
        }
