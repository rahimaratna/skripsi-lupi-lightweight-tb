import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricNTXentLoss(nn.Module):
    """
    Symmetric Contrastive NT-Xent Loss lintas-modalitas (Persamaan 6 Rajaraman et al., 2026).
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
        z_img = F.normalize(z_img.float(), dim=-1)
        z_txt = F.normalize(z_txt.float(), dim=-1)

        logits_i2t = torch.matmul(z_img, z_txt.t()) / self.temperature
        logits_t2i = logits_i2t.t()

        batch_size = z_img.size(0)
        targets = torch.arange(batch_size, device=z_img.device)

        loss_i2t = F.cross_entropy(logits_i2t, targets)
        loss_t2i = F.cross_entropy(logits_t2i, targets)
        return 0.5 * (loss_i2t + loss_t2i)


class LUPITotalLoss(nn.Module):
    """
    Persamaan (7) Rajaraman et al. (2026):
    L_total = L_{c,img} + L_{c,txt} + L_cos + lambda * L_NTX(tau)
    """

    def __init__(self, lambda_align: float = 0.6, temperature: float = 0.05):
        super().__init__()
        self.lambda_align = lambda_align
        self.ce = nn.CrossEntropyLoss()
        self.ntxent = SymmetricNTXentLoss(temperature=temperature)
        self.cos_embed = nn.CosineEmbeddingLoss()

    def forward(
        self,
        logits_img: torch.Tensor,
        logits_txt: torch.Tensor,
        z_img: torch.Tensor,
        z_txt: torch.Tensor,
        labels: torch.Tensor,
    ):
        loss_cls_img = self.ce(logits_img, labels)
        loss_cls_txt = self.ce(logits_txt, labels)

        loss_ntxent = self.ntxent(z_img, z_txt)
        cos_target = torch.ones(z_img.size(0), device=z_img.device)
        loss_cos = self.cos_embed(z_img.float(), z_txt.float(), cos_target)

        total_loss = loss_cls_img + loss_cls_txt + loss_cos + (self.lambda_align * loss_ntxent)

        metrics = {
            "total": total_loss.item(),
            "cls_img": loss_cls_img.item(),
            "cls_txt": loss_cls_txt.item(),
            "ntxent": loss_ntxent.item(),
            "cos": loss_cos.item(),
        }
        return total_loss, metrics