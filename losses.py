from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive / NT-Xent Loss lintas modalitas (Image <-> Text)
    dengan bobot positif berdasarkan kesamaan label kelas.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_img: torch.Tensor, z_txt: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = z_img.device
        batch_size = z_img.shape[0]

        features = torch.cat([z_img, z_txt], dim=0)
        targets = torch.cat([labels, labels], dim=0).view(-1, 1)

        mask = torch.eq(targets, targets.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(2 * batch_size, device=device)
        mask = mask * logits_mask

        sim_matrix = torch.matmul(features, features.T) / self.temperature
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        pos_per_row = mask.sum(dim=1)
        valid_rows = pos_per_row > 0
        if not valid_rows.any():
            return torch.tensor(0.0, device=device)

        mean_log_prob_pos = (mask[valid_rows] * log_prob[valid_rows]).sum(dim=1) / pos_per_row[valid_rows]
        return -mean_log_prob_pos.mean()


class TBMultimodalLoss(nn.Module):
    """
    Total Loss:
      - Mode Unimodal: L_total = L_CE
      - Mode LUPI:     L_total = L_CE + lambda_align * (0.5 * L_cos + 0.5 * L_supcon)
    """
    def __init__(
        self,
        mode: str = "unimodal",
        lambda_align: float = 0.5,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.lambda_align = lambda_align
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.cos_loss = nn.CosineEmbeddingLoss(margin=0.0)
        self.supcon_loss = SupervisedContrastiveLoss(temperature=temperature)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        z_img: Optional[torch.Tensor] = None,
        z_txt: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        l_ce = self.ce_loss(logits, labels)

        if self.mode == "lupi" and (z_img is not None) and (z_txt is not None):
            target_pos = torch.ones(z_img.size(0), device=z_img.device)
            l_cos = self.cos_loss(z_img, z_txt, target_pos)
            l_con = self.supcon_loss(z_img, z_txt, labels)
            l_align = 0.5 * l_cos + 0.5 * l_con
            l_total = l_ce + self.lambda_align * l_align
            return {
                "loss": l_total,
                "loss_ce": l_ce.detach(),
                "loss_align": l_align.detach(),
            }

        return {
            "loss": l_ce,
            "loss_ce": l_ce.detach(),
            "loss_align": torch.tensor(0.0, device=logits.device),
        }
