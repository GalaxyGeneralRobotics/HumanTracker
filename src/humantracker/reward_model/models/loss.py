import torch
import torch.nn.functional as F


class SoftTargetBradleyTerryLoss(torch.nn.Module):
    """Bradley-Terry loss with a soft target for `Similar` pairs.

    A strict preference targets 1.0 on the sigmoid of the score gap, matching the
    standard Bradley-Terry objective. A `Similar` pair instead targets 0.5 (neither
    trajectory preferred) and is reweighted by `tie_weight` since ties are rarer
    than strict preferences. `temperature` scales the score gap before the sigmoid.
    """

    def __init__(self, tie_weight: float, temperature: float):
        super().__init__()
        if tie_weight <= 0 or temperature <= 0:
            raise ValueError("tie_weight and temperature must be positive")
        self.tie_weight = tie_weight
        self.temperature = temperature

    def forward(self, score_a: torch.Tensor, score_b: torch.Tensor, is_preference: torch.Tensor) -> torch.Tensor:
        gap = (score_a - score_b).flatten()
        targets = torch.where(is_preference, torch.ones_like(gap), torch.full_like(gap, 0.5))
        weights = torch.where(is_preference, torch.ones_like(gap), torch.full_like(gap, self.tie_weight))
        losses = F.binary_cross_entropy_with_logits(gap / self.temperature, targets, reduction="none")
        return (weights * losses).sum() / weights.sum()
