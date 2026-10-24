import torch
import torch.nn as nn
import torch.nn.functional as F


class ToFClearanceCost(nn.Module):
    """Penalise ToF zone readings closer than a clearance distance.

    Per sensor: c = relu(1 - d_min / clearance) ** 2, with d_min the closest zone
    reading; summed over sensors. 0 while everything is beyond the clearance,
    1 per sensor at contact. Zones with no target (inf) contribute nothing.

    forward(zones) → (B,)
    zones: (B, S, rows, cols) zone distances (m) for S sensors.
    """

    def __init__(self, clearance: float = 0.05):
        super().__init__()
        self.clearance = clearance

    def forward(self, zones: torch.Tensor) -> torch.Tensor:
        d_min = zones.flatten(start_dim=2).min(dim=-1).values          # (B, S)
        return (F.relu(1.0 - d_min / self.clearance) ** 2).sum(dim=-1)
