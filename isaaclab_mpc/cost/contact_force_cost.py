import torch
import torch.nn as nn
import torch.nn.functional as F



class ContactForceCost(nn.Module):
    """Penalise contact forces on the sensor body.

    forward(forces) → (B,)
    forces: (B, num_bodies, 3) net contact force tensor (net_forces_w).
    Returns the sum of 3-D force norms across all sensor bodies.
    """
    
    def __init__(self, threshold: float = 2.5):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, forces: torch.Tensor) -> torch.Tensor:
        return F.relu(( torch.abs(forces[:, 0, 2]) - self.threshold))/self.threshold
