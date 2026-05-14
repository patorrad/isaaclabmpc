import torch
import torch.nn as nn


class DistCost(nn.Module):
    """L2 distance between two 3-D positions.

    forward(disp_vec) → (B,)
    disp_vec: (B, 3) displacement vector (a - b).
    """

    def forward(self, disp_vec: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(disp_vec, dim=1)
