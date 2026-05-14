import torch
import torch.nn as nn


class HeightMatchCost(nn.Module):
    """Penalise vertical offset between TCP and the target object.

    forward(tcp_z, obj_z) → (B,)
    tcp_z, obj_z: (B,) Z-coordinates in world frame.
    """

    def forward(self, tcp_z: torch.Tensor, obj_z: torch.Tensor) -> torch.Tensor:
        return torch.abs(tcp_z - obj_z)
