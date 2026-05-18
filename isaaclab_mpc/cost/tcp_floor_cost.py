import torch
import torch.nn as nn
import torch.nn.functional as F


class TcpFloorCost(nn.Module):
    """Penalise TCP Z dropping below threshold.  forward(tcp_z) -> (B,)

    Normalised so cost = 1 when tcp_z == table_surface_z (TCP at the table top).
    Cost is 0 above threshold and rises linearly below it.
    """

    def __init__(self, threshold: float, table_surface_z: float):
        super().__init__()
        self.threshold = threshold
        self._norm = threshold - table_surface_z  # = tcp_floor_offset
        print(threshold, table_surface_z)

    def forward(self, tcp_z: torch.Tensor) -> torch.Tensor: 
        return F.relu(self.threshold - tcp_z) / self._norm
