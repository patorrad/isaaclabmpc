import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import quat_apply


class AboveObjectCost(nn.Module):
    """Penalise TCP for passing above a target object's rotated XY footprint.

    Transforms the TCP into the object's local frame (using the object quaternion)
    then checks against the axis-aligned cube [-half, half]^3 in that frame.

    Cost is zero when the TCP is below the cube's top face or outside its XY
    footprint; rises as the TCP moves further above and more centred.

    Normalised so cost == 1 when TCP is exactly one half-size above the top face
    and centred over the cube.

    forward(tcp_pos, obj_pos, obj_quat) -> (B,)
    """

    def __init__(self, obj_half_size: float = 0.025):
        super().__init__()
        self.half = obj_half_size

    def forward(
        self,
        tcp_pos:  torch.Tensor,   # (B, 3) world frame
        obj_pos:  torch.Tensor,   # (B, 3) world frame
        obj_quat: torch.Tensor,   # (B, 4) wxyz unit quaternion
    ) -> torch.Tensor:            # (B,)
        half = self.half

        # Transform TCP displacement into the object's local frame.
        # Inverse of unit quaternion (w,x,y,z) = conjugate (w,-x,-y,-z).
        rel = tcp_pos - obj_pos                                         # (B, 3)
        sign = torch.tensor([1.0, -1.0, -1.0, -1.0],
                            device=obj_quat.device, dtype=obj_quat.dtype)
        q_inv = obj_quat * sign                                         # (B, 4)
        local = quat_apply(q_inv, rel)                                  # (B, 3)

        # Z: how far above the top face (local z > half)
        above_z = F.relu(local[:, 2] - half) / half                    # (B,)

        # XY: normalised overlap within the footprint
        x_in = F.relu(half - local[:, 0].abs()) / half                 # (B,)
        y_in = F.relu(half - local[:, 1].abs()) / half                 # (B,)

        return above_z * x_in * y_in
