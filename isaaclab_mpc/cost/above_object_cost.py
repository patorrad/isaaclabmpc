import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import quat_to_yaw_pitch


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
        circle_radius = self.half*math.sqrt(2)
        half = self.half - .005

        rel = tcp_pos - obj_pos                                         # (B, 3)
        local_x = rel[:, 0]
        local_y = rel[:, 1]
        # Rotate XY by -yaw only (Z-axis rotation, ignoring roll/pitch).
        # yaw = quat_to_yaw_pitch(obj_quat)[:, 0]                        # (B,)
        # cos_y, sin_y = torch.cos(-yaw), torch.sin(-yaw)
        # local_x = cos_y * rel[:, 0] - sin_y * rel[:, 1]               # (B,)
        # local_y = sin_y * rel[:, 0] + cos_y * rel[:, 1]               # (B,)

        # Z: peaks at 1 when TCP is exactly at the top face, Gaussian decay above,
        # zero when below.
        height_above = rel[:, 2]-half                               # (B,) signed
        above_z = torch.where(
            height_above >= 0,
            F.relu(1.0 - (height_above / half)**2),
            torch.zeros_like(height_above),
        )                                                               # (B,)

        # XY: negative parabolic decay — 1 at center, 0 at circle_radius and beyond
        r = torch.sqrt(local_x**2 + local_y**2)
        xy_cost = F.relu(1.0 - (r / circle_radius)**2)
        return above_z * xy_cost


        # rel = tcp_pos - obj_pos
        # half = self.half - .005
        # height_above = rel[:, 2]-half 
        # radius = self.half*math.sqrt(2)/2
        # cost = torch.where(
        #     torch.logical_and(height_above > 0, rel.norm(dim=1) < radius),
        #     torch.ones_like(height_above),
        #     torch.zeros_like(height_above)
        # )
        # return cost

