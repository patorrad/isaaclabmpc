import torch
import torch.nn as nn

from .utils import quat_apply


class OrientationCost(nn.Module):
    """Penalise polar tilt of the tool axis away from pointing straight down.

    Projects the EE's local +Z axis (= tool direction, per tcp_offset_local convention)
    into world coordinates and returns the angle from [0, 0, -1] (straight down).

    Yaw (rotation around the vertical axis) is free — it does not tilt the tool
    and should not be penalised here; push_align handles azimuthal positioning.

    forward(ee_quat) → (B,)  angle in radians; 0 = tool pointing straight down.
    ee_quat: (B, 4) wxyz quaternion of the end-effector.
    """

    def forward(self, ee_quat: torch.Tensor) -> torch.Tensor:
        B = ee_quat.shape[0]
        tool_local = ee_quat.new_zeros(B, 3)
        tool_local[:, 2] = 1.0               # EE +Z = tool axis
        tool_world = quat_apply(ee_quat, tool_local)
        # angle from [0, 0, -1]: cos = dot(tool_world, [0,0,-1]) = -tool_world[:, 2]
        cos_angle = (-tool_world[:, 2]).clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.acos(cos_angle)
