import torch
import torch.nn as nn


class PushAlignCost(nn.Module):
    """Gated cosine alignment cost for pushing tasks.

    Measures whether the TCP is behind the object relative to the push direction.
    A sigmoid gate fades the cost to zero once the TCP reaches the object's back face,
    so the robot stops repositioning and commits to pushing.

    Args:
        align_gate_dist: distance threshold at which the gate activates (metres).
            Typically obj_size / 2 + standoff (e.g. 0.036 m for a 5 cm cube).
        gate_width: sigmoid transition width in metres (default 0.03 m).

    forward(robot_to_obj, obj_to_goal, robot_to_obj_dist) → (B,)
        robot_to_obj:      (B, 3) vector from object to TCP (tcp_pos - obj_pos)
        obj_to_goal:       (B, 3) vector from object to goal (goal_pos - obj_pos)
        robot_to_obj_dist: (B,)   pre-computed L2 norm of robot_to_obj
    """

    def __init__(self, align_gate_dist: float, gate_width: float = 0.03):
        super().__init__()
        self.align_gate_dist = align_gate_dist
        self.gate_width = gate_width

    def forward(
        self,
        robot_to_obj: torch.Tensor,
        obj_to_goal: torch.Tensor,
        robot_to_obj_dist: torch.Tensor,
    ) -> torch.Tensor:
        r2b_2d = robot_to_obj[:, :2]
        b2g_2d = obj_to_goal[:, :2]
        cosine = (
            torch.sum(r2b_2d * b2g_2d, dim=1)
            / (torch.linalg.norm(r2b_2d, dim=1).clamp(min=1e-6)
               * torch.linalg.norm(b2g_2d, dim=1).clamp(min=1e-6))
        )
        align = cosine + 1.0  # shift to [0, 2]: 0 = ideal, 2 = worst
        gate = torch.sigmoid((robot_to_obj_dist - self.align_gate_dist) / self.gate_width)
        return align * gate
