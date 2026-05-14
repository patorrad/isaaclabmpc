import torch
import torch.nn as nn


class JointVelCost(nn.Module):
    """Penalise high joint velocities to encourage smooth, slow motion.

    forward(joint_vel) → (B,)
    joint_vel: (B, DOF) joint velocity tensor.
    """

    def forward(self, joint_vel: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(joint_vel, dim=1)
