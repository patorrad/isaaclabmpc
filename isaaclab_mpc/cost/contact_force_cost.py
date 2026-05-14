import torch
import torch.nn as nn


class ContactForceCost(nn.Module):
    """Penalise vertical contact forces on the robot wrist.

    Proxies for undesired collisions with the environment.

    forward(forces) → (B,)
    forces: (B, 1, 3) contact force tensor — uses the Z component.
    """

    def forward(self, forces: torch.Tensor) -> torch.Tensor:
        return torch.abs(forces[:, 0, 2])
