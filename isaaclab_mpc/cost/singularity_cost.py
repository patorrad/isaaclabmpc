import torch
import torch.nn as nn


class SingularityCost(nn.Module):
    """Penalise near-singular configurations via reciprocal manipulability.

    Uses 1 / (|det J| + ε) which grows large as the Jacobian loses rank.

    forward(J) → (B,)
    J: (B, 6, DOF) end-effector Jacobian.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, J: torch.Tensor) -> torch.Tensor:
        return 1.0 / (torch.abs(torch.linalg.det(J)) + self.eps)
