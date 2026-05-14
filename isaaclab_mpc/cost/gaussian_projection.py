"""Gaussian projection for MPPI cost shaping.

Ported from STORM (NVlabs/storm/storm_kit/mpc/cost/gaussian_projection.py).

Maps an unbounded cost scalar to a bounded range using a Gaussian-shaped transfer:

    f(x) = 1 - (-1)^n * exp(-(x - s)^2 / (2 * c^2)) + r * (x - s)^4

Key properties:
  - n=1, s=0: bell centred at 0; f(0) = 0, f(±∞) → 1.  Maps distances to [0, 1].
  - c=0: passthrough identity (no projection applied).
  - r: small quartic coefficient (e.g. 1e-5) prevents complete saturation far from centre.
"""

import torch
import torch.nn as nn


class GaussianProjection(nn.Module):
    """Project a (B,) cost tensor through a Gaussian-shaped transfer function.

    Args:
        n: sign exponent.  n=1 → 0 at centre, ~1 far away (use for distance costs).
        c: bandwidth in cost units.  c=0 → passthrough (identity).
        s: shift; centre of the bell (default 0).
        r: quartic tail coefficient; prevents hard saturation (default 0).
    """

    def __init__(self, n: int = 1, c: float = 0.0, s: float = 0.0, r: float = 0.0):
        super().__init__()
        self.n = n
        self.c = c
        self.s = s
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c == 0:
            return x
        xp = x - self.s
        return 1.0 - ((-1.0) ** self.n) * torch.exp(-xp ** 2 / (2.0 * self.c ** 2)) + self.r * xp ** 4
