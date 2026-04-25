"""Offline comparison of MPPI planner round-trip latency:
  Isaac Lab (isaaclabmpc) vs Genesis (genesismpc).

Usage:
    python examples/ur16e_push/plot_rtt.py --isaaclab examples/ur16e_push/logs/timing_YYYYMMDD_HHMMSS.npy

The Genesis file is loaded from its fixed location. Pass --no-genesis to skip it.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

GENESIS_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "genesismpc", "examples", "ur5_stick_block_push", "rtt_log.npy",
)

parser = argparse.ArgumentParser()
parser.add_argument("--isaaclab", required=True,
                    help="Path to Isaac Lab timing .npy file")
parser.add_argument("--genesis", default=os.path.normpath(GENESIS_DEFAULT),
                    help="Path to Genesis rtt_log.npy")
parser.add_argument("--no-genesis", action="store_true",
                    help="Skip Genesis data")
parser.add_argument("--skip-warmup", type=int, default=1,
                    help="Skip first N samples (default 1, avoids cold-start spike)")
args = parser.parse_args()

il = np.load(args.isaaclab)[args.skip_warmup:]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Planner round-trip latency comparison")

# ---- time-series ----
ax = axes[0]
ax.plot(il, lw=1, color="steelblue", label=f"Isaac Lab  (mean {il.mean():.1f} ms, n={len(il)})")

if not args.no_genesis and os.path.exists(args.genesis):
    ge = np.load(args.genesis)[args.skip_warmup:]
    ax.plot(ge, lw=1, color="tomato", alpha=0.8,
            label=f"Genesis  (mean {ge.mean():.1f} ms, n={len(ge)})")
else:
    ge = None
    if not args.no_genesis:
        print(f"[warn] Genesis file not found: {args.genesis}")

ax.set_xlabel("step")
ax.set_ylabel("RTT (ms)")
ax.set_title("RTT per step")
ax.legend()

# ---- histogram ----
ax = axes[1]
bins = np.linspace(0, max(il.max(), ge.max() if ge is not None else il.max()) * 1.05, 60)
ax.hist(il, bins=bins, color="steelblue", alpha=0.7,
        label=f"Isaac Lab  μ={il.mean():.1f} σ={il.std():.1f} ms")
if ge is not None:
    ax.hist(ge, bins=bins, color="tomato", alpha=0.7,
            label=f"Genesis  μ={ge.mean():.1f} σ={ge.std():.1f} ms")
ax.set_xlabel("RTT (ms)")
ax.set_ylabel("count")
ax.set_title("RTT distribution")
ax.legend()

fig.tight_layout()

out_path = args.isaaclab.replace(".npy", "_comparison.png")
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
plt.show()
