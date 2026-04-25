"""MPPI planner server for the UR16e reach task (Isaac Lab backend).

Analogous to genesismpc/examples/ur5_stick_stand/planner.py.

Run with:
    cd /home/paolo/Documents/isaaclabmpc
    conda activate env_isaaclab
    python examples/ur16e_reach/planner.py

The server listens on tcp://0.0.0.0:4242 and accepts the same zerorpc
calls as the genesismpc planner (compute_action_tensor, set_goal, …).

IMPORTANT
---------
Isaac Lab requires AppLauncher to be created and the simulation app started
BEFORE any isaaclab modules are imported.  All isaaclab imports therefore
appear AFTER the AppLauncher block below.
"""

# ===========================================================================
# 1. Simulator bootstrap  — must happen first
# ===========================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR16e MPPI reach planner (Isaac Lab)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True          # planner always runs headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. All other imports (safe now that the app is running)
# ===========================================================================
import os
import sys

import torch
import yaml
import zerorpc
from dataclasses import dataclass, field
from typing import List, Optional

# Make project root importable regardless of cwd
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mppi_torch.mppi import MPPIConfig
from isaaclab_mpc.planner.mppi_isaaclab import MPPIIsaacLabPlanner
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabConfig
from robots.ur16e import UR16E_CFG
from examples.ur16e_reach_stand.scene import make_static_cfgs


# ===========================================================================
# 3. Config loading
# ===========================================================================

@dataclass
class IsaacLabCfg:
    dt: float = 1.0 / 60.0
    visualize_rollouts: bool = True


@dataclass
class PlannerConfig:
    n_steps: int = 10000
    nx: int = 12
    goal: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.6])
    ee_link_name: str = "wrist_3_link"
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)


def _load_config(yaml_path: str) -> PlannerConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cfg = PlannerConfig()
    cfg.n_steps = raw.get("n_steps", cfg.n_steps)
    cfg.nx = raw.get("nx", cfg.nx)
    cfg.goal = raw.get("goal", cfg.goal)
    cfg.ee_link_name = raw.get("ee_link_name", cfg.ee_link_name)

    if "mppi" in raw:
        mppi_raw = raw["mppi"]
        cfg.mppi = MPPIConfig(**{k: v for k, v in mppi_raw.items()})

    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(
            dt=il.get("dt", 1.0 / 60.0),
            visualize_rollouts=il.get("visualize_rollouts", True),
        )

    return cfg


# ===========================================================================
# 4. Objective (cost function)
# ===========================================================================

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (w, x, y, z convention)."""
    w, x, y, z = q.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return torch.stack([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ], dim=-1)


class Objective:
    """
    Reach cost: penalise distance from pipe-nipple TCP tip to goal.

    The TCP tip is 0.14 m along +Z of wrist_3_link (tool0 is merged into
    wrist_3_link by merge_fixed_joints=True; the pipe nipple extends 0.14 m
    along that link's local Z axis).
    """

    TCP_OFFSET_LOCAL = torch.tensor([0.0, 0.0, 0.14])

    def __init__(self, cfg: PlannerConfig):
        self.weights = {
            "ee_to_goal": 1.0,
        }

    def reset(self):
        pass

    def compute_cost(self, sim) -> torch.Tensor:
        ee_pos  = sim.get_ee_pos()   # (num_envs, 3)
        ee_quat = sim.get_ee_quat()  # (num_envs, 4)
        goal    = sim.get_goal()     # (3,)

        tcp_offset = self.TCP_OFFSET_LOCAL.to(sim.device).expand(sim.num_envs, 3)
        tcp_pos = ee_pos + _quat_apply(ee_quat, tcp_offset)  # (num_envs, 3)

        dist = torch.linalg.norm(tcp_pos - goal.unsqueeze(0), dim=1)
        return self.weights["ee_to_goal"] * dist


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)

    objective = Objective(cfg)
    planner = MPPIIsaacLabPlanner(
        cfg,
        objective,
        robot_cfg=UR16E_CFG,
        prior=None,
        static_cfgs=make_static_cfgs(),
    )

    server = zerorpc.Server(planner)
    server.bind("tcp://0.0.0.0:4242")
    print("[planner] MPPI server listening on tcp://0.0.0.0:4242")
    server.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
