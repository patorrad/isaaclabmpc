"""MPPI planner server for the UR16e push task (Isaac Lab backend).

The robot must push a box to a goal position on the table.
Cost function mirrors genesismpc/examples/ur5_stick_push/planner.py:
  - robot_to_block: TCP distance to box
  - block_to_goal:  box distance to goal
  - block_height:   keep TCP at box height
  - push_align:     TCP-to-box direction should align with box-to-goal direction
  - robot_ori:      keep EE upright (wrist orientation penalty)

Run with:
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_push/planner.py
"""

# ===========================================================================
# 1. Simulator bootstrap — must happen first
# ===========================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR16e MPPI push planner")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. All other imports
# ===========================================================================
import os
import sys

import torch
import yaml
import zerorpc
from dataclasses import dataclass, field
from typing import List

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mppi_torch.mppi import MPPIConfig
from isaaclab_mpc.planner.mppi_isaaclab import MPPIIsaacLabPlanner
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabConfig
from robots.ur16e import UR16E_CFG
from examples.ur16e_push.box_cfg import make_box_cfg


# ===========================================================================
# 3. Config
# ===========================================================================

@dataclass
class IsaacLabCfg:
    dt: float = 1.0 / 60.0
    visualize_rollouts: bool = True


@dataclass
class BoxCfgEntry:
    init_pos: List[float] = field(default_factory=lambda: [0.4, 0.0, 0.025])
    size: List[float] = field(default_factory=lambda: [0.05, 0.05, 0.05])
    mass: float = 0.5


@dataclass
class PlannerConfig:
    n_steps: int = 10000
    nx: int = 12
    goal: List[float] = field(default_factory=lambda: [0.6, 0.3, 0.05])
    ee_link_name: str = "wrist_3_link"
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)
    boxes: List[BoxCfgEntry] = field(default_factory=list)


def _load_config(yaml_path: str) -> PlannerConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    cfg = PlannerConfig()
    cfg.n_steps      = raw.get("n_steps",      cfg.n_steps)
    cfg.nx           = raw.get("nx",           cfg.nx)
    cfg.goal         = raw.get("goal",         cfg.goal)
    cfg.ee_link_name = raw.get("ee_link_name", cfg.ee_link_name)
    if "mppi" in raw:
        cfg.mppi = MPPIConfig(**{k: v for k, v in raw["mppi"].items()})
    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(
            dt=il.get("dt", 1.0 / 60.0),
            visualize_rollouts=il.get("visualize_rollouts", True),
        )
    cfg.boxes = [
        BoxCfgEntry(
            init_pos=b.get("init_pos", [0.4, 0.0, 0.025]),
            size=b.get("size", [0.05, 0.05, 0.05]),
            mass=b.get("mass", 0.5),
        )
        for b in raw.get("boxes", [])
    ]
    return cfg


# ===========================================================================
# 4. Objective
# ===========================================================================

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by q (w,x,y,z). Supports batched inputs (..., 4) and (..., 3)."""
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
    """Push cost: move box to goal.

    Mirrors genesismpc/examples/ur5_stick_push/planner.py Objective.
    """

    TCP_OFFSET_LOCAL = torch.tensor([0.0, 0.0, 0.14])  # wrist_3_link → tool tip

    def __init__(self):
        self.weights = {
            "robot_to_block": 45.0,
            "block_to_goal":  25.0,
            "robot_ori":       5.0,
            "block_height":   20.0,
            "push_align":     45.0,
        }

    def reset(self):
        pass

    def compute_cost(self, sim) -> torch.Tensor:
        device = sim.device

        # EE link pose (local frame)
        ee_pos  = sim.get_ee_pos()   # (num_envs, 3)
        ee_quat = sim.get_ee_quat()  # (num_envs, 4)

        # TCP position (shift EE origin by rotated tool offset)
        tcp_offset = self.TCP_OFFSET_LOCAL.to(device).expand(sim.num_envs, 3)
        tcp_pos = ee_pos + _quat_apply(ee_quat, tcp_offset)  # (num_envs, 3)

        # Box and goal positions (local frame)
        box_pos  = sim.get_object_pos()  # (num_envs, 3)
        goal_pos = sim.get_goal()        # (3,)

        robot_to_block = tcp_pos - box_pos                # (num_envs, 3)
        block_to_goal  = goal_pos.unsqueeze(0) - box_pos  # (num_envs, 3)

        # --- distance costs ---
        robot_to_block_dist = torch.linalg.norm(robot_to_block, dim=1)
        block_to_goal_dist  = torch.linalg.norm(block_to_goal,  dim=1)

        # --- height cost: keep TCP at box Z ---
        block_height = torch.abs(tcp_pos[:, 2] - box_pos[:, 2])

        # --- push alignment: tcp→box should point same dir as box→goal (2‑D) ---
        r2b_2d = robot_to_block[:, :2]
        b2g_2d = block_to_goal[:, :2]
        r2b_2d_norm = torch.linalg.norm(r2b_2d, dim=1).clamp(min=1e-6)
        b2g_2d_norm = torch.linalg.norm(b2g_2d, dim=1).clamp(min=1e-6)
        push_align = (
            torch.sum(r2b_2d * b2g_2d, dim=1) / (r2b_2d_norm * b2g_2d_norm) + 1.0
        )

        # --- orientation cost: penalise wrist roll/pitch away from identity ---
        # Use the Z-component of the EE quaternion as a cheap uprightness proxy.
        # (A fully upright wrist_3_link has qz ≈ 0 in the home config.)
        robot_ori = torch.abs(ee_quat[:, 3])  # |qz|

        # --- NaN guard ---
        for t in (robot_to_block_dist, block_to_goal_dist, block_height, push_align, robot_ori):
            t[torch.isnan(t)] = 100.0

        return (
            self.weights["robot_to_block"] * robot_to_block_dist
            + self.weights["block_to_goal"]  * block_to_goal_dist
            + self.weights["robot_ori"]      * robot_ori
            + self.weights["block_height"]   * block_height
            + self.weights["push_align"]     * push_align
        )


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)

    object_cfgs = [make_box_cfg(b.size, b.mass, b.init_pos) for b in cfg.boxes]

    objective = Objective()
    planner = MPPIIsaacLabPlanner(
        cfg,
        objective,
        robot_cfg=UR16E_CFG,
        prior=None,
        object_cfgs=object_cfgs,
    )

    server = zerorpc.Server(planner)
    server.bind("tcp://0.0.0.0:4242")
    print("[planner] MPPI push server listening on tcp://0.0.0.0:4242")
    server.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
