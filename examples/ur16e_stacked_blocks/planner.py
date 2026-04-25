"""MPPI planner server for the UR16e stacked-blocks push task (Isaac Lab backend).

Sequential multi-object push: the robot pushes objects one at a time to their
goal positions, in the order defined by 'steps' in config.yaml.  When the
current object reaches its goal the planner automatically advances to the next
step and updates the goal marker.

Run with:
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_stacked_blocks/planner.py
"""

# ===========================================================================
# 1. Simulator bootstrap — must happen first
# ===========================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR16e stacked-blocks MPPI planner")
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
from typing import List, Optional

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
    n_steps: int = 20000
    nx: int = 12
    ee_link_name: str = "wrist_3_link"
    step_threshold: float = 0.04
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)
    boxes: List[BoxCfgEntry] = field(default_factory=list)
    steps: List[dict] = field(default_factory=list)


def _load_config(yaml_path: str) -> PlannerConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    cfg = PlannerConfig()
    cfg.n_steps        = raw.get("n_steps",        cfg.n_steps)
    cfg.nx             = raw.get("nx",             cfg.nx)
    cfg.ee_link_name   = raw.get("ee_link_name",   cfg.ee_link_name)
    cfg.step_threshold = raw.get("step_threshold", cfg.step_threshold)
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
    cfg.steps = raw.get("steps", [])
    return cfg


# ===========================================================================
# 4. Objective
# ===========================================================================

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by quaternion q (w,x,y,z). Supports batched (..., 4) / (..., 3)."""
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
    """Sequential push cost: move objects one at a time to their goal positions.

    Mirrors genesismpc/examples/ur5_stick_stacked_blocks_stand/planner.py
    but loads the step sequence from config.yaml instead of a JSON file.
    """

    TCP_OFFSET_LOCAL = torch.tensor([0.0, 0.0, 0.14])

    def __init__(self, steps: List[dict], step_threshold: float = 0.04):
        self.weights = {
            "robot_to_obj": 45.0,
            "obj_to_goal":  25.0,
            "robot_ori":     5.0,
            "obj_height":   20.0,
            "push_align":   45.0,
        }
        self.steps = steps
        self.current_step = 0
        self.step_threshold = step_threshold
        self._last_obj_pos: Optional[torch.Tensor] = None
        self._first_step = True
        self.sim = None   # injected after MPPIIsaacLabPlanner creates self.sim

    def reset(self):
        """Advance step if the current object has reached its goal."""
        if self._last_obj_pos is not None and self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            goal = torch.tensor(step["end_pos"], dtype=torch.float32)
            dist = torch.linalg.norm(self._last_obj_pos.cpu() - goal).item()
            if dist < self.step_threshold:
                self.current_step += 1
                if self.current_step < len(self.steps):
                    nxt = self.steps[self.current_step]
                    new_goal = torch.tensor(nxt["end_pos"], dtype=torch.float32)
                    if self.sim is not None:
                        self.sim.set_goal(new_goal)
                    print(
                        f"\n[Objective] Step {self.current_step}/{len(self.steps)}: "
                        f"obj_idx={nxt['obj_idx']} → {nxt['end_pos']}",
                        flush=True,
                    )
                else:
                    print(
                        f"\n[Objective] All {len(self.steps)} steps completed!",
                        flush=True,
                    )
        self._first_step = True

    def compute_cost(self, sim) -> torch.Tensor:
        device = sim.device

        if self.current_step >= len(self.steps):
            return torch.zeros(sim.num_envs, device=device)

        step    = self.steps[self.current_step]
        obj_idx = step["obj_idx"]

        # EE pose → TCP position
        ee_pos  = sim.get_ee_pos()   # (num_envs, 3)
        ee_quat = sim.get_ee_quat()  # (num_envs, 4)
        tcp_offset = self.TCP_OFFSET_LOCAL.to(device).expand(sim.num_envs, 3)
        tcp_pos = ee_pos + _quat_apply(ee_quat, tcp_offset)

        # Active object and goal
        obj_pos  = sim.get_object_pos(obj_idx)   # (num_envs, 3)
        goal_pos = torch.tensor(step["end_pos"], dtype=torch.float32, device=device)

        # Capture real position on the first call of each planning cycle
        if self._first_step:
            self._last_obj_pos = obj_pos[0].detach().clone()
            self._first_step = False

        robot_to_obj = tcp_pos - obj_pos
        obj_to_goal  = goal_pos.unsqueeze(0) - obj_pos

        # Distance costs
        robot_to_obj_dist = torch.linalg.norm(robot_to_obj, dim=1)
        obj_to_goal_dist  = torch.linalg.norm(obj_to_goal,  dim=1)

        # Height cost: keep TCP at object Z
        obj_height = torch.abs(tcp_pos[:, 2] - obj_pos[:, 2])

        # Push alignment (2-D): tcp→obj should align with obj→goal
        r2o_2d = robot_to_obj[:, :2]
        o2g_2d = obj_to_goal[:, :2]
        r2o_norm = torch.linalg.norm(r2o_2d, dim=1).clamp(min=1e-6)
        o2g_norm = torch.linalg.norm(o2g_2d, dim=1).clamp(min=1e-6)
        push_align = torch.sum(r2o_2d * o2g_2d, dim=1) / (r2o_norm * o2g_norm) + 1.0

        # Orientation cost: |qz| penalty (cheap uprightness proxy)
        robot_ori = torch.abs(ee_quat[:, 3])

        # NaN guard
        for t in (robot_to_obj_dist, obj_to_goal_dist, obj_height, push_align, robot_ori):
            t[torch.isnan(t)] = 100.0

        return (
            self.weights["robot_to_obj"] * robot_to_obj_dist
            + self.weights["obj_to_goal"]  * obj_to_goal_dist
            + self.weights["robot_ori"]    * robot_ori
            + self.weights["obj_height"]   * obj_height
            + self.weights["push_align"]   * push_align
        )


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)

    assert len(cfg.boxes) == 4, "config.yaml must define exactly 4 boxes"
    assert len(cfg.steps) > 0,  "config.yaml must define at least one step"

    # MPPIIsaacLabPlanner expects cfg.goal to seed the sim's initial goal
    cfg.goal = cfg.steps[0]["end_pos"]

    object_cfgs = [make_box_cfg(b.size, b.mass, b.init_pos) for b in cfg.boxes]

    objective = Objective(cfg.steps, cfg.step_threshold)

    planner = MPPIIsaacLabPlanner(
        cfg,
        objective,
        robot_cfg=UR16E_CFG,
        prior=None,
        object_cfgs=object_cfgs,
    )

    # Inject sim reference so the objective can update the goal when steps advance
    objective.sim = planner.sim

    server = zerorpc.Server(planner)
    server.bind("tcp://0.0.0.0:4242")
    print("[planner] Stacked-blocks MPPI server listening on tcp://0.0.0.0:4242")
    for i, step in enumerate(cfg.steps):
        print(f"  Step {i}: obj_idx={step['obj_idx']} → {step['end_pos']}")
    server.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
