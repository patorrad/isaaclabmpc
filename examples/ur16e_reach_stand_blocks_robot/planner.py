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

import json

from mppi_torch.mppi import MPPIConfig
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import RigidBodyPropertiesCfg
from isaaclab_mpc.planner.mppi_isaaclab import MPPIIsaacLabPlanner
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabConfig
from isaaclab_mpc.cost import (
    DistCost, OrientationCost, HeightMatchCost, PushAlignCost,
    ContactForceCost, JointVelCost,
)
from isaaclab_mpc.cost.utils import quat_apply
from assets.robots.ur16e import make_ur16e_cfg
from examples.ur16e_reach_stand_blocks_robot.scene import make_static_cfgs, make_block_cfgs


# ===========================================================================
# 3. Config loading
# ===========================================================================

@dataclass
class IsaacLabCfg:
    dt: float = 1.0 / 60.0
    visualize_rollouts: bool = True


@dataclass
class CostWeights:
    robot_to_obj: float = 5.0
    obj_to_goal:  float = 25.0
    robot_ori:    float = 5.0
    height_match: float = 20.0
    push_align:   float = 45.0
    collision:    float = 1.0
    joint_vel:    float = 2.25


@dataclass
class CostConfig:
    weights: CostWeights = field(default_factory=CostWeights)
    push_align_gate_width: float = 0.03


@dataclass
class PlannerConfig:
    n_steps: int = 10000
    nx: int = 12
    goal: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.6])
    ee_link_name: str = "wrist_3_link"
    solution_path: str = "solution_obs_3_simple_extraction_robot.json"
    step_threshold: float = 0.02
    stand_urdf: str = ""
    robot_init_pos: List[float] = field(default_factory=lambda: [0.208, 0.0, 2.075])
    robot_init_joints: List[float] = field(default_factory=lambda: [0.549, -2.2557, 1.0872, 0.8265, 1.5802, 0.5275])
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)
    costs: CostConfig = field(default_factory=CostConfig)


def _load_config(yaml_path: str) -> PlannerConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cfg = PlannerConfig()
    cfg.n_steps         = raw.get("n_steps",         cfg.n_steps)
    cfg.nx              = raw.get("nx",              cfg.nx)
    cfg.goal            = raw.get("goal",            cfg.goal)
    cfg.ee_link_name    = raw.get("ee_link_name",    cfg.ee_link_name)
    cfg.solution_path   = raw.get("solution_path",   cfg.solution_path)
    cfg.step_threshold  = raw.get("step_threshold",  cfg.step_threshold)
    cfg.stand_urdf      = raw.get("stand_urdf",      cfg.stand_urdf)
    cfg.robot_init_pos    = raw.get("robot_init_pos",    cfg.robot_init_pos)
    cfg.robot_init_joints = raw.get("robot_init_joints", cfg.robot_init_joints)

    if "mppi" in raw:
        cfg.mppi = MPPIConfig(**{k: v for k, v in raw["mppi"].items()})

    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(
            dt=il.get("dt", 1.0 / 60.0),
            visualize_rollouts=il.get("visualize_rollouts", True),
        )

    if "costs" in raw:
        c = raw["costs"]
        if "weights" in c:
            cfg.costs.weights = CostWeights(**{k: float(v) for k, v in c["weights"].items()})
        if "push_align_gate_width" in c:
            cfg.costs.push_align_gate_width = float(c["push_align_gate_width"])

    return cfg


# ===========================================================================
# 4. Objective (cost function)
# ===========================================================================

class Objective:
    """Multi-step sequential block-push objective.

    Mirrors genesismpc/examples/ur5_stick_stacked_blocks_stand/planner.py
    but uses Isaac Lab data accessors.

    Steps are loaded from a JSON solution file.  Each step specifies which
    block (obj_idx) to push and where (end_pos).  When the block is close
    enough to its goal the objective advances to the next step.

    Cost terms (same weights as genesismpc):
      robot_to_obj  — TCP tip distance to the current block
      obj_to_goal   — block distance to its goal position
      robot_ori     — wrist yaw+pitch deviation from upright (tool pointing down)
      height_match  — TCP Z matches block Z (push at the right height)
      push_align    — TCP is behind the block relative to the push direction
    """

    TCP_OFFSET_LOCAL = torch.tensor([0.0, 0.0, 0.115])

    def __init__(self, cfg: PlannerConfig):
        w = cfg.costs.weights
        self.weights = {
            "robot_to_obj": w.robot_to_obj,
            "obj_to_goal":  w.obj_to_goal,
            "robot_ori":    w.robot_ori,
            "height_match": w.height_match,
            "push_align":   w.push_align,
            "collision":    w.collision,
            "joint_vel":    w.joint_vel,
        }
        self._costs = {
            "robot_to_obj": DistCost(),
            "obj_to_goal":  DistCost(),
            "robot_ori":    OrientationCost(),
            "height_match": HeightMatchCost(),
            "push_align":   PushAlignCost(align_gate_dist=0.08,
                                          gate_width=cfg.costs.push_align_gate_width),
            "collision":    ContactForceCost(),
            "joint_vel":    JointVelCost(),
        }
        self.step_threshold = cfg.step_threshold

        with open(cfg.solution_path) as f:
            solution = json.load(f)

        self.steps = solution["steps"]
        self.current_step = 0
        self._last_obj_pos: Optional[torch.Tensor] = None
        self._first_call = True
        self._printed_initial_poses = False

        print(f"[Objective] Loaded {len(self.steps)} steps from {cfg.solution_path}")
        for i, step in enumerate(self.steps):
            print(f"  Step {i}: push {step['obj_name']} (idx {step['obj_idx']}) → {step['end_pos']}")

        final_poses = {}
        for step in self.steps:
            final_poses[step['obj_name']] = (step['obj_idx'], step['end_pos'])
        print("[Objective] Final object world poses:")
        for name, (idx, pos) in final_poses.items():
            print(f"  {name} (idx {idx}): {pos}")

    def reset(self):
        """Advance to next step if current block reached its goal."""
        if self._last_obj_pos is not None and self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            goal = torch.tensor(step["end_pos"], dtype=torch.float32)
            dist = torch.linalg.norm(self._last_obj_pos.cpu() - goal).item()
            if dist < self.step_threshold:
                self.current_step += 1
                if self.current_step < len(self.steps):
                    ns = self.steps[self.current_step]
                    print(f"\n[Step {self.current_step}/{len(self.steps)}] "
                          f"now pushing {ns['obj_name']} → {ns['end_pos']}")
                else:
                    print(f"\n[Step] All {len(self.steps)} steps completed!")
        self._first_call = True

    def compute_cost(self, sim) -> torch.Tensor:
        device = sim.device

        # TCP tip position
        ee_pos  = sim.get_ee_pos()   # (num_envs, 3)
        ee_quat = sim.get_ee_quat()  # (num_envs, 4)
        tcp_offset = self.TCP_OFFSET_LOCAL.to(device).expand(sim.num_envs, 3)
        tcp_pos = ee_pos + quat_apply(ee_quat, tcp_offset)  # (num_envs, 3)

        # All steps done — zero cost
        if self.current_step >= len(self.steps):
            return torch.zeros(sim.num_envs, device=device)

        step = self.steps[self.current_step]
        obj_idx  = step["obj_idx"]
        goal_pos = torch.tensor(step["end_pos"], dtype=torch.float32, device=device)
        sim.set_goal(goal_pos)

        obj_pos = sim.get_object_pos(obj_idx)  # (num_envs, 3)

        # Cache real object position (env 0) for step-advance check in reset()
        if self._first_call:
            self._last_obj_pos = obj_pos[0].detach().clone()
            self._first_call = False

        robot_to_obj      = tcp_pos - obj_pos                 # (num_envs, 3)
        obj_to_goal       = goal_pos.unsqueeze(0) - obj_pos   # (num_envs, 3)
        robot_to_obj_dist = self._costs["robot_to_obj"](robot_to_obj)

        raw = {
            "robot_to_obj": robot_to_obj_dist,
            "obj_to_goal":  self._costs["obj_to_goal"](obj_to_goal),
            "robot_ori":    self._costs["robot_ori"](ee_quat),
            "height_match": self._costs["height_match"](tcp_pos[:, 2], obj_pos[:, 2]),
            "push_align":   self._costs["push_align"](robot_to_obj, obj_to_goal, robot_to_obj_dist),
            "collision":    self._costs["collision"](sim.get_contact_forces(0)),
            "joint_vel":    self._costs["joint_vel"](sim.get_joint_vel()),
        }

        for t in raw.values():
            t[torch.isnan(t)] = 100.0

        return sum(self.weights[k] * v for k, v in raw.items())


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)

    robot_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
    )

    _base_robot_cfg = make_ur16e_cfg(pos=cfg.robot_init_pos, joint_pos=cfg.robot_init_joints)
    robot_cfg = _base_robot_cfg.replace(
        spawn=_base_robot_cfg.spawn.replace(
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
            activate_contact_sensors=True,
        )
    )

    objective = Objective(cfg)
    planner = MPPIIsaacLabPlanner(
        cfg,
        objective,
        robot_cfg=robot_cfg,
        prior=None,
        object_cfgs=make_block_cfgs(),
        static_cfgs=make_static_cfgs(stand_urdf=cfg.stand_urdf),
        contact_sensor_cfgs=[robot_contact_sensor],
    )

    server = zerorpc.Server(planner)
    server.bind("tcp://0.0.0.0:4242")
    print("[planner] Blocks MPPI server listening on tcp://0.0.0.0:4242")
    server.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
