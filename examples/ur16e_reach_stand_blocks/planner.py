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
from robots.ur16e import make_ur16e_cfg
from examples.ur16e_reach_stand_blocks.scene import make_static_cfgs, make_block_cfgs


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
    solution_path: str = "solution_obs_3_simple_extraction_robot.json"
    step_threshold: float = 0.04
    robot_init_pos: List[float] = field(default_factory=lambda: [0.208, 0.0, 2.075])
    robot_init_joints: List[float] = field(default_factory=lambda: [0.549, -2.2557, 1.0872, 0.8265, 1.5802, 0.5275])
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)


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

    return cfg


# ===========================================================================
# 4. Objective (cost function)
# ===========================================================================

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) v by quaternion(s) q  (w, x, y, z convention).

    Supports batched (num_envs, 4) × (num_envs, 3) → (num_envs, 3).
    """
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


def _quat_to_yaw_pitch(q: torch.Tensor) -> torch.Tensor:
    """Extract ZYX yaw and pitch from (num_envs, 4) wxyz quaternion.

    Returns (num_envs, 2) tensor [yaw, pitch].
    Used to penalise wrist tilt — keeping the tool pointing downward.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw   = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return torch.stack([yaw, pitch], dim=1)


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

    TCP_OFFSET_LOCAL = torch.tensor([0.0, 0.0, 0.12])

    def __init__(self, cfg: PlannerConfig):
        self.weights = {
            # "robot_to_obj": 30.0,
            # "obj_to_goal":  40.0,
            # "robot_ori":    10.0,
            # "push_align":   20.0,
            # "height_match": 20.0,
            # "collision":     2.0,
            # "robot_to_obj": 30.0,
            # "obj_to_goal":  40.0,
            # "robot_ori":     15.0,
            # "height_match": 20.0,
            # "push_align":   40.0,
            # "collision":     1.0,
            "robot_to_obj":  5.0,
            "obj_to_goal":   25.0,
            "robot_ori":      5.0,
            "height_match":  40.0,
            "push_align":    45.0,
            "collision":      .0,
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

        # if not self._printed_initial_poses:
        #     print("[Objective] Initial object poses in simulation (env 0):")
        #     for i in range(len(self.steps)):
        #         obj_idx = self.steps[i]["obj_idx"]
        #         pos = sim.get_object_pos(obj_idx)[0].tolist()
        #         print(f"  {self.steps[i]['obj_name']} (idx {obj_idx}): {[round(v,4) for v in pos]}")
        #     self._printed_initial_poses = True
        # [Objective] Initial object poses in simulation (env 0):
        #             obstacle_2 (idx 3): [0.6095, -0.0735, 0.595]
        #             obstacle_0 (idx 1): [0.4825, 0.0874, 0.595]
        #             target (idx 0): [0.6127, 0.0797, 0.595]
        #             target (idx 0): [0.6127, 0.0797, 0.595]
        # TCP tip position
        ee_pos  = sim.get_ee_pos()   # (num_envs, 3)
        ee_quat = sim.get_ee_quat()  # (num_envs, 4)
        tcp_offset = self.TCP_OFFSET_LOCAL.to(device).expand(sim.num_envs, 3)
        tcp_pos = ee_pos + _quat_apply(ee_quat, tcp_offset)  # (num_envs, 3)

        # All steps done — zero cost
        if self.current_step >= len(self.steps):
            return torch.zeros(sim.num_envs, device=device)

        step = self.steps[self.current_step]
        obj_idx  = step["obj_idx"]
        goal_pos = torch.tensor(step["end_pos"], dtype=torch.float32, device=device)
        # end_pos is in IsaacLab local (env-relative) frame — same frame as
        # get_object_pos() — so no coordinate conversion is needed here.

        obj_pos = sim.get_object_pos(obj_idx)  # (num_envs, 3)

        # Cache real object position (env 0) for step-advance check in reset()
        if self._first_call:
            self._last_obj_pos = obj_pos[0].detach().clone()
            self._first_call = False

        robot_to_obj = tcp_pos - obj_pos                          # (num_envs, 3)
        obj_to_goal  = goal_pos.unsqueeze(0) - obj_pos            # (num_envs, 3)

        robot_to_obj_dist = torch.linalg.norm(robot_to_obj, dim=1)
        obj_to_goal_dist  = torch.linalg.norm(obj_to_goal,  dim=1)

        # Orientation: penalise wrist yaw + pitch (keep tool pointing down)
        yaw_pitch = _quat_to_yaw_pitch(ee_quat)                  # (num_envs, 2)
        robot_ori = torch.linalg.norm(yaw_pitch, dim=1)

        # Height match: TCP Z ≈ block Z
        height_match = torch.abs(tcp_pos[:, 2] - obj_pos[:, 2])

        # Push alignment: 0 when TCP is directly behind block, 2 when in front
        r2b_2d = robot_to_obj[:, :2]
        b2g_2d = obj_to_goal[:, :2]
        push_align = (
            torch.sum(r2b_2d * b2g_2d, dim=1)
            / (torch.linalg.norm(r2b_2d, dim=1).clamp(min=1e-6)
               * torch.linalg.norm(b2g_2d, dim=1).clamp(min=1e-6))
            + 1.0
        )

        forces    = sim.get_contact_forces(0)           # (num_envs, 1, 3)
        collision = torch.abs(forces[:, 0, 2])          # Z component

        for t in (robot_to_obj_dist, obj_to_goal_dist, robot_ori,
                  height_match, push_align, collision):
            t[torch.isnan(t)] = 100.0

        return (
            self.weights["robot_to_obj"] * robot_to_obj_dist
            + self.weights["obj_to_goal"]  * obj_to_goal_dist
            + self.weights["robot_ori"]    * robot_ori
            + self.weights["push_align"]   * push_align
            + self.weights["height_match"] * height_match
            + self.weights["collision"]    * collision
        )


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
        static_cfgs=make_static_cfgs(),
        contact_sensor_cfgs=[robot_contact_sensor],
    )

    server = zerorpc.Server(planner)
    server.bind("tcp://0.0.0.0:4242")
    print("[planner] Stacked-blocks MPPI server listening on tcp://0.0.0.0:4242")
    server.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
