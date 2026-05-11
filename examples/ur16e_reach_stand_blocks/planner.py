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
parser.add_argument("--scenario", type=str, default=None,
                    help="Path to a puzzles YAML scenario file. "
                         "Overrides the hardcoded block positions in scene.py.")
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

import matplotlib.pyplot as plt
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
from assets.robots.ur16e import make_ur16e_cfg
from robots import STAND_URDF_PATH as _STAND_URDF_PATH
from examples.ur16e_reach_stand_blocks.scene import make_static_cfgs, make_block_cfgs, _bin_to_mppi_local


# ===========================================================================
# 3. Config loading
# ===========================================================================

@dataclass
class IsaacLabCfg:
    dt: float = 1.0 / 60.0
    visualize_rollouts: bool = True
    debug: bool = False


@dataclass
class PlannerConfig:
    n_steps: int = 10000
    nx: int = 12
    goal: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.6])
    ee_link_name: str = "wrist_3_link"
    solution_path: str = "solution_obs_3_simple_extraction_robot.json"
    scenario: Optional[str] = None
    step_threshold: float = 0.02
    stand_urdf: str = _STAND_URDF_PATH
    robot_init_pos: List[float] = field(default_factory=lambda: [0.208, 0.0, 2.075])
    robot_init_joints: List[float] = field(default_factory=lambda: [0.549, -2.2557, 1.0872, 0.8265, 1.5802, 0.5275])
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)


def _load_config(yaml_path: str) -> PlannerConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cfg_dir = os.path.dirname(os.path.abspath(yaml_path))

    def _resolve(p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        return p if os.path.isabs(p) else os.path.join(cfg_dir, p)

    cfg = PlannerConfig()
    cfg.n_steps         = raw.get("n_steps",         cfg.n_steps)
    cfg.nx              = raw.get("nx",              cfg.nx)
    cfg.goal            = raw.get("goal",            cfg.goal)
    cfg.ee_link_name    = raw.get("ee_link_name",    cfg.ee_link_name)
    cfg.solution_path   = _resolve(raw.get("solution_path", cfg.solution_path))
    cfg.scenario        = _resolve(raw.get("scenario",      cfg.scenario))
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
            debug=il.get("debug", False),
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

    TCP_OFFSET_LOCAL = torch.tensor([0.0, 0.0, 0.115])

    def __init__(self, cfg: PlannerConfig, debug: bool = False):
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
            # "robot_to_obj":  5.0,
            # "obj_to_goal":   25.0,
            # "robot_ori":      5.0,
            # "height_match":  20.0,
            # "push_align":    45.0,
            # "collision":      1.0,
            # "joint_vel":      2.25, #.25,
            "robot_to_obj":  5.0,
            "obj_to_goal":   25.0,
            "robot_ori":      5.0,
            "height_match":  20.0,
            "push_align":    45.0,
            "collision":      1.0,
            "joint_vel":      0.,
            "singularity":    0.05,  # reciprocal manipulability; tune up to penalise near-singular configs
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

        self.debug = debug
        self._debug_term_keys = ["robot_to_obj", "obj_to_goal", "robot_ori",
                                  "height_match", "push_align", "collision", "joint_vel",
                                  "singularity"]
        self._debug_last_terms: list = [0.0] * len(self._debug_term_keys)
        self._debug_capture = False
        if debug:
            self._init_debug_plot()

    def _init_debug_plot(self):
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(9, 4))
        labels = [f"{k}\n(w={self.weights[k]})" for k in self._debug_term_keys]
        self._bars = self._ax.bar(labels, [0.0] * len(self._debug_term_keys), color="steelblue")
        self._ax.set_ylabel("Weighted cost (env 0)")
        self._ax.set_title("MPPI cost terms")
        self._fig.tight_layout()
        plt.show(block=False)

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
        self._debug_capture = self.debug

    def update_debug_plot(self):
        if not self.debug:
            return
        for bar, val in zip(self._bars, self._debug_last_terms):
            bar.set_height(val)
        self._ax.relim()
        self._ax.autoscale_view()
        step_info = (f"Step {self.current_step}/{len(self.steps)}"
                     if self.current_step < len(self.steps) else "All steps done")
        self._ax.set_title(f"MPPI cost terms — {step_info}")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

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
        sim.set_goal(goal_pos)

        obj_pos = sim.get_object_pos(obj_idx)  # (num_envs, 3)
        # import pdb; pdb.set_trace()
        # obj_pos = sim.get_object_states
        # print(f'{obj_idx} {obj_pos[0, :]}')
        # print(sim.get_object_states())

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

        # Push alignment: 0 when TCP is directly behind block, 2 when in front.
        # Gated by distance: fades to zero when TCP is already at the block so
        # the robot stops trying to reposition and just pushes.
        r2b_2d = robot_to_obj[:, :2]
        b2g_2d = obj_to_goal[:, :2]
        push_align = (
            torch.sum(r2b_2d * b2g_2d, dim=1)
            / (torch.linalg.norm(r2b_2d, dim=1).clamp(min=1e-6)
               * torch.linalg.norm(b2g_2d, dim=1).clamp(min=1e-6))
            + 1.0
        )
        align_gate = torch.sigmoid((robot_to_obj_dist - 0.08) / 0.03)
        push_align = push_align * align_gate

        forces    = sim.get_contact_forces(0)           # (num_envs, 1, 3)
        collision = torch.abs(forces[:, 0, 2])          # Z component

        joint_vel = torch.linalg.norm(sim.get_joint_vel(), dim=1)  # (num_envs,)

        J = sim.get_ee_jacobian()                          # (num_envs, 6, 6)
        det_J = torch.abs(torch.linalg.det(J))             # (num_envs,)
        singularity = 1.0 / (det_J + 1e-6)                # (num_envs,); large near singularity

        for t in (robot_to_obj_dist, obj_to_goal_dist, robot_ori,
                  height_match, push_align, collision, joint_vel,
                  singularity):
            t[torch.isnan(t)] = 100.0

        if self._debug_capture:
            raw_terms = [robot_to_obj_dist, obj_to_goal_dist, robot_ori,
                         height_match, push_align, collision, joint_vel,
                         singularity]
            self._debug_last_terms = [
                (self.weights[k] * t[0]).item()
                for k, t in zip(self._debug_term_keys, raw_terms)
            ]
            self._debug_capture = False

        return (
            self.weights["robot_to_obj"]  * robot_to_obj_dist
            + self.weights["obj_to_goal"]   * obj_to_goal_dist
            + self.weights["robot_ori"]     * robot_ori
            + self.weights["push_align"]    * push_align
            + self.weights["height_match"]  * height_match
            + self.weights["collision"]     * collision
            + self.weights["joint_vel"]     * joint_vel
            + self.weights["singularity"]   * singularity
        )


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)

    scenario_path = args_cli.scenario or cfg.scenario
    block_positions = None
    if scenario_path is not None:
        with open(scenario_path) as f:
            sc = yaml.safe_load(f)
        is_ = sc["initial_state"]
        bin_positions = [is_["target_pos"]] + [o["pos"] for o in is_["obstacles"]]
        block_positions = [_bin_to_mppi_local(p) for p in bin_positions]

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

    objective = Objective(cfg, debug=cfg.isaaclab.debug)
    planner = MPPIIsaacLabPlanner(
        cfg,
        objective,
        robot_cfg=robot_cfg,
        prior=None,
        object_cfgs=make_block_cfgs(positions=block_positions),
        static_cfgs=make_static_cfgs(stand_urdf=cfg.stand_urdf),
        contact_sensor_cfgs=[robot_contact_sensor],
    )

    server = zerorpc.Server(planner)
    server.bind("tcp://0.0.0.0:4242")
    print("[planner] Stacked-blocks MPPI server listening on tcp://0.0.0.0:4242")
    server.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
