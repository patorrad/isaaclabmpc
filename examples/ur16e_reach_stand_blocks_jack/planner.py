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
parser.add_argument("--solution_path", type=str, default=None,
                    help="Path to puzzle solution JSON. Overrides cfg.solution_path.")
parser.add_argument("--telemetry_path", type=str, default=None,
                    help="If set, write MPPI cost history and step events as JSON on exit.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True          # planner always runs headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. All other imports (safe now that the app is running)
# ===========================================================================
import json
import os
import signal
import sys
import time

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

# Module-level reference set in main() so the SIGTERM handler can write telemetry.
_objective_ref = None

from mppi_torch.mppi import MPPIConfig
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import RigidBodyPropertiesCfg
from isaaclab_mpc.planner.mppi_isaaclab import MPPIIsaacLabPlanner
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabConfig
from isaaclab_mpc.cost import (
    DistCost, OrientationCost, HeightMatchCost, PushAlignCost,
    ContactForceCost, JointVelCost, SingularityCost,
)
from isaaclab_mpc.cost.utils import quat_apply
from assets.robots.ur16e import make_ur16e_cfg, get_tool_length
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
class CostWeights:
    robot_to_obj: float = 5.0
    obj_to_goal:  float = 25.0
    robot_ori:    float = 15.0
    height_match: float = 20.0
    push_align:   float = 45.0
    collision:    float = 1.0
    joint_vel:    float = 0.0
    singularity:  float = 0.05


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
    scenario: Optional[str] = None
    step_threshold: float = 0.02
    stand_urdf: str = _STAND_URDF_PATH
    robot_init_pos: List[float] = field(default_factory=lambda: [0.208, 0.0, 2.075])
    robot_init_joints: List[float] = field(default_factory=lambda: [0.549, -2.2557, 1.0872, 0.8265, 1.5802, 0.5275])
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)
    costs: CostConfig = field(default_factory=CostConfig)


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

    def __init__(self, cfg: PlannerConfig, debug: bool = False):
        w = cfg.costs.weights
        self.weights = {
            "robot_to_obj": w.robot_to_obj,
            "obj_to_goal":  w.obj_to_goal,
            "robot_ori":    w.robot_ori,
            "height_match": w.height_match,
            "push_align":   w.push_align,
            "collision":    w.collision,
            "joint_vel":    w.joint_vel,
            "singularity":  w.singularity,
        }
        self.step_threshold = cfg.step_threshold
        self.tcp_offset_local = torch.tensor([0.0, 0.0, get_tool_length()])
        self._t_start = time.time()
        self._cost_history: list[float] = []
        self._step_events: list[dict] = []

        with open(cfg.solution_path) as f:
            solution = json.load(f)

        self.steps = solution["steps"]
        obj_size = solution.get("env_config", {}).get("OBJ_SIZE", 0.05)
        self.align_gate_dist = obj_size / 2 + 0.01  # back face + 1 cm standoff
        self.current_step = 0
        self._last_obj_pos: Optional[torch.Tensor] = None
        self._first_call = True
        self._printed_initial_poses = False

        self._costs = {
            "robot_to_obj": DistCost(),
            "obj_to_goal":  DistCost(),
            "robot_ori":    OrientationCost(),
            "height_match": HeightMatchCost(),
            "push_align":   PushAlignCost(align_gate_dist=self.align_gate_dist,
                                          gate_width=cfg.costs.push_align_gate_width),
            "collision":    ContactForceCost(),
            "joint_vel":    JointVelCost(),
            "singularity":  SingularityCost(),
        }

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
                self._step_events.append({
                    "step_idx": self.current_step,
                    "elapsed_s": time.time() - self._t_start,
                })
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
        tcp_offset = self.tcp_offset_local.to(device).expand(sim.num_envs, 3)
        tcp_pos = ee_pos + quat_apply(ee_quat, tcp_offset)  # (num_envs, 3)

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
            "singularity":  self._costs["singularity"](sim.get_ee_jacobian()),
        }

        for t in raw.values():
            t[torch.isnan(t)] = 100.0

        if self._debug_capture:
            self._debug_last_terms = [
                (self.weights[k] * raw[k][0]).item()
                for k in self._debug_term_keys
            ]
            self._debug_capture = False

        cost = sum(self.weights[k] * v for k, v in raw.items())
        self._cost_history.append(float(cost.min().item()))
        return cost


# ===========================================================================
# 5. Main
# ===========================================================================

def _write_telemetry(path: str, objective: 'Objective') -> None:
    data = {
        "mppi_cost_history": objective._cost_history,
        "step_completion_events": objective._step_events,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[planner] Telemetry written to {path}")


def main():
    global _objective_ref

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)

    # Apply CLI overrides
    if args_cli.solution_path:
        cfg.solution_path = args_cli.solution_path

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
    _objective_ref = objective

    def _sigterm_handler(*_):
        if args_cli.telemetry_path and _objective_ref is not None:
            _write_telemetry(args_cli.telemetry_path, _objective_ref)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

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
