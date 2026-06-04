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
parser.add_argument("--defer_solution", action="store_true",
                    help="Start without a solution; receive steps via reset_episode() RPC.")
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
import numpy as np
import matplotlib.pyplot as plt
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
    ContactForceCost, JointVelCost, SingularityCost, GaussianProjection,
    TcpFloorCost, AboveObjectCost,
)
from isaaclab_mpc.cost.utils import quat_apply
from assets.robots.ur16e import make_ur16e_cfg, get_tool_length
from examples.ur16e_stacked_robot.scene import make_static_cfgs, make_block_cfgs, _bin_to_mppi_local, _BLOCK_SPECS


# ===========================================================================
# 3. Config loading
# ===========================================================================

@dataclass
class CostWeights:
    robot_to_obj:  float = 0.
    obj_to_goal:   float = 0.0
    robot_ori:     float = 0.
    height_match:  float = 0.0
    push_align:    float = 0.0
    collision:     float = 0.
    joint_vel:     float = 0.
    singularity:   float = 0.0
    tcp_floor:     float = 0.0
    above_target:  float = 0.
    obj_avoidance: float = 0.0  # inverse-distance repulsion from all blocks (approach mode)


@dataclass
class GaussianProjParams:
    """Parameters for one cost term's Gaussian projection. c=0 → passthrough."""
    n: int   = 1
    c: float = 0.0
    s: float = 0.0
    r: float = 0.0


@dataclass
class GaussianProjectionConfig:
    """Per-cost Gaussian projection params. Inactive when enabled=False or c=0.

    n=2 → f(0)=0, f(∞)=1: ideal for distance/angle costs (zero error = zero cost).
    n=1 → f(0)=2, f(∞)=1: bump semantics (penalise proximity to a point).
    """
    enabled:      bool               = False
    robot_to_obj: GaussianProjParams = field(default_factory=lambda: GaussianProjParams(n=2, c=0.0, r=1e-5))
    obj_to_goal:  GaussianProjParams = field(default_factory=lambda: GaussianProjParams(n=2, c=0.0, r=1e-5))
    height_match: GaussianProjParams = field(default_factory=lambda: GaussianProjParams(n=2, c=0.0, r=1e-5))
    robot_ori:    GaussianProjParams = field(default_factory=lambda: GaussianProjParams(n=2, c=0.0, r=1e-5))
    joint_vel:    GaussianProjParams = field(default_factory=lambda: GaussianProjParams(n=2, c=0.0, r=1e-5))
    push_align:    GaussianProjParams = field(default_factory=GaussianProjParams)
    collision:     GaussianProjParams = field(default_factory=GaussianProjParams)
    singularity:   GaussianProjParams = field(default_factory=GaussianProjParams)
    tcp_floor:     GaussianProjParams = field(default_factory=GaussianProjParams)
    above_target:  GaussianProjParams = field(default_factory=GaussianProjParams)
    obj_avoidance: GaussianProjParams = field(default_factory=GaussianProjParams)


@dataclass
class CostConfig:
    weights: CostWeights = field(default_factory=CostWeights)
    push_align_gate_width: float = 0.03
    tcp_floor_offset: float = 0.05
    obj_half_size: float = 0.025
    gaussian_projection: GaussianProjectionConfig = field(default_factory=GaussianProjectionConfig)
    approach_push_align_threshold:   float = 0.3   # push_align below this → "aligned enough"
    approach_height_match_threshold: float = 0.05  # height_match below this → "at height"
    obj_avoidance_eps:               float = 0.1   # epsilon in 1/(d + eps) repulsion
    obj_avoidance_dist_threshold:    float = 0.3   # repulsion is zero beyond this distance (m)


@dataclass
class IsaacLabCfg:
    dt: float = 1.0 / 60.0
    visualize_rollouts: bool = True
    render: bool = False
    env_spacing: float = 1.5


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
    print(cfg.mppi)

    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(
            dt=il.get("dt", 1.0 / 60.0),
            visualize_rollouts=il.get("visualize_rollouts", True),
            render=not args_cli.headless,
            env_spacing=il.get("env_spacing", 1.5),
        )

    if "costs" in raw:
        c = raw["costs"]
        if "weights" in c:
            cfg.costs.weights = CostWeights(**{k: float(v) for k, v in c["weights"].items()})
        if "push_align_gate_width" in c:
            cfg.costs.push_align_gate_width = float(c["push_align_gate_width"])
        if "tcp_floor_offset" in c:
            cfg.costs.tcp_floor_offset = float(c["tcp_floor_offset"])
        if "gaussian_projection" in c:
            gp_raw = c["gaussian_projection"]
            gp = GaussianProjectionConfig()
            gp.enabled = bool(gp_raw.get("enabled", False))
            _cost_keys = ["robot_to_obj", "obj_to_goal", "robot_ori", "height_match",
                          "push_align", "joint_vel", "collision", "singularity", "tcp_floor",
                          "above_target", "obj_avoidance"]
            for key in _cost_keys:
                if key in gp_raw:
                    p = gp_raw[key]
                    setattr(gp, key, GaussianProjParams(
                        n=int(p.get("n", 1)),
                        c=float(p.get("c", 0.0)),
                        s=float(p.get("s", 0.0)),
                        r=float(p.get("r", 0.0)),
                    ))
            cfg.costs.gaussian_projection = gp
        if "approach_push_align_threshold" in c:
            cfg.costs.approach_push_align_threshold = float(c["approach_push_align_threshold"])
        if "approach_height_match_threshold" in c:
            cfg.costs.approach_height_match_threshold = float(c["approach_height_match_threshold"])
        if "obj_avoidance_eps" in c:
            cfg.costs.obj_avoidance_eps = float(c["obj_avoidance_eps"])
        if "obj_avoidance_dist_threshold" in c:
            cfg.costs.obj_avoidance_dist_threshold = float(c["obj_avoidance_dist_threshold"])

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

    _PLOT_INTERVAL = 50
    _EMA_ALPHA     = 0.05

    def __init__(self, cfg: PlannerConfig, table_surface_z: float = 0.0, steps_override: list | None = None):
        w = cfg.costs.weights
        self.weights = {
            "robot_to_obj":  w.robot_to_obj,
            "obj_to_goal":   w.obj_to_goal,
            "robot_ori":     w.robot_ori,
            "height_match":  w.height_match,
            "push_align":    w.push_align,
            "collision":     w.collision,
            "joint_vel":     w.joint_vel,
            "singularity":   w.singularity,
            "tcp_floor":     w.tcp_floor,
            "above_target":  w.above_target,
            "obj_avoidance": w.obj_avoidance,
        }
        self._approach_pa_threshold    = cfg.costs.approach_push_align_threshold
        self._approach_hm_threshold    = cfg.costs.approach_height_match_threshold
        self._obj_avoidance_eps        = cfg.costs.obj_avoidance_eps
        self._obj_avoidance_dist_thr   = cfg.costs.obj_avoidance_dist_threshold
        self._costs = {
            "robot_to_obj": DistCost(),
            "obj_to_goal":  DistCost(),
            "robot_ori":    OrientationCost(),
            "height_match": HeightMatchCost(),
            "push_align":   PushAlignCost(align_gate_dist=0.05,
                                          gate_width=cfg.costs.push_align_gate_width),
            "collision":    ContactForceCost(),
            "joint_vel":    JointVelCost(),
            "singularity":  SingularityCost(),
            "tcp_floor":    TcpFloorCost(threshold=table_surface_z + cfg.costs.tcp_floor_offset,
                                         table_surface_z=table_surface_z),
            "above_target": AboveObjectCost(obj_half_size=cfg.costs.obj_half_size)
        }
        print(f"[Objective] TcpFloorCost: table_surface_z={table_surface_z:.4f}, "
              f"offset={cfg.costs.tcp_floor_offset:.4f}, "
              f"threshold={table_surface_z + cfg.costs.tcp_floor_offset:.4f}")

        self._active_costs = {k for k, w in self.weights.items() if w != 0.0}
        _skipped = sorted(set(self.weights) - self._active_costs)
        if _skipped:
            print(f"[Objective] Skipping zero-weight costs: {_skipped}")

        gp_cfg = cfg.costs.gaussian_projection
        if gp_cfg.enabled:
            self._projections = {
                k: GaussianProjection(n=getattr(gp_cfg, k).n,
                                      c=getattr(gp_cfg, k).c,
                                      s=getattr(gp_cfg, k).s,
                                      r=getattr(gp_cfg, k).r)
                for k in self._active_costs
            }
            _active = [k for k in self._active_costs if getattr(gp_cfg, k).c != 0]
            print(f"[Objective] GaussianProjection enabled for: {_active}")
        else:
            self._projections = {}

        self.step_threshold = cfg.step_threshold

        if steps_override is not None:
            self.steps = steps_override
            self._obj_half_size = 0.025
        else:
            with open(cfg.solution_path) as f:
                solution = json.load(f)

            self.steps = solution["steps"]
            frame = solution.get("coordinate_frame", "robot")
            if frame == "bin":
                print("[Objective] coordinate_frame=bin — converting step positions via _bin_to_mppi_local")
                for step in self.steps:
                    step["end_pos"] = _bin_to_mppi_local(step["end_pos"])
                    if "start_pos" in step:
                        step["start_pos"] = _bin_to_mppi_local(step["start_pos"])
            elif frame == "target":
                self._target_frame_unresolved = True
                print("[Objective] coordinate_frame=target — will resolve positions on first state call")
            obj_size = solution.get("env_config", {}).get("OBJ_SIZE", 0.05)
            self._obj_half_size = obj_size / 2
            self._costs["above_target"] = AboveObjectCost(obj_half_size=self._obj_half_size)
        self.current_step = 0
        self._last_obj_pos: Optional[torch.Tensor] = None
        self._first_call = True
        self._is_push_mode = False   # locked for full rollout, re-evaluated each planning cycle
        self._printed_initial_poses = False

        self._labels = list(self.weights.keys())
        self._cost_avg = {k: 0.0 for k in self._labels}
        self._call_count = 0

        plt.ion()
        colors = ["steelblue", "tomato", "forestgreen", "goldenrod",
                  "mediumpurple", "darkorange", "teal", "sienna", "crimson", "darkviolet",
                  "olivedrab"]
        self._fig, self._ax = plt.subplots(figsize=(8, 4))
        self._fig.suptitle("Avg weighted cost per component (across trajectories)")
        self._bars = self._ax.bar(self._labels, [0.0] * len(self._labels),
                                  color=colors[:len(self._labels)])
        self._ax.set_ylabel("Avg weighted cost")
        self._ax.set_ylim(0, 20)
        plt.tight_layout()
        plt.show()

        self._last_total_costs: np.ndarray | None = None
        self._SPEC_BINS    = 60
        self._SPEC_MAX     = 30.0
        self._SPEC_HISTORY = 200
        self._cost_spec    = np.zeros((self._SPEC_BINS, self._SPEC_HISTORY))

        self._fig2, self._ax2 = plt.subplots(figsize=(10, 4))
        self._fig2.suptitle("Rollout cost distribution over time")
        self._im_spec = self._ax2.imshow(
            self._cost_spec, aspect="auto", origin="lower",
            cmap="inferno", interpolation="nearest",
            vmin=0, vmax=1,
            extent=[0, self._SPEC_HISTORY, 0, self._SPEC_MAX],
        )
        self._ax2.set_ylabel("Total rollout cost")
        self._ax2.set_xlabel("MPC step  (← older  |  newer →)")
        self._fig2.colorbar(self._im_spec, ax=self._ax2, label="Fraction of rollouts")
        self._fig2.tight_layout()
        self._fig2.show()

        try:
            from isaacsim.util.debug_draw import _debug_draw
            self._draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            self._draw = None

        src = "steps_override" if steps_override is not None else cfg.solution_path
        print(f"[Objective] Loaded {len(self.steps)} steps from {src}")
        for i, step in enumerate(self.steps):
            print(f"  Step {i}: push {step['obj_name']} (idx {step['obj_idx']}) → {step['end_pos']}")

        final_poses = {}
        for step in self.steps:
            final_poses[step['obj_name']] = (step['obj_idx'], step['end_pos'])
        print("[Objective] Final object world poses:")
        for name, (idx, pos) in final_poses.items():
            print(f"  {name} (idx {idx}): {pos}")

    def resolve_target_frame(self, target_pos_local: torch.Tensor) -> None:
        """Resolve target-relative step positions to absolute MPC-local positions.

        Called on the first compute_action_tensor tick that carries object states.
        No-ops after the first successful resolution.
        """
        if not getattr(self, '_target_frame_unresolved', False):
            return
        t = target_pos_local.tolist()
        for step in self.steps:
            ep = step["end_pos"]
            step["end_pos"] = [t[0] + ep[0], t[1] + ep[1], t[2] + ep[2]]
            if "start_pos" in step:
                sp = step["start_pos"]
                step["start_pos"] = [t[0] + sp[0], t[1] + sp[1], t[2] + sp[2]]
        self._target_frame_unresolved = False
        print(f"[Objective] Resolved target-frame steps using target pos {t}")

    def _update_plot(self):
        for bar, label in zip(self._bars, self._labels):
            bar.set_height(self._cost_avg[label])
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

        if self._last_total_costs is not None:
            hist, _ = np.histogram(self._last_total_costs,
                                   bins=self._SPEC_BINS,
                                   range=(0, self._SPEC_MAX))
            self._cost_spec = np.roll(self._cost_spec, -1, axis=1)
            self._cost_spec[:, -1] = hist / hist.sum()
            self._im_spec.set_data(self._cost_spec)
            self._fig2.canvas.draw_idle()
            self._fig2.canvas.flush_events()

    def reset(self):
        """Advance to next step if current block reached its goal."""
        if self._last_obj_pos is not None and self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            goal = torch.tensor(step["end_pos"][:2], dtype=torch.float32)
            dist = torch.linalg.norm(self._last_obj_pos.cpu()[:2] - goal).item()
            if dist < self.step_threshold:
                print(self.current_step, step, goal, dist)
                self.current_step += 1
                if self.current_step < len(self.steps):
                    ns = self.steps[self.current_step]
                    print(f"\n[Step {self.current_step}/{len(self.steps)}] "
                          f"now pushing {ns['obj_name']} → {ns['end_pos']}")
                else:
                    print(f"\n[Step] All {len(self.steps)} steps completed!")
        self._first_call = True

    def reset_episode(self, steps: list | None = None):
        """Replace solution steps and reset to step 0. Called by MPPIIsaacLabPlanner.reset_episode()."""
        if steps is not None:
            self.steps = steps
            print(f"[Objective] reset_episode: loaded {len(steps)} steps")
            for i, step in enumerate(steps):
                print(f"  Step {i}: push {step['obj_name']} (idx {step['obj_idx']}) → {step['end_pos']}")
        self.current_step = 0
        self._first_call = True
        self._last_obj_pos = None

    def compute_cost(self, sim) -> torch.Tensor:
        device = sim.device

        # TCP tip position
        ee_pos  = sim.get_ee_pos()   # (num_envs, 3)
        ee_quat = sim.get_ee_quat()  # (num_envs, 4)
        tcp_offset = torch.tensor([0,0, get_tool_length()])
        tcp_pos = ee_pos + quat_apply(ee_quat, tcp_offset)  # (num_envs, 3)

        # All steps done — penalise joint velocity so MPPI drives the robot to a stop
        if self.current_step >= len(self.steps):
            joint_vel = torch.linalg.norm(sim.get_joint_vel(), dim=1)
            return 100.0 * joint_vel

        step = self.steps[self.current_step]
        obj_idx  = step["obj_idx"]
        goal_pos = torch.tensor(step["end_pos"], dtype=torch.float32, device=device)
        sim.set_goal(goal_pos)

        obj_pos = sim.get_object_pos(obj_idx)  # (num_envs, 3)

        # Cache real object position (env 0) for step-advance check in reset()
        if self._first_call:
            self._last_obj_pos = obj_pos[0].detach().clone()
            # Lock mode for the entire rollout based on the actual current state (env 0).
            r2o0 = tcp_pos[0:1] - obj_pos[0:1]
            o2g0 = goal_pos.unsqueeze(0) - obj_pos[0:1]
            d0   = self._costs["robot_to_obj"](r2o0)
            hm0  = self._costs["height_match"](tcp_pos[0:1, 2], obj_pos[0:1, 2]).item()
            pa0  = self._costs["push_align"](r2o0, o2g0, d0).item()
            self._is_push_mode = (pa0 < self._approach_pa_threshold) \
                               and (hm0 < self._approach_hm_threshold)
            self._first_call = False

        robot_to_obj = tcp_pos - obj_pos                 # (num_envs, 3)
        obj_to_goal  = goal_pos.unsqueeze(0) - obj_pos  # (num_envs, 3)

        robot_to_obj_dist = self._costs["robot_to_obj"](robot_to_obj)  # (B,) always needed

        # ── Mode locked for full rollout horizon ──────────────────────────────
        # Compute push_align and height_match unconditionally — shared cost terms.
        height_match_raw = self._costs["height_match"](tcp_pos[:, 2], obj_pos[:, 2])   # (B,)
        push_align_raw   = self._costs["push_align"](robot_to_obj, obj_to_goal, robot_to_obj_dist)  # (B,)

        is_push = torch.full((tcp_pos.shape[0],), self._is_push_mode,
                             dtype=torch.bool, device=device)

        raw = {}

        # ── Shared costs (both modes) ─────────────────────────────────────────
        raw["height_match"] = height_match_raw
        raw["push_align"]   = push_align_raw
        if "robot_ori" in self._active_costs:
            raw["robot_ori"] = self._costs["robot_ori"](ee_quat)
        if "collision" in self._active_costs:
            raw["collision"] = self._costs["collision"](
                sim.get_contact_forces(0)
            )
        if "joint_vel" in self._active_costs:
            raw["joint_vel"] = self._costs["joint_vel"](sim.get_joint_vel())
        if "singularity" in self._active_costs:
            raw["singularity"] = self._costs["singularity"](sim.get_ee_jacobian())
        if "tcp_floor" in self._active_costs:
            raw["tcp_floor"] = self._costs["tcp_floor"](tcp_pos[:, 2])

        # ── Push-mode-only costs (zeroed in approach mode) ────────────────────
        if "robot_to_obj" in self._active_costs:
            raw["robot_to_obj"] = robot_to_obj_dist
        if "obj_to_goal" in self._active_costs:
            otg = self._costs["obj_to_goal"](obj_to_goal)
            raw["obj_to_goal"] = torch.where(is_push, otg, torch.zeros_like(otg))

        # ── Approach-mode-only costs (zeroed in push mode) ───────────────────
        if "obj_avoidance" in self._active_costs:
            avoid = torch.zeros(tcp_pos.shape[0], device=device)
            block_pos0 = []
            for i in range(len(_BLOCK_SPECS)):
                blk_pos = sim.get_object_pos(i)  # (B, 3)
                block_pos0.append(blk_pos[0].detach().cpu())
                d = torch.linalg.norm(tcp_pos - blk_pos, dim=1)
                alpha = torch.clamp(1.0 - d / self._obj_avoidance_dist_thr, min=0.0)
                avoid += alpha ** 2
            raw["obj_avoidance"] = torch.where(is_push, torch.zeros_like(avoid), avoid)

            if self._draw is not None:
                origin = sim.scene.env_origins[0].cpu()
                self._draw.clear_points()
                tp = tuple((tcp_pos[0].detach().cpu() + origin).tolist())
                self._draw.draw_points([tp], [(0.0, 0.6, 1.0, 1.0)], [12.0])
                bps = [tuple((bp + origin).tolist()) for bp in block_pos0]
                self._draw.draw_points(bps, [(1.0, 0.2, 0.2, 1.0)] * len(bps), [10.0] * len(bps))
        elif "above_target" in self._active_costs:
            # fallback: above_target still works if obj_avoidance weight is 0
            raw["above_target"] = torch.zeros(tcp_pos.shape[0], device=device)
            for i in range(len(_BLOCK_SPECS)):
                target_pos  = sim.get_object_pos(i)
                target_quat = sim.get_object_quat(i)
                raw["above_target"] += self._costs["above_target"](tcp_pos, target_pos, target_quat)
            raw["above_target"] = torch.where(is_push, torch.zeros_like(raw["above_target"]),
                                              raw["above_target"])

        for t in raw.values():
            t[torch.isnan(t)] = 100.0

        for k, proj in self._projections.items():
            if k in raw:
                raw[k] = proj(raw[k])

        weighted = {k: self.weights[k] * v for k, v in raw.items()}
        total = sum(weighted.values())   # (num_envs,)
        self._last_total_costs = total.detach().cpu().numpy()

        for k, v in weighted.items():
            self._cost_avg[k] = ((1 - self._EMA_ALPHA) * self._cost_avg[k]
                                 + self._EMA_ALPHA * v.mean().item())

        self._call_count += 1
        if self._call_count % self._PLOT_INTERVAL == 0:
            self._update_plot()

        return total


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)

    if args_cli.solution_path:
        cfg.solution_path = args_cli.solution_path

    scenario_path = args_cli.scenario
    block_positions = None
    if scenario_path is not None:
        with open(scenario_path) as f:
            sc = yaml.safe_load(f)
        is_ = sc["initial_state"]
        bin_positions = [is_["target_pos"]] + [o["pos"] for o in is_["obstacles"]]
        block_positions = [_bin_to_mppi_local(p) for p in bin_positions]

    block_cfgs = make_block_cfgs(positions=block_positions)
    static_cfgs = make_static_cfgs(stand_urdf=cfg.stand_urdf)
    # static_cfgs[1] is the table: center Z + half-thickness = surface Z
    _table_cfg = static_cfgs[0]
    table_surface_z = _table_cfg.init_state.pos[2] + _table_cfg.spawn.size[2] / 2

    robot_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        # filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Object{i}" for i in range(len(block_cfgs))],
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

    objective = Objective(cfg, table_surface_z=table_surface_z,
                          steps_override=[] if args_cli.defer_solution else None)
    planner = MPPIIsaacLabPlanner(
        cfg,
        objective,
        robot_cfg=robot_cfg,
        prior=None,
        object_cfgs=block_cfgs,
        static_cfgs=static_cfgs,
        contact_sensor_cfgs=[robot_contact_sensor],
    )

    server = zerorpc.Server(planner)
    server.bind("tcp://0.0.0.0:4242")
    print("[planner] Stacked-blocks MPPI server listening on tcp://0.0.0.0:4242")
    server.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
