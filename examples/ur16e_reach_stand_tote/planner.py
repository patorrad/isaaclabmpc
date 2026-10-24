"""MPPI planner server for the UR16e reach task (Isaac Lab backend).

Analogous to genesismpc/examples/ur5_stick_stand/planner.py.

Run with:
    cd /home/paolo/Documents/isaaclabmpc
    conda activate env_isaaclab
    python examples/ur16e_reach_stand_tote/planner.py

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
parser.add_argument("--no_waypoint", action="store_true", help="search: don't move MPPI's goal along the tree's path")
parser.add_argument("--no_prior", action="store_true", help="search: no path-tracking prior sample")
parser.add_argument("--no_search", action="store_true", help="disable the hidden-object search")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True          # planner always runs headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. All other imports (safe now that the app is running)
# ===========================================================================
import json
import math
import os
import sys

import numpy as np
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
from isaaclab.sim import RigidBodyPropertiesCfg
from isaaclab_mpc.planner.mppi_isaaclab import MPPIIsaacLabPlanner
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabConfig
from isaaclab_mpc.cost import OrientationCost, JointVelCost, GaussianProjection, ToFClearanceCost
from isaaclab_mpc.sensors.tof import zone_distances
from isaaclab_mpc.cost.utils import quat_apply
from isaaclab_mpc.utils.transport import bytes_to_torch, torch_to_bytes
from assets.robots.ur16e import make_ur16e_cfg, get_tool_length
from isaaclab_mpc.search.guide import PathTrackingPrior, SearchGuide
from isaaclab_mpc.perception.scene_perception import ScenePerception, models_from_catalogue
from examples.ur16e_reach_stand_tote.scene import (
    TOTE_FLOOR_Z, TOTE_INTERIOR, known_objects, make_static_cfgs, make_tof_cfgs, make_ycb_cfgs,
    object_halves, tote_static_boxes, tote_wall_boxes, ycb_object_poses,
)


# ===========================================================================
# 3. Config loading
# ===========================================================================

@dataclass
class IsaacLabCfg:
    dt: float = 1.0 / 60.0
    visualize_rollouts: bool = True
    render: bool = False
    env_spacing: float = 1.5


@dataclass
class CostWeights:
    ee_to_goal: float = 30.0   # L1 distance TCP tip → goal (m)
    robot_ori:  float = 7.0    # tool tilt from straight down (rad); yaw is free
    joint_vel:  float = 1.0    # joint velocity magnitude
    tof_collision: float = 0.0  # ToF readings closer than costs.tof_clearance (0 clear → 1 per sensor at contact)


@dataclass
class GaussianProjParams:
    """Parameters for one cost term's Gaussian projection. c=0 → passthrough."""
    n: int   = 1
    c: float = 0.0
    s: float = 0.0
    r: float = 0.0


@dataclass
class GaussianProjectionConfig:
    """Per-cost Gaussian projection params. Inactive when enabled=False or c=0."""
    enabled:    bool               = False
    ee_to_goal: GaussianProjParams = field(default_factory=GaussianProjParams)
    robot_ori:  GaussianProjParams = field(default_factory=GaussianProjParams)
    joint_vel:  GaussianProjParams = field(default_factory=GaussianProjParams)
    tof_collision: GaussianProjParams = field(default_factory=GaussianProjParams)


@dataclass
class CostConfig:
    weights: CostWeights = field(default_factory=CostWeights)
    gaussian_projection: GaussianProjectionConfig = field(default_factory=GaussianProjectionConfig)
    tof_clearance: float = 0.05   # m; ToF readings closer than this are penalised


@dataclass
class PlannerConfig:
    n_steps: int = 10000
    nx: int = 12
    goal: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.6])
    ee_link_name: str = "wrist_3_link"
    stand_urdf: str = ""
    robot_init_pos: List[float] = field(default_factory=lambda: [0.208, 0.0, 2.075])
    robot_init_joints: List[float] = field(default_factory=lambda: [0.549, -2.2557, 1.0872, 0.8265, 1.5802, 0.5275])
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)
    costs: CostConfig = field(default_factory=CostConfig)
    objects: List[dict] = field(default_factory=list)   # YCB objects in the tote
    ycb_friction: float = 0.5
    tof_sensors: List[dict] = field(default_factory=list)   # VL53L5CX mounts
    tof_rays_per_zone: int = 2
    search: dict = field(default_factory=dict)   # hidden-object search (config `search:`)
    perception: dict = field(default_factory=lambda: {"mode": "privileged"})


def _load_config(yaml_path: str) -> PlannerConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cfg = PlannerConfig()
    cfg.n_steps = raw.get("n_steps", cfg.n_steps)
    cfg.nx = raw.get("nx", cfg.nx)
    cfg.goal = raw.get("goal", cfg.goal)
    cfg.ee_link_name = raw.get("ee_link_name", cfg.ee_link_name)
    cfg.stand_urdf        = raw.get("stand_urdf",        cfg.stand_urdf)
    cfg.robot_init_pos    = raw.get("robot_init_pos",    cfg.robot_init_pos)
    cfg.robot_init_joints = raw.get("robot_init_joints", cfg.robot_init_joints)
    cfg.objects           = raw.get("objects") or []
    cfg.ycb_friction      = float(raw.get("ycb_friction", cfg.ycb_friction))
    cfg.tof_sensors       = raw.get("tof_sensors") or []
    cfg.tof_rays_per_zone = int(raw.get("tof_rays_per_zone", cfg.tof_rays_per_zone))
    cfg.search            = raw.get("search") or {}
    cfg.perception        = raw.get("perception") or {"mode": "privileged"}

    if "mppi" in raw:
        mppi_raw = raw["mppi"]
        cfg.mppi = MPPIConfig(**{k: v for k, v in mppi_raw.items()})

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
        if "gaussian_projection" in c:
            gp_raw = c["gaussian_projection"]
            gp = GaussianProjectionConfig(enabled=bool(gp_raw.get("enabled", False)))
            for key in ("ee_to_goal", "robot_ori", "joint_vel", "tof_collision"):
                if key in gp_raw:
                    p = gp_raw[key]
                    setattr(gp, key, GaussianProjParams(
                        n=int(p.get("n", 1)), c=float(p.get("c", 0.0)),
                        s=float(p.get("s", 0.0)), r=float(p.get("r", 0.0)),
                    ))
            cfg.costs.gaussian_projection = gp
        if "tof_clearance" in c:
            cfg.costs.tof_clearance = float(c["tof_clearance"])

    return cfg


# ===========================================================================
# 4. Objective (cost function)
# ===========================================================================

class Objective:
    """Reach cost: drive the TCP tip to the goal with the tool pointing down.

    Cost terms (weights and Gaussian projections from config.yaml):
      ee_to_goal — L1 distance from the TCP tip to the goal
      robot_ori  — polar tilt of the tool axis from straight down (yaw is free)
      joint_vel  — joint velocity magnitude
      tof_collision — ToF zone readings closer than costs.tof_clearance (needs tof_sensors)
    """

    def __init__(self, cfg: PlannerConfig):
        w = cfg.costs.weights
        self.weights = {
            "ee_to_goal": w.ee_to_goal,
            "robot_ori":  w.robot_ori,
            "joint_vel":  w.joint_vel,
            "tof_collision": w.tof_collision,
        }
        self._costs = {
            "robot_ori": OrientationCost(),
            "joint_vel": JointVelCost(),
            "tof_collision": ToFClearanceCost(clearance=cfg.costs.tof_clearance),
        }
        self._tof_rays_per_zone = cfg.tof_rays_per_zone
        # TCP tip along wrist_3_link +Z (tool0 is merged into wrist_3_link).
        self._tcp_offset = torch.tensor([0.0, 0.0, get_tool_length()])
        print(f"[Objective] TCP offset from URDF: {self._tcp_offset[2]:.4f} m")

        gp_cfg = cfg.costs.gaussian_projection
        self._projections = {}
        if gp_cfg.enabled:
            for k in self.weights:
                p = getattr(gp_cfg, k)
                if p.c != 0:
                    self._projections[k] = GaussianProjection(n=p.n, c=p.c, s=p.s, r=p.r)
            print(f"[Objective] GaussianProjection enabled for: {sorted(self._projections)}")

    def reset(self):
        pass

    def compute_cost(self, sim) -> torch.Tensor:
        ee_pos  = sim.get_ee_pos()   # (num_envs, 3)
        ee_quat = sim.get_ee_quat()  # (num_envs, 4)
        goal    = sim.get_goal().to(sim.device)  # (3,)

        tcp_offset = self._tcp_offset.to(sim.device).expand(sim.num_envs, 3)
        tcp_pos = ee_pos + quat_apply(ee_quat, tcp_offset)  # (num_envs, 3)

        raw = {"ee_to_goal": torch.linalg.norm(tcp_pos - goal.unsqueeze(0), dim=1, ord=1)}
        if self.weights.get("robot_ori", 0.0) != 0.0:
            raw["robot_ori"] = self._costs["robot_ori"](ee_quat)
        if self.weights.get("joint_vel", 0.0) != 0.0:
            raw["joint_vel"] = self._costs["joint_vel"](sim.get_joint_vel())
        if self.weights.get("tof_collision", 0.0) != 0.0 and sim.ray_casters:
            # min_range=0: readings under the sensor's 2 cm floor still count (that is contact).
            zones = torch.stack([zone_distances(rc, rays_per_zone=self._tof_rays_per_zone, min_range=0.0)
                                 for rc in sim.ray_casters], dim=1)       # (num_envs, S, 8, 8)
            raw["tof_collision"] = self._costs["tof_collision"](zones)

        for t in raw.values():
            t[torch.isnan(t)] = 100.0
        for k, proj in self._projections.items():
            if k in raw:
                raw[k] = proj(raw[k])

        return sum(self.weights.get(k, 0.0) * v for k, v in raw.items())


# ===========================================================================
# 5. Planner with object-pose fallback
# ===========================================================================

def _tree_json(tree, max_segments: int = 4000):
    """Search tree segments for drawing; the most-visited first if there are too many."""
    if tree is None or not len(tree["seg"]):
        return None
    order = np.argsort(-tree["visits"])[:max_segments]
    return {"seg": tree["seg"][order].round(4).tolist(), "visits": tree["visits"][order].tolist(),
            "term": tree["term"][order].tolist()}


class TotePlanner(MPPIIsaacLabPlanner):
    """MPPI planner for the tote, with optional hidden-object search guidance.

    Objects: world.py sends the live poses of the known objects after q, dq; when a caller
    sends joint state only (the real-robot bridge), the configured poses are used.

    Search (config `search.enabled`): the client sends each ToF frame with set_tof_frame();
    every `search.every` steps the tree re-plans, and MPPI's goal becomes the waypoint along
    its best path (the client's own goal is ignored). The prior, if enabled, is passed in as
    a PathTrackingPrior on the same guide.
    """

    def __init__(self, cfg, objective, *, object_poses: list, guide: Optional[SearchGuide] = None,
                 perception: Optional[ScenePerception] = None, object_names: list = (),
                 zone_angle: float = 0.0, perception_every: int = 10, **kwargs):
        super().__init__(cfg, objective, **kwargs)
        self._default_object_state = torch.tensor(
            [v for pos, quat in object_poses for v in (*pos, *quat)], dtype=torch.float32)
        self.guide = guide
        self._client_goal = None
        # Perception (config perception.mode != privileged): object poses come from the sensors,
        # never from the client. Undetected objects are parked out of the way in the rollouts.
        self.perception = perception
        self.object_names = list(object_names)
        self.zone_angle = zone_angle
        self.perception_every = perception_every
        self._perc_calls = 0

    def set_goal(self, goal_bytes: bytes):
        if self.guide is None:
            return super().set_goal(goal_bytes)
        self._client_goal = bytes_to_torch(goal_bytes)    # searching: the tree sets the goal

    def set_tof_frame(self, rays_bytes: bytes, tcp_bytes: bytes) -> int:
        """Latest ToF frame as (Z, 7) zone rays [origin(3), dir(3), dist] in the env frame, plus
        the current TCP (3,). Returns the number of belief cells it cleared."""
        rays_t = bytes_to_torch(rays_bytes).reshape(-1, 7)
        if self.perception is not None:
            r = rays_t.to(self.perception.map.device)
            self.perception.add_tof(r[:, :3], r[:, 3:6], r[:, 6], self.zone_angle)
        if self.guide is None:
            return 0
        rays = rays_t.cpu().numpy()
        tcp = bytes_to_torch(tcp_bytes).reshape(3).cpu().numpy()
        return int(self.guide.observe(rays[:, :3], rays[:, 3:6], rays[:, 6], tcp))

    def set_depth_frame(self, rays_bytes: bytes) -> int:
        """Depth-camera rays (R, 8) [origin(3), dir(3), range, hit weight] in the env frame;
        weight 0 = no return within range (carve only)."""
        if self.perception is None:
            return 0
        r = bytes_to_torch(rays_bytes).reshape(-1, 8).to(self.perception.map.device)
        self.perception.add_depth(r[:, :3], r[:, 3:6], r[:, 6], occ_weight=r[:, 7])
        return int(len(r))

    def _perceived_object_states(self):
        """[(pos, quat) or None] per known object in config order, from perception."""
        poses = self.perception.poses()
        out = []
        for name in self.object_names:
            if name in poses:
                p, yaw = poses[name]
                out.append((p, np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])))
            else:
                out.append(None)
        return out

    def get_perceived_objects(self) -> str:
        """JSON {name: [x, y, z, yaw, score]} of the objects perception currently places."""
        if self.perception is None:
            return json.dumps({})
        return json.dumps({n: [*map(float, f.pos), float(f.yaw), float(f.score)]
                           for n, f in self.perception.tracks.items()})

    def compute_action_tensor(self, dof_state_bytes: bytes, root_state_bytes: bytes) -> bytes:
        perceived = None
        if self.perception is not None and not getattr(self, "_read_only", False):
            if self._perc_calls % self.perception_every == 0:
                self.perception.update()
            self._perc_calls += 1
            perceived = self._perceived_object_states()
            # Replace whatever object state the client sent with the perceived one (parked if unseen).
            dof = bytes_to_torch(dof_state_bytes).cpu().float()[: 2 * self.sim.num_dof]
            parts = [dof]
            for i, pq in enumerate(perceived):
                pos, quat = pq if pq is not None else (np.array([2.0 + 0.3 * i, 2.0, 1.0]), np.array([1.0, 0, 0, 0]))
                parts.append(torch.tensor(np.concatenate([pos, quat]), dtype=torch.float32))
            dof_state_bytes = torch_to_bytes(torch.cat(parts))
        elif self._default_object_state.numel() and not getattr(self, "_read_only", False):
            dof_state = bytes_to_torch(dof_state_bytes)
            if dof_state.numel() <= 2 * self.sim.num_dof:
                dof_state_bytes = torch_to_bytes(torch.cat([dof_state.cpu().float(),
                                                            self._default_object_state]))
        if self.guide is not None and not getattr(self, "_read_only", False):
            if perceived is not None:
                self.guide.set_objects(perceived)
            else:
                obj = bytes_to_torch(dof_state_bytes).cpu().float()[2 * self.sim.num_dof:].reshape(-1, 7).numpy()
                # Test hook: TOTE_POSE_NOISE=<m> perturbs the privileged poses (resampled every 10 steps)
                # to measure how sensitive search + MPPI are to perception-sized errors.
                noise = float(os.environ.get("TOTE_POSE_NOISE", "0") or 0)
                if noise > 0 and len(obj):
                    self._noise_calls = getattr(self, "_noise_calls", 0) + 1
                    if self._noise_calls % 10 == 1:
                        self._pose_noise = np.random.uniform(-noise, noise, (len(obj), 2))
                    noisy = obj.copy()
                    noisy[:, :2] += self._pose_noise
                    where = os.environ.get("TOTE_POSE_NOISE_TO", "both")      # tree | mppi | both
                    if where in ("mppi", "both"):
                        dof = bytes_to_torch(dof_state_bytes).cpu().float()[: 2 * self.sim.num_dof]
                        dof_state_bytes = torch_to_bytes(torch.cat([dof, torch.tensor(noisy.reshape(-1), dtype=torch.float32)]))
                    if where in ("tree", "both"):
                        obj = noisy
                self.guide.set_objects([(o[:3], o[3:7]) for o in obj])
            res = self.guide.plan()
            if res is not None:
                print(f"[search] ({res['reason']}) {res['num_nodes']} nodes in {res['ms']:.0f} ms; best path reveals "
                      f"{res['best_gain']:.2f} of the remaining mass; actions {res['best_actions'][:4]}; "
                      f"belief support {self.guide.belief.mass():.2f}", flush=True)
            goal = self.guide.mppi_goal()
            if goal is not None:
                self.sim.set_goal(torch.tensor(goal, dtype=torch.float32))
            out = super().compute_action_tensor(dof_state_bytes, root_state_bytes)
            self._diag_calls = getattr(self, "_diag_calls", 0) + 1
            if self.guide.p.get("debug") and (res is not None or self._diag_calls % 25 == 0):
                g = self.guide
                fmt = lambda v: "None" if v is None else "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
                prog = f"{g.progress(g.tcp):.3f}/{g.progress(g.path[-1]):.3f}" if g.path is not None else "-"
                roll = self.sim.get_object_pos(0)[0].tolist() if self.sim.objects else None
                if perceived is not None:
                    obj0 = None if not perceived or perceived[0] is None else perceived[0][0]
                    src = "perceived"
                else:
                    obj0 = obj[0][:3] if len(obj) else None
                    src = "world"
                print(f"[diag] tcp {fmt(g.tcp)} goal {fmt(goal)} path progress {prog} m | "
                      f"obj0 {src} {fmt(obj0)} rollout-end {fmt(roll)}", flush=True)
            return out
        return super().compute_action_tensor(dof_state_bytes, root_state_bytes)

    def get_search_viz(self) -> str:
        """JSON: best path, MPPI goal, view candidates and the belief's remaining cells (subsampled)."""
        if self.guide is None:
            return json.dumps({})
        g = self.guide
        cells = g.belief.cells[g.belief.p > 0]
        if len(cells) > 400:
            cells = cells[:: len(cells) // 400 + 1]
        goal = g.mppi_goal()
        res = g.result or {}
        return json.dumps({
            "path": [] if g.path is None else g.path.round(4).tolist(),
            "goal": None if goal is None else [round(float(v), 4) for v in goal],
            "views": res.get("views", []).round(4).tolist() if len(res.get("views", [])) else [],
            "cells": cells.round(4).tolist(),
            "floor_z": g.belief.floor_z,
            "support": g.belief.mass(),
            "push_planned": bool(res.get("push_planned", False)),
            "tree": _tree_json(res.get("tree")) if g.p.get("draw_tree", False) else None,
        })


# ===========================================================================
# 6. Main
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)

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
    known = known_objects(cfg.objects)          # the hidden target is not modelled by the planner
    if known:
        print(f"[planner] YCB objects: {[o['name'] for o in known]}")

    guide, prior = None, None
    if cfg.search.get("enabled") and not args_cli.no_search:
        sp = dict(cfg.search)
        if args_cli.no_waypoint:
            sp["waypoint"] = {**sp.get("waypoint", {}), "enabled": False}
        if args_cli.no_prior:
            sp["prior"] = {**sp.get("prior", {}), "enabled": False}
        tool = get_tool_length()                                     # EE origin → tip (0.135 m)
        sensor_z = float(cfg.tof_sensors[0]["pos"][2]) if cfg.tof_sensors else 0.017
        (x0, x1), (y0, y1) = TOTE_INTERIOR
        z0, z1 = sp.get("workspace_z", [TOTE_FLOOR_Z, 1.15])
        sp.update(
            workspace_lo=[x0, y0, z0], workspace_hi=[x1, y1, z1],
            rod_radius=sp.get("rod_radius", 0.006),                  # tip sphere / pipe nipple
            coupling_radius=sp.get("coupling_radius", 0.04),         # tool0 coupling, 0.101–0.135 above tip
            coupling_bottom=sp.get("coupling_bottom", tool - 0.034), coupling_top=tool,
            sensor_height=tool - sensor_z, fov=math.radians(sp.get("fov_deg", 45.0)),
        )
        guide = SearchGuide(sp, TOTE_INTERIOR, TOTE_FLOOR_Z, tote_wall_boxes(), object_halves(known))
        prior = PathTrackingPrior(guide, tool, float(cfg.mppi.u_min[0]), float(cfg.mppi.u_max[0]))
        cfg.mppi.sample_other_priors = prior.enabled
        objective.weights["tof_collision"] = 0.0                     # would fight deliberate pushes
        print(f"[planner] search on: waypoint={sp.get('waypoint', {}).get('enabled', True)} "
              f"prior={prior.enabled}; tof_collision off while searching")

    perception = None
    mode = cfg.perception.get("mode", "privileged")
    if mode != "privileged":
        (x0, x1), (y0, y1) = TOTE_INTERIOR
        pp = dict(cfg.perception)
        perception = ScenePerception(
            lo=(x0 - 0.03, y0 - 0.03, TOTE_FLOOR_Z - 0.01), hi=(x1 + 0.03, y1 + 0.03, TOTE_FLOOR_Z + 0.30),
            static_boxes=tote_static_boxes(), models=models_from_catalogue([o["name"] for o in known]),
            floor_z=TOTE_FLOOR_Z, res=pp.get("res", 0.01), device=cfg.mppi.device, p=pp)
        print(f"[planner] perception: {mode} (object poses from sensors, not the client)")

    planner = TotePlanner(
        cfg,
        objective,
        object_poses=ycb_object_poses(known),
        guide=guide,
        perception=perception,
        object_names=[o["name"] for o in known],
        zone_angle=math.radians(cfg.search.get("fov_deg", 45.0)) / 8,
        perception_every=int(cfg.perception.get("update_every", 10)),
        robot_cfg=robot_cfg,
        prior=prior,
        object_cfgs=make_ycb_cfgs(known, friction=cfg.ycb_friction, rollout_model=bool(guide)),
        ray_caster_cfgs=make_tof_cfgs(cfg.tof_sensors, rays_per_zone=cfg.tof_rays_per_zone),
        static_cfgs=make_static_cfgs(stand_urdf=cfg.stand_urdf),
    )

    server = zerorpc.Server(planner)
    server.bind("tcp://0.0.0.0:4242")
    print("[planner] MPPI server listening on tcp://0.0.0.0:4242")
    server.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
