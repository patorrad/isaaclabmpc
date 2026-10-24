"""World runner for UR16e reach (Isaac Lab backend).

Runs a single-environment rendered simulation and calls the MPPI planner
server (planner.py) via zerorpc to get joint-velocity commands.
MPPI rollout trajectories are drawn in the viewer each step.

Start the planner first, then this runner:

    # Terminal 1 — headless MPPI planner:
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_reach/planner.py

    # Terminal 2 — rendered world:
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_reach/world.py

    # Headless world (no viewer, no rollout vis):
    ... world.py --headless

Keyboard controls (viewer window):
    Arrow keys  — move goal in X/Y  (+/- 2 cm per key press)
    PgUp/PgDn   — move goal in Z
    Ctrl-C      — quit
"""

# ===========================================================================
# 1. Simulator bootstrap — must happen first
# ===========================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR16e world runner")
parser.add_argument("--n_steps", type=int, default=100000)
parser.add_argument("--planner_addr", type=str, default="tcp://localhost:4242")
parser.add_argument("--n_rollouts_draw", type=int, default=50,
                    help="Number of MPPI rollout trajectories to visualise (0 = off)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# A wrist depth camera (perception.mode: tof+camera) needs camera rendering enabled at launch.
import os as _os
import yaml as _yaml
with open(_os.path.join(_os.path.dirname(__file__), "config.yaml")) as _f:
    _PERCEPTION = (_yaml.safe_load(_f).get("perception") or {})
USE_CAMERA = _PERCEPTION.get("mode") == "tof+camera"
if USE_CAMERA:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. All other imports
# ===========================================================================
import json
import math
import os
import sys
import time
import threading

import torch
import yaml
import zerorpc
from dataclasses import dataclass, field
from typing import List

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.sim import RigidBodyPropertiesCfg
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabWrapper, IsaacLabConfig
from isaaclab_mpc.utils.transport import torch_to_bytes, bytes_to_torch
from isaaclab_mpc.sensors.tof import zone_distances, zone_rays_from_sensor
if USE_CAMERA:
    from isaaclab_mpc.sensors.depth_camera import depth_rays, make_depth_camera_cfg, set_camera_pose
from assets.robots.ur16e import make_ur16e_cfg, get_tool_length
from examples.ur16e_reach_stand_tote.scene import (make_static_cfgs, make_tof_cfgs, make_ycb_cfgs,
                                                   object_halves, target_index)


# ===========================================================================
# 3. Config
# ===========================================================================

@dataclass
class IsaacLabCfg:
    dt: float = 1.0 / 60.0


@dataclass
class WorldConfig:
    n_steps: int = 100000
    goal: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.6])
    ee_link_name: str = "wrist_3_link"
    stand_urdf: str = ""
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)
    robot_init_pos: List[float] = field(default_factory=lambda: [0.208, 0.0, 2.075])
    robot_init_joints: List[float] = field(default_factory=lambda: [0.549, -2.2557, 1.0872, 0.8265, 1.5802, 0.5275])
    objects: List[dict] = field(default_factory=list)   # YCB objects in the tote
    ycb_friction: float = 0.5
    tof_sensors: List[dict] = field(default_factory=list)   # VL53L5CX mounts
    tof_rays_per_zone: int = 2
    tof_hit_radius: float = 0.005
    search: dict = field(default_factory=dict)   # hidden-object search (config `search:`)
    perception: dict = field(default_factory=lambda: {"mode": "privileged"})


def _load_config(yaml_path: str) -> WorldConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    cfg = WorldConfig()
    cfg.n_steps = raw.get("n_steps", cfg.n_steps)
    cfg.goal = raw.get("goal", cfg.goal)
    cfg.ee_link_name = raw.get("ee_link_name", cfg.ee_link_name)
    cfg.stand_urdf        = raw.get("stand_urdf",        cfg.stand_urdf)
    cfg.robot_init_pos    = raw.get("robot_init_pos",    cfg.robot_init_pos)
    cfg.robot_init_joints = raw.get("robot_init_joints", cfg.robot_init_joints)
    cfg.objects           = raw.get("objects") or []
    cfg.ycb_friction      = float(raw.get("ycb_friction", cfg.ycb_friction))
    cfg.tof_sensors       = raw.get("tof_sensors") or []
    cfg.tof_rays_per_zone = int(raw.get("tof_rays_per_zone", cfg.tof_rays_per_zone))
    cfg.tof_hit_radius    = float(raw.get("tof_hit_radius", cfg.tof_hit_radius))
    cfg.search            = raw.get("search") or {}
    cfg.perception        = raw.get("perception") or {"mode": "privileged"}

    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(dt=il.get("dt", 1.0 / 60.0))
    return cfg


# ===========================================================================
# 4. Keyboard goal control
# ===========================================================================

class GoalController:
    """Listens for arrow/PgUp/PgDn keys and adjusts the world goal."""

    STEP = 0.02  # metres per key press

    def __init__(self, goal: torch.Tensor, lock: threading.Lock):
        self._goal = goal
        self._lock = lock
        self._start()

    def _start(self):
        try:
            from pynput import keyboard

            def on_press(key):
                delta = torch.zeros(3, device=self._goal.device)
                try:
                    if key == keyboard.Key.up:
                        delta[0] = self.STEP
                    elif key == keyboard.Key.down:
                        delta[0] = -self.STEP
                    elif key == keyboard.Key.right:
                        delta[1] = -self.STEP
                    elif key == keyboard.Key.left:
                        delta[1] = self.STEP
                    elif key == keyboard.Key.page_up:
                        delta[2] = self.STEP
                    elif key == keyboard.Key.page_down:
                        delta[2] = -self.STEP
                except Exception:
                    pass
                if delta.any():
                    with self._lock:
                        self._goal.add_(delta)
                    print(f"\n[goal] {self._goal.tolist()}", flush=True)

            listener = keyboard.Listener(on_press=on_press)
            listener.daemon = True
            listener.start()
        except Exception as e:
            print(f"[GoalController] keyboard listener not available: {e}")


# ===========================================================================
# 5. Rollout + goal visualisation
# ===========================================================================

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (w, x, y, z convention).

    Args:
        q: (4,) quaternion
        v: (3,) vector
    Returns:
        (3,) rotated vector
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


class RolloutVisualiser:
    """Draws MPPI rollout trajectories and the goal marker each step."""

    # Rollout line colour (RGBA) and thickness
    ROLLOUT_COLOR = (0.1, 0.9, 0.1, 0.25)
    ROLLOUT_WIDTH = 1
    # Goal sphere colour and size
    GOAL_COLOR = (1.0, 0.4, 0.0, 1.0)
    GOAL_SIZE = 15.0

    def __init__(self, tcp_offset_local: torch.Tensor):
        """
        Args:
            tcp_offset_local: (3,) tool-tip offset in the EE link frame.
                              Set to zeros if tracking the link origin directly.
        """
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self.tcp_offset_local = tcp_offset_local

    def update(
        self,
        rollouts_bytes: bytes,
        goal: torch.Tensor,
        ee_quat_world: torch.Tensor,
        env_origin: torch.Tensor,
        n_draw: int,
    ):
        """Refresh debug geometry for this control step.

        The planner tracks the EE link origin.  To shift rollouts to the
        actual tool tip, we rotate tcp_offset_local into world frame using
        the current EE quaternion and add it to every rollout point — the
        same pattern as genesismpc/examples/ur5_suction/world.py.

        Args:
            rollouts_bytes:  serialised (horizon, num_envs, 3) tensor from planner.
                             Positions are in local (env-relative) frame.
            goal:            (3,) goal position in local frame.
            ee_quat_world:   (4,) current EE quaternion in world frame (w, x, y, z).
            env_origin:      (3,) world-frame offset of env 0 (scene.env_origins[0]).
            n_draw:          how many rollout trajectories to draw (subsampled).
        """
        self._draw.clear_lines()
        self._draw.clear_points()

        origin = env_origin.cpu()

        # Rotate TCP offset into world frame using current EE orientation
        tcp_offset_world = _quat_apply(ee_quat_world.cpu(), self.tcp_offset_local)

        # ---- goal marker (local → world) ----
        gp = tuple((goal.cpu() + origin).tolist())
        self._draw.draw_points([gp], [self.GOAL_COLOR], [self.GOAL_SIZE])

        # ---- rollout trajectories ----
        if n_draw <= 0:
            return

        rollouts = bytes_to_torch(rollouts_bytes)   # (horizon, num_envs, 3)
        if rollouts.shape[0] < 1 or rollouts.shape[1] < 1:
            return

        # local → world, then shift to TCP tip
        rollouts = rollouts.permute(1, 0, 2).cpu() + origin + tcp_offset_world  # (num_envs, H, 3)
        num_envs = rollouts.shape[0]
        stride = max(1, num_envs // n_draw)
        rollouts_sub = rollouts[::stride]

        for traj in rollouts_sub:
            pts = [tuple(p.tolist()) for p in traj]
            self._draw.draw_lines_spline(pts, self.ROLLOUT_COLOR, self.ROLLOUT_WIDTH, False)


class SearchVisualiser:
    """Draws the search state after RolloutVisualiser (which clears the canvas each step):
    the tree's best path, MPPI's goal, the belief's remaining cells and the ToF hits."""

    PATH_COLOR = (1.0, 0.2, 0.8, 1.0)
    TREE_COLOR = (0.1, 0.85, 1.0)       # cyan; alpha from visit count
    TREE_COLLISION_COLOR = (1.0, 0.3, 0.3, 0.35)
    GOAL_COLOR = (1.0, 1.0, 0.0, 1.0)
    CELL_COLOR = (0.2, 0.6, 1.0, 0.7)
    HIT_COLOR = (1.0, 0.1, 0.1, 1.0)

    def __init__(self):
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()

    def update(self, viz: dict, origin: torch.Tensor, hits: list):
        o = origin.cpu().tolist()
        shift = lambda p: (p[0] + o[0], p[1] + o[1], p[2] + o[2])
        if viz.get("cells"):
            z = viz["floor_z"] + 0.003
            pts = [shift((c[0], c[1], z)) for c in viz["cells"]]
            self._draw.draw_points(pts, [self.CELL_COLOR] * len(pts), [4.0] * len(pts))
        tree = viz.get("tree")
        if tree and tree["seg"]:
            vmax = max(tree["visits"]) or 1
            starts = [shift(s[:3]) for s in tree["seg"]]
            ends = [shift(s[3:]) for s in tree["seg"]]
            colors = [self.TREE_COLLISION_COLOR if t == 1 else
                      (*self.TREE_COLOR, 0.15 + 0.85 * math.log1p(v) / math.log1p(vmax))
                      for v, t in zip(tree["visits"], tree["term"])]
            self._draw.draw_lines(starts, ends, colors, [1.0] * len(starts))
        path = viz.get("path") or []
        if len(path) > 1:
            self._draw.draw_lines([shift(p) for p in path[:-1]], [shift(p) for p in path[1:]],
                                  [self.PATH_COLOR] * (len(path) - 1), [3.0] * (len(path) - 1))
        if viz.get("goal"):
            self._draw.draw_points([shift(viz["goal"])], [self.GOAL_COLOR], [14.0])
        if hits:
            self._draw.draw_points([shift(h) for h in hits], [self.HIT_COLOR] * len(hits), [6.0] * len(hits))


def _wrist_camera_cfg(c: dict):
    """Depth camera from config perception.wrist_camera {width, height} (its pose is set every step)."""
    return make_depth_camera_cfg(width=int(c.get("width", 160)), height=int(c.get("height", 96)))


def _wrist_camera_mount(c: dict):
    """(link, offset pos, offset quat wxyz) from config perception.wrist_camera."""
    from isaaclab.utils.math import quat_from_euler_xyz
    r, p, y = (math.radians(float(a)) for a in c.get("rpy_deg", (0.0, 0.0, 0.0)))
    rot = quat_from_euler_xyz(torch.tensor(r), torch.tensor(p), torch.tensor(y)).tolist()
    return c.get("link", "wrist_3_link"), tuple(c.get("pos", (0.0, 0.06, 0.03))), tuple(rot)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


# ===========================================================================
# 6. Control loop
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)
    cfg.n_steps = args_cli.n_steps
    headless = getattr(args_cli, "headless", False)
    n_rollouts_draw = 0 if headless else args_cli.n_rollouts_draw

    _base_robot_cfg = make_ur16e_cfg(pos=cfg.robot_init_pos, rot=(0, 1, 0, 0), joint_pos=cfg.robot_init_joints)
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

    # ------------------------------------------------------------------
    # World simulation: single env, rendered
    # ------------------------------------------------------------------
    world = IsaacLabWrapper(
        cfg=IsaacLabConfig(
            dt=cfg.isaaclab.dt,
            device="cuda:0",
            render=(not headless) or USE_CAMERA,     # cameras only update when the sim renders
        ),
        robot_cfg=robot_cfg,
        num_envs=1,
        ee_link_name=cfg.ee_link_name,
        goal=cfg.goal,
        object_cfgs=make_ycb_cfgs(cfg.objects, friction=cfg.ycb_friction),
        camera_cfgs=[_wrist_camera_cfg(cfg.perception.get("wrist_camera", {}))] if USE_CAMERA else None,
        ray_caster_cfgs=make_tof_cfgs(cfg.tof_sensors, rays_per_zone=cfg.tof_rays_per_zone,
                                      debug_vis=not headless, hit_radius=cfg.tof_hit_radius),
        static_cfgs=make_static_cfgs(stand_urdf=cfg.stand_urdf),
    )
    device = world.device
    DOF = world.num_dof

    # Keyboard goal control
    GoalController(world._goal, world._goal_lock)

    # TCP offset: offset from wrist_3_link origin to tool tip in wrist_3_link frame.
    # tool0 +Z == wrist_3_link +Z (fixed joint chain has no translation, only rotation).
    # Tip length comes from the tool0 collision geometry in the URDF (same as the planner).
    tcp_offset_local = torch.tensor([0.0, 0.0, get_tool_length()])

    # Rollout visualiser (only when rendering)
    vis = RolloutVisualiser(tcp_offset_local) if not headless else None

    # ------------------------------------------------------------------
    # Connect to the MPPI planner server
    # ------------------------------------------------------------------
    print(f"[world] Connecting to planner at {args_cli.planner_addr} …", flush=True)
    planner = zerorpc.Client(timeout=60, heartbeat=None)
    planner.connect(args_cli.planner_addr)
    planner.test("world connected")
    print("[world] Connected.", flush=True)

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------
    q = world.get_joint_pos()[0].clone()   # (DOF,)
    dq = world.get_joint_vel()[0].clone()  # (DOF,)

    print(f"[world] Goal: {cfg.goal}  (use arrow keys / PgUp / PgDn to move)")
    print(f"[world] Body names: {list(world.robot.body_names)}")

    # Hidden-object search: the target is spawned here but never reported to the planner.
    searching = bool(cfg.search.get("enabled"))
    t_idx = target_index(cfg.objects)
    known_idx = [i for i in range(len(cfg.objects)) if i != t_idx]
    halves = object_halves(cfg.objects)
    found_min = int(cfg.search.get("found_min_zones", 2))
    tool_off = torch.tensor([0.0, 0.0, get_tool_length()], device=device)
    known_start = {i: world.get_object_pos(i)[0].clone() for i in known_idx}
    search_vis = SearchVisualiser() if (searching and not headless) else None

    # Perception modes: the planner gets sensor frames only, never the objects' true poses.
    perceiving = cfg.perception.get("mode", "privileged") != "privileged"
    if perceiving:
        from isaaclab_mpc.perception.scene_perception import models_from_catalogue
        yaw_period = {n: m.yaw_period for n, m in
                      models_from_catalogue([cfg.objects[i]["name"] for i in known_idx]).items()}
        print(f"[world] Perception: {cfg.perception['mode']}; object poses are not sent to the planner")

    def perception_report() -> str:
        """Perceived vs true pose of each known object."""
        got = json.loads(planner.get_perceived_objects())
        parts = []
        for i in known_idx:
            name = cfg.objects[i]["name"]
            tp = world.get_object_pos(i)[0].cpu()
            if name not in got:
                parts.append(f"{name}: not detected")
                continue
            x, y, z, yaw, score = got[name]
            q = world.get_object_quat(i)[0].cpu().tolist()
            tyaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            per = yaw_period[name]
            dyaw = 0.0 if per == 0 else min((yaw - tyaw) % per, per - (yaw - tyaw) % per)
            parts.append(f"{name}: {1000 * math.hypot(x - tp[0], y - tp[1]):.0f} mm xy, "
                         f"{1000 * abs(z - tp[2]):.0f} mm z, {math.degrees(dyaw):.0f} deg")
        return "; ".join(parts)

    if searching:
        print(f"[world] Search: target = {cfg.objects[t_idx]['name'] if t_idx is not None else None}; "
              f"found when >= {found_min} ToF zones see it")

    def tof_frame():
        """(Z, 7) zone rays [origin, dir, dist] of every ToF sensor, env-local frame."""
        o = world.scene.env_origins[0]
        rows = []
        for rc in world.ray_casters:
            org, dirs, dist = zone_rays_from_sensor(rc, 0, cfg.tof_rays_per_zone)
            rows.append(torch.cat([org - o, dirs, dist[:, None]], dim=1))
        return torch.cat(rows) if rows else torch.zeros(0, 7, device=device)

    def target_zones_seen() -> int:
        """Ground truth: ToF zones with at least one ray hitting the target's box (+5 mm)."""
        if t_idx is None:
            return 0
        tp, tq = world.get_object_pos(t_idx)[0], world.get_object_quat(t_idx)[0]
        half = torch.tensor(halves[t_idx], device=device) + 0.005
        k2 = cfg.tof_rays_per_zone ** 2
        seen = 0
        for rc in world.ray_casters:
            hits = rc.data.ray_hits_w[0] - world.scene.env_origins[0]          # (B, 3)
            rel = _quat_apply(_quat_conj(tq).expand(len(hits), 4), hits - tp)
            inside = (rel.abs() <= half).all(dim=1)
            seen += int(inside.view(-1, k2).any(dim=1).sum())
        return seen

    def tof_status() -> str:
        """Closest zone reading of each ToF sensor (env 0)."""
        mins = [zone_distances(rc, rays_per_zone=cfg.tof_rays_per_zone)[0].min().item()
                for rc in world.ray_casters]
        return "".join(f"tof{i} {m:.3f} m  " for i, m in enumerate(mins))

    def search_status() -> str:
        if not searching:
            return ""
        moved = sum(int(torch.linalg.norm(world.get_object_pos(i)[0] - known_start[i]) > 0.02) for i in known_idx)
        return f"target zones {target_zones_seen()}  objects moved {moved}  "

    t_prev = time.time()
    t_start = time.time()

    for step in range(cfg.n_steps):
        if not simulation_app.is_running():
            break

        # ------------------------------------------------------------------
        # 1. Sync current goal to planner
        # ------------------------------------------------------------------
        with world._goal_lock:
            goal_now = world._goal.clone()
        planner.set_goal(torch_to_bytes(goal_now.cpu()))
        if searching or perceiving:
            ee_p, ee_q = world.get_ee_pos()[0], world.get_ee_quat()[0]
            tcp_now = ee_p + _quat_apply(ee_q, tool_off)
            planner.set_tof_frame(torch_to_bytes(tof_frame().cpu()), torch_to_bytes(tcp_now.cpu()))
            if USE_CAMERA:
                cam = cfg.perception.get("wrist_camera", {})
                o = world.scene.env_origins[0]
                tool = (ee_p + o, tcp_now + o, float(cam.get("tool_filter_radius", 0.05)))
                org, dirs, rng, occ = depth_rays(world.cameras[0], 0, stride=int(cam.get("stride", 2)), tool=tool)
                planner.set_depth_frame(torch_to_bytes(torch.cat([org - o, dirs, rng[:, None], occ[:, None]], 1).cpu()))

        # ------------------------------------------------------------------
        # 2. Call MPPI planner — returns optimal joint-velocity command
        # ------------------------------------------------------------------
        # Live object poses (local frame) follow q, dq so the planner's rollouts start from them.
        obj_state = [] if perceiving else [t for i in known_idx
                                           for t in (world.get_object_pos(i)[0], world.get_object_quat(i)[0])]
        dof_state = torch.cat([q, dq, *obj_state]).cpu()
        u_bytes = planner.compute_action_tensor(torch_to_bytes(dof_state), b"")
        u = bytes_to_torch(u_bytes).to(device)   # (DOF,)

        # ------------------------------------------------------------------
        # 3. Visualise rollouts + goal (before stepping so viewer is current)
        # ------------------------------------------------------------------
        if vis is not None:
            rollout_bytes = planner.get_rollouts()
            origin = world.scene.env_origins[0]
            ee_quat = world.get_ee_quat()[0]      # (4,) w,x,y,z world frame
            vis.update(rollout_bytes, goal_now, ee_quat, origin, n_rollouts_draw)
            if search_vis is not None:
                hits = [h.tolist() for rc in world.ray_casters
                        for h in (rc.data.ray_hits_w[0] - origin) if torch.isfinite(h).all()]
                search_vis.update(json.loads(planner.get_search_viz()), origin, hits)

        # ------------------------------------------------------------------
        # 4. Apply command and step the world simulation
        # ------------------------------------------------------------------
        if USE_CAMERA:      # place the wrist camera at the link pose before this step renders
            link, cam_p, cam_q = _wrist_camera_mount(cfg.perception.get("wrist_camera", {}))
            li = list(world.robot.body_names).index(link)
            lp = world.robot.data.body_link_pose_w[:, li]
            set_camera_pose(world.cameras[0], lp[:, :3], lp[:, 3:7], cam_p, cam_q)
        world.apply_robot_cmd(u.view(1, DOF))
        world.step()                              # render=True if not headless

        # ------------------------------------------------------------------
        # 5. Read new state
        # ------------------------------------------------------------------
        q = world.get_joint_pos()[0].clone()
        dq = world.get_joint_vel()[0].clone()
        ee_pos = world.get_ee_pos()[0]            # (3,) local frame

        # ------------------------------------------------------------------
        # 6. Logging
        # ------------------------------------------------------------------
        dist = torch.linalg.norm(ee_pos - goal_now).item()
        ee_pos  = world.get_ee_pos()   # (num_envs, 3)
        ee_quat = world.get_ee_quat()  # (num_envs, 4)
        goal    = world.get_goal()     # (3,)

        tcp_pos = ee_pos + _quat_apply(ee_quat, tcp_offset_local)  # (num_envs, 3)

        dist = torch.linalg.norm(tcp_pos - goal.unsqueeze(0), dim=1, ord=1)
        elapsed = time.time() - t_prev
        t_prev = time.time()
        print(
            # f"\r[{step:05d}] "
            # f"EE [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]  "
            # f"goal [{goal_now[0]:.2f}, {goal_now[1]:.2f}, {goal_now[2]:.2f}]  "
            f"dist {dist[0].item():.4f} m  "
            f"{tof_status()}"
            f"{search_status()}"
            f"{elapsed*1000:.0f} ms/step",
            end="",
            flush=True,
        )

        if perceiving and (step + 1) % 100 == 0:
            print(f"\n[perception] step {step + 1}: {perception_report()}", flush=True)

        if searching and target_zones_seen() >= found_min:
            moved = sum(int(torch.linalg.norm(world.get_object_pos(i)[0] - known_start[i]) > 0.02)
                        for i in known_idx)
            print(f"\n[world] TARGET FOUND at step {step + 1} ({time.time() - t_start:.1f} s wall, "
                  f"{(step + 1) * cfg.isaaclab.dt:.1f} s sim); objects moved: {moved}", flush=True)
            break

    if perceiving:
        print(f"\n[perception] final: {perception_report()}", flush=True)
    print("\n[world] Done.")

    # Keep the viewer alive so the scene can still be explored: hold the robot still and keep
    # stepping/rendering until the window is closed. The last search drawing stays on screen.
    if not headless:
        print("[world] Holding the robot; explore the scene, close the window to exit.", flush=True)
        hold = torch.zeros(1, DOF, device=device)
        while simulation_app.is_running():
            world.apply_robot_cmd(hold)
            world.step()


if __name__ == "__main__":
    main()
    # simulation_app.close() can hang indefinitely in Kit's shutdown; nothing is left to save,
    # so end the process directly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
