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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. All other imports
# ===========================================================================
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
from assets.robots.ur16e import make_ur16e_cfg
from examples.ur16e_reach_stand.scene import make_static_cfgs, make_block_cfgs


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
            render=not headless,
        ),
        robot_cfg=robot_cfg,
        num_envs=1,
        ee_link_name=cfg.ee_link_name,
        goal=cfg.goal,
        object_cfgs=make_block_cfgs(),
        static_cfgs=make_static_cfgs(stand_urdf=cfg.stand_urdf),
    )
    device = world.device
    DOF = world.num_dof

    # Keyboard goal control
    GoalController(world._goal, world._goal_lock)

    # TCP offset: offset from wrist_3_link origin to tool tip in wrist_3_link frame.
    # tool0 +Z == wrist_3_link +Z (fixed joint chain has no translation, only rotation).
    # The pipe-nipple gripper cylinder extends 0.14 m along +Z.
    tcp_offset_local = torch.tensor([0.0, 0.0, 0.12])

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

    t_prev = time.time()

    for step in range(cfg.n_steps):
        if not simulation_app.is_running():
            break

        # ------------------------------------------------------------------
        # 1. Sync current goal to planner
        # ------------------------------------------------------------------
        with world._goal_lock:
            goal_now = world._goal.clone()
        planner.set_goal(torch_to_bytes(goal_now.cpu()))
        print(goal_now)

        # ------------------------------------------------------------------
        # 2. Call MPPI planner — returns optimal joint-velocity command
        # ------------------------------------------------------------------
        dof_state = torch.cat([q, dq]).cpu()
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

        # ------------------------------------------------------------------
        # 4. Apply command and step the world simulation
        # ------------------------------------------------------------------
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
            f"{elapsed*1000:.0f} ms/step",
            end="",
            flush=True,
        )

    print("\n[world] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
