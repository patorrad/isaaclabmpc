"""World runner for UR16e push task (Isaac Lab backend).

Single rendered environment.  Calls the MPPI planner server (planner.py)
via zerorpc to get joint-velocity commands.  MPPI rollout trajectories and
goal are drawn in the viewer each step.

Start the planner first, then this runner:

    # Terminal 1 — headless MPPI planner:
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_push/planner.py

    # Terminal 2 — rendered world:
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_push/world.py

    # Headless (no viewer, no rollout vis):
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

parser = argparse.ArgumentParser(description="UR16e push world runner")
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
import csv
import os
import sys
import time
import threading

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import zerorpc
from dataclasses import dataclass, field
from typing import List

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabWrapper, IsaacLabConfig
from isaaclab.sim import RigidBodyPropertiesCfg
from isaaclab_mpc.utils.transport import torch_to_bytes, bytes_to_torch
from robots.ur16e import make_ur16e_cfg
from examples.ur16e_push.box_cfg import make_box_cfg


# ===========================================================================
# 3. Config
# ===========================================================================

@dataclass
class IsaacLabCfg:
    dt: float = 1.0 / 60.0


@dataclass
class BoxCfgEntry:
    init_pos: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.025])
    size: List[float] = field(default_factory=lambda: [0.05, 0.05, 0.05])
    mass: float = 1.0


@dataclass
class WorldConfig:
    n_steps: int = 100000
    goal: List[float] = field(default_factory=lambda: [0.6, 0.3, 0.05])
    ee_link_name: str = "wrist_3_link"
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)
    boxes: List[BoxCfgEntry] = field(default_factory=list)
    robot_init_pos: List[float] = field(default_factory=lambda: [0.208, 0.0, 2.075])
    robot_init_joints: List[float] = field(default_factory=lambda: [0.549, -2.2557, 1.0872, 0.8265, 1.5802, 0.5275])

def _load_config(yaml_path: str) -> WorldConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    cfg = WorldConfig()
    cfg.n_steps      = raw.get("n_steps",      cfg.n_steps)
    cfg.goal         = raw.get("goal",         cfg.goal)
    cfg.ee_link_name = raw.get("ee_link_name", cfg.ee_link_name)
    cfg.robot_init_pos    = raw.get("robot_init_pos",    cfg.robot_init_pos)
    cfg.robot_init_joints = raw.get("robot_init_joints", cfg.robot_init_joints)

    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(dt=il.get("dt", 1.0 / 60.0))
    cfg.boxes = [
        BoxCfgEntry(
            init_pos=b.get("init_pos", [0.0, 0.2, 0.025]),
            size=b.get("size", [0.05, 0.05, 0.05]),
            mass=b.get("mass", 1.0),
        )
        for b in raw.get("boxes", [])
    ]
    return cfg


# ===========================================================================
# 4. Keyboard goal control
# ===========================================================================

class GoalController:
    STEP = 0.02

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
                    if key == keyboard.Key.up:        delta[0] =  self.STEP
                    elif key == keyboard.Key.down:    delta[0] = -self.STEP
                    elif key == keyboard.Key.right:   delta[1] = -self.STEP
                    elif key == keyboard.Key.left:    delta[1] =  self.STEP
                    elif key == keyboard.Key.page_up: delta[2] =  self.STEP
                    elif key == keyboard.Key.page_down: delta[2] = -self.STEP
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
    ROLLOUT_COLOR = (0.1, 0.9, 0.1, 0.25)
    ROLLOUT_WIDTH = 1
    GOAL_COLOR    = (1.0, 0.4, 0.0, 1.0)
    GOAL_SIZE     = 15.0

    TCP_OFFSET_LOCAL = torch.tensor([0.0, 0.0, 0.12])

    def __init__(self):
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()

    def update(
        self,
        rollouts_bytes: bytes,
        goal: torch.Tensor,
        ee_quat_world: torch.Tensor,
        env_origin: torch.Tensor,
        n_draw: int,
    ):
        self._draw.clear_lines()
        self._draw.clear_points()

        origin = env_origin.cpu()
        tcp_offset_world = _quat_apply(ee_quat_world.cpu(), self.TCP_OFFSET_LOCAL)

        # Goal marker
        gp = tuple((goal.cpu() + origin).tolist())
        self._draw.draw_points([gp], [self.GOAL_COLOR], [self.GOAL_SIZE])

        if n_draw <= 0:
            return

        rollouts = bytes_to_torch(rollouts_bytes)   # (horizon, num_envs, 3)
        if rollouts.shape[0] < 1 or rollouts.shape[1] < 1:
            return

        rollouts = rollouts.permute(1, 0, 2).cpu() + origin + tcp_offset_world
        num_envs = rollouts.shape[0]
        stride   = max(1, num_envs // n_draw)
        for traj in rollouts[::stride]:
            pts = [tuple(p.tolist()) for p in traj]
            self._draw.draw_lines_spline(pts, self.ROLLOUT_COLOR, self.ROLLOUT_WIDTH, False)


# ===========================================================================
# 6. Round-trip timing: live plot + CSV saver
# ===========================================================================

class TimingMonitor:
    """Collects planner round-trip latency, shows a plot and saves CSV/PNG when done."""

    def __init__(self, save_path: str):
        self._save_path = save_path
        self._steps: list = []
        self._rtt_ms: list = []

    def record(self, step: int, rtt_ms: float):
        self._steps.append(step)
        self._rtt_ms.append(rtt_ms)

    def save(self):
        if not self._steps:
            return

        ys = np.array(self._rtt_ms)

        # Save CSV
        with open(self._save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "rtt_ms"])
            writer.writerows(zip(self._steps, self._rtt_ms))

        # Save npy (flat RTT array, same format as Genesis rtt_log.npy)
        npy_path = self._save_path.replace(".csv", ".npy")
        np.save(npy_path, ys)

        print(f"\n[timing] saved {self._save_path} and {npy_path}", flush=True)

        # Build and show plot
        xs = np.array(self._steps)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, ys, lw=1, color="steelblue", label="RTT")
        ax.axhline(float(np.mean(ys)), color="orange", lw=1.5, linestyle="--",
                   label=f"mean {np.mean(ys):.1f} ms")
        ax.set_xlabel("step")
        ax.set_ylabel("RTT (ms)")
        ax.set_title("Planner round-trip latency")
        ax.legend()
        fig.tight_layout()

        img_path = self._save_path.replace(".csv", ".png")
        fig.savefig(img_path, dpi=150)
        plt.show(block=True)


# ===========================================================================
# 7. Control loop
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)
    cfg.n_steps = args_cli.n_steps
    headless = getattr(args_cli, "headless", False)
    n_rollouts_draw = 0 if headless else args_cli.n_rollouts_draw

    object_cfgs = [make_box_cfg(b.size, b.mass, b.init_pos) for b in cfg.boxes]
    
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
    world = IsaacLabWrapper(
        cfg=IsaacLabConfig(dt=cfg.isaaclab.dt, device="cuda:0", render=not headless),
        robot_cfg=robot_cfg,
        num_envs=1,
        ee_link_name=cfg.ee_link_name,
        goal=cfg.goal,
        object_cfgs=object_cfgs,
    )
    device = world.device
    DOF    = world.num_dof

    GoalController(world._goal, world._goal_lock)
    vis = RolloutVisualiser() if not headless else None

    print(f"[world] Connecting to planner at {args_cli.planner_addr} …", flush=True)
    planner = zerorpc.Client(timeout=60, heartbeat=None)
    planner.connect(args_cli.planner_addr)
    planner.test("world connected")
    print("[world] Connected.", flush=True)

    q     = world.get_joint_pos()[0].clone()
    dq    = world.get_joint_vel()[0].clone()

    log_dir  = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"timing_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    monitor  = TimingMonitor(csv_path)

    GOAL_THRESHOLD = 0.02  # metres

    print(f"[world] Goal: {cfg.goal}")

    for step in range(cfg.n_steps):
        if not simulation_app.is_running():
            break

        # 1. Sync goal to planner
        with world._goal_lock:
            goal_now = world._goal.clone()
        planner.set_goal(torch_to_bytes(goal_now.cpu()))

        # 2. Read object state and call MPPI planner (timed)
        box_pos  = world.get_object_pos(0)[0]
        box_quat = world.objects[0].data.root_link_quat_w[0]
        dof_state = torch.cat([q, dq, box_pos, box_quat])
        t_rtt = time.perf_counter()
        u_bytes = planner.compute_action_tensor(torch_to_bytes(dof_state), b"")
        rtt_ms  = (time.perf_counter() - t_rtt) * 1e3
        u = bytes_to_torch(u_bytes).to(device)

        monitor.record(step, rtt_ms)

        # 3. Visualise rollouts + goal
        if vis is not None:
            rollout_bytes = planner.get_rollouts()
            origin        = world.scene.env_origins[0]
            ee_quat       = world.get_ee_quat()[0]
            vis.update(rollout_bytes, goal_now, ee_quat, origin, n_rollouts_draw)

        # 4. Apply command and step
        world.apply_robot_cmd(u.view(1, DOF))
        world.step()

        # 5. Read new state
        q   = world.get_joint_pos()[0].clone()
        print(q)
        dq  = world.get_joint_vel()[0].clone()
        box_pos_now = world.get_object_pos()[0]

        # 6. Logging
        dist = torch.linalg.norm(box_pos_now.cpu() - goal_now.cpu()).item()
        print(
            f"\r[{step:05d}] "
            f"box [{box_pos_now[0]:.3f}, {box_pos_now[1]:.3f}, {box_pos_now[2]:.3f}]  "
            f"goal [{goal_now[0]:.2f}, {goal_now[1]:.2f}, {goal_now[2]:.2f}]  "
            f"dist {dist:.4f} m  "
            f"rtt {rtt_ms:.1f} ms",
            end="", flush=True,
        )

        # # 7. Stop when box reaches goal
        # if dist < GOAL_THRESHOLD:
        #     print(f"\n[world] Goal reached (dist={dist:.4f} m < {GOAL_THRESHOLD} m).")
        #     break

    monitor.save()
    print("\n[world] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
