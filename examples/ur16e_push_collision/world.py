"""World runner for UR16e push+collision task (Isaac Lab backend).

Identical to ur16e_push/world.py — only the config path differs.

Start the planner first, then this runner:

    # Terminal 1 — headless MPPI planner:
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_push_collision/planner.py

    # Terminal 2 — rendered world:
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_push_collision/world.py

    # Headless:
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

parser = argparse.ArgumentParser(description="UR16e push+collision world runner")
parser.add_argument("--n_steps", type=int, default=100000)
parser.add_argument("--planner_addr", type=str, default="tcp://localhost:4242")
parser.add_argument("--n_rollouts_draw", type=int, default=50)
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
from collections import deque
from dataclasses import dataclass, field
from typing import List

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import RigidBodyPropertiesCfg
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabWrapper, IsaacLabConfig
from isaaclab_mpc.utils.transport import torch_to_bytes, bytes_to_torch
from robots.ur16e import UR16E_CFG
from examples.ur16e_push.box_cfg import make_box_cfg


# ===========================================================================
# 3. Config  (mirrors ur16e_push/world.py)
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
    goal: List[float] = field(default_factory=lambda: [0.0, 0.6, 0.025])
    ee_link_name: str = "wrist_3_link"
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)
    boxes: List[BoxCfgEntry] = field(default_factory=list)


def _load_config(yaml_path: str) -> WorldConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    cfg = WorldConfig()
    cfg.n_steps      = raw.get("n_steps",      cfg.n_steps)
    cfg.goal         = raw.get("goal",         cfg.goal)
    cfg.ee_link_name = raw.get("ee_link_name", cfg.ee_link_name)
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
                    if key == keyboard.Key.up:          delta[0] =  self.STEP
                    elif key == keyboard.Key.down:      delta[0] = -self.STEP
                    elif key == keyboard.Key.right:     delta[1] = -self.STEP
                    elif key == keyboard.Key.left:      delta[1] =  self.STEP
                    elif key == keyboard.Key.page_up:   delta[2] =  self.STEP
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
    TCP_OFFSET_LOCAL = torch.tensor([0.0, 0.0, 0.14])

    def __init__(self):
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()

    def update(self, rollouts_bytes, goal, ee_quat_world, env_origin, n_draw):
        self._draw.clear_lines()
        self._draw.clear_points()

        origin = env_origin.cpu()
        tcp_offset_world = _quat_apply(ee_quat_world.cpu(), self.TCP_OFFSET_LOCAL)

        gp = tuple((goal.cpu() + origin).tolist())
        self._draw.draw_points([gp], [self.GOAL_COLOR], [self.GOAL_SIZE])

        if n_draw <= 0:
            return

        rollouts = bytes_to_torch(rollouts_bytes)
        if rollouts.shape[0] < 1 or rollouts.shape[1] < 1:
            return

        rollouts = rollouts.permute(1, 0, 2).cpu() + origin + tcp_offset_world
        stride = max(1, rollouts.shape[0] // n_draw)
        for traj in rollouts[::stride]:
            pts = [tuple(p.tolist()) for p in traj]
            self._draw.draw_lines_spline(pts, self.ROLLOUT_COLOR, self.ROLLOUT_WIDTH, False)


# ===========================================================================
# 6. Timing monitor
# ===========================================================================

class TimingMonitor:
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
        with open(self._save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "rtt_ms"])
            writer.writerows(zip(self._steps, self._rtt_ms))
        npy_path = self._save_path.replace(".csv", ".npy")
        np.save(npy_path, ys)
        print(f"\n[timing] saved {self._save_path} and {npy_path}", flush=True)
        xs = np.array(self._steps)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, ys, lw=1, color="steelblue", label="RTT")
        ax.axhline(float(np.mean(ys)), color="orange", lw=1.5, linestyle="--",
                   label=f"mean {np.mean(ys):.1f} ms")
        ax.set_xlabel("step")
        ax.set_ylabel("RTT (ms)")
        ax.set_title("Planner round-trip latency (push+collision)")
        ax.legend()
        fig.tight_layout()
        img_path = self._save_path.replace(".csv", ".png")
        fig.savefig(img_path, dpi=150)
        plt.show(block=True)


# ===========================================================================
# 7. Live wrist force plot
# ===========================================================================

class LiveForcePlotter:
    """Rolling live plot of wrist F/T sensor data (Fx, Fy, Fz, |F|).

    Opens a non-blocking matplotlib window that updates every ``update_every``
    steps.  On ``save()``, writes a .npy and a static .png.
    """

    WINDOW   = 500   # number of steps visible at once
    COLORS   = {"Fx": "tomato", "Fy": "steelblue", "Fz": "seagreen", "|F|": "darkorange"}

    def __init__(self, save_path: str, update_every: int = 5):
        self._save_path   = save_path
        self._update_every = update_every
        self._steps: list = []
        self._fx: list    = []
        self._fy: list    = []
        self._fz: list    = []
        self._fn: list    = []

        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(10, 4))
        self._fig.suptitle("Wrist F/T sensor — contact force")
        self._lines = {
            "Fx": self._ax.plot([], [], lw=1, color=self.COLORS["Fx"],   label="Fx")[0],
            "Fy": self._ax.plot([], [], lw=1, color=self.COLORS["Fy"],   label="Fy")[0],
            "Fz": self._ax.plot([], [], lw=1, color=self.COLORS["Fz"],   label="Fz")[0],
            "|F|": self._ax.plot([], [], lw=1.5, color=self.COLORS["|F|"], label="|F|")[0],
        }
        self._ax.set_xlabel("step")
        self._ax.set_ylabel("force (N)")
        self._ax.legend(loc="upper left")
        self._fig.tight_layout()
        plt.pause(0.001)

    def record(self, step: int, force_xyz: torch.Tensor):
        """Record one sample.  force_xyz: (3,) tensor in world frame."""
        fx, fy, fz = force_xyz.cpu().tolist()
        fn = float(torch.linalg.norm(force_xyz).item())
        self._steps.append(step)
        self._fx.append(fx)
        self._fy.append(fy)
        self._fz.append(fz)
        self._fn.append(fn)

        if len(self._steps) % self._update_every == 0:
            self._refresh()

    def _refresh(self):
        xs = self._steps[-self.WINDOW:]
        for key, ys in [("Fx", self._fx), ("Fy", self._fy),
                        ("Fz", self._fz), ("|F|", self._fn)]:
            self._lines[key].set_data(xs, ys[-self.WINDOW:])
        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def save(self):
        if not self._steps:
            return
        data = np.array([self._steps, self._fx, self._fy, self._fz, self._fn])
        np.save(self._save_path, data)

        xs = np.array(self._steps)
        fig, ax = plt.subplots(figsize=(10, 4))
        for key, ys, c in [
            ("Fx",  self._fx, self.COLORS["Fx"]),
            ("Fy",  self._fy, self.COLORS["Fy"]),
            ("Fz",  self._fz, self.COLORS["Fz"]),
            ("|F|", self._fn, self.COLORS["|F|"]),
        ]:
            ax.plot(xs, ys, lw=1 if key != "|F|" else 1.5, color=c, label=key)
        ax.set_xlabel("step")
        ax.set_ylabel("force (N)")
        ax.set_title("Wrist F/T sensor — contact force")
        ax.legend()
        fig.tight_layout()
        img_path = self._save_path.replace(".npy", ".png")
        fig.savefig(img_path, dpi=150)
        print(f"[force] saved {self._save_path} and {img_path}", flush=True)
        plt.ioff()
        plt.show(block=True)


# ===========================================================================
# 8. Control loop
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)
    cfg.n_steps = args_cli.n_steps
    headless = getattr(args_cli, "headless", False)
    n_rollouts_draw = 0 if headless else args_cli.n_rollouts_draw

    object_cfgs = [make_box_cfg(b.size, b.mass, b.init_pos) for b in cfg.boxes]

    # Contact sensor on the wrist flange — same location as the physical F/T sensor.
    wrist_sensor_cfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
    )

    # Enable PhysX contact reporting so the sensor receives force data.
    robot_cfg = UR16E_CFG.replace(
        spawn=UR16E_CFG.spawn.replace(
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
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
        contact_sensor_cfgs=[wrist_sensor_cfg],
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

    q  = world.get_joint_pos()[0].clone()
    dq = world.get_joint_vel()[0].clone()

    log_dir  = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    stamp    = time.strftime('%Y%m%d_%H%M%S')
    csv_path   = os.path.join(log_dir, f"timing_{stamp}.csv")
    force_path = os.path.join(log_dir, f"force_{stamp}.npy")
    monitor      = TimingMonitor(csv_path)
    force_plot   = LiveForcePlotter(force_path) if not headless else None

    GOAL_THRESHOLD = 0.02

    print(f"[world] Goal: {cfg.goal}")

    for step in range(cfg.n_steps):
        if not simulation_app.is_running():
            break

        with world._goal_lock:
            goal_now = world._goal.clone()
        planner.set_goal(torch_to_bytes(goal_now.cpu()))

        box_pos  = world.get_object_pos(0)[0]
        box_quat = world.objects[0].data.root_link_quat_w[0]
        dof_state = torch.cat([q, dq, box_pos, box_quat])

        t_rtt  = time.perf_counter()
        u_bytes = planner.compute_action_tensor(torch_to_bytes(dof_state), b"")
        rtt_ms  = (time.perf_counter() - t_rtt) * 1e3
        u = bytes_to_torch(u_bytes).to(device)

        monitor.record(step, rtt_ms)

        if vis is not None:
            rollout_bytes = planner.get_rollouts()
            origin        = world.scene.env_origins[0]
            ee_quat       = world.get_ee_quat()[0]
            vis.update(rollout_bytes, goal_now, ee_quat, origin, n_rollouts_draw)

        world.apply_robot_cmd(u.view(1, DOF))
        world.step()

        q  = world.get_joint_pos()[0].clone()
        dq = world.get_joint_vel()[0].clone()
        box_pos_now = world.get_object_pos(0)[0]

        # Wrist contact force — shape (1, 1, 3) → squeeze to (3,)
        force_xyz = world.get_contact_forces(0)[0, 0]
        force_mag = torch.abs(force_xyz[2]).item()   # vertical (Z) only
        if force_plot is not None:
            force_plot.record(step, force_xyz)

        dist = torch.linalg.norm(box_pos_now.cpu() - goal_now.cpu()).item()
        print(
            f"\r[{step:05d}] "
            f"box [{box_pos_now[0]:.3f}, {box_pos_now[1]:.3f}, {box_pos_now[2]:.3f}]  "
            f"goal [{goal_now[0]:.2f}, {goal_now[1]:.2f}, {goal_now[2]:.2f}]  "
            f"dist {dist:.4f} m  "
            f"Fz {force_mag:.1f} N  "
            f"rtt {rtt_ms:.1f} ms",
            end="", flush=True,
        )

        if dist < GOAL_THRESHOLD:
            print(f"\n[world] Goal reached (dist={dist:.4f} m).")
            break

    monitor.save()
    if force_plot is not None:
        force_plot.save()
    print("\n[world] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
