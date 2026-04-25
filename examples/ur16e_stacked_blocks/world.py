"""World runner for UR16e stacked-blocks push task (Isaac Lab backend).

Single rendered environment.  Sends all 4 box states to the MPPI planner
each step so rollout envs stay synchronised with the real world.  The
planner advances through the push steps internally; world.py queries the
current step and goal for logging and visualisation.

Start the planner first, then this runner:

    # Terminal 1 — headless MPPI planner:
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_stacked_blocks/planner.py

    # Terminal 2 — rendered world:
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_stacked_blocks/world.py

    # Headless (no viewer, no rollout vis):
    ... world.py --headless
"""

# ===========================================================================
# 1. Simulator bootstrap — must happen first
# ===========================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR16e stacked-blocks world runner")
parser.add_argument("--n_steps", type=int, default=20000)
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
from isaaclab_mpc.utils.transport import torch_to_bytes, bytes_to_torch
from robots.ur16e import UR16E_CFG
from examples.ur16e_push.box_cfg import make_box_cfg


# ===========================================================================
# 3. Config
# ===========================================================================

@dataclass
class IsaacLabCfg:
    dt: float = 1.0 / 60.0


@dataclass
class BoxCfgEntry:
    init_pos: List[float] = field(default_factory=lambda: [0.4, 0.0, 0.025])
    size: List[float] = field(default_factory=lambda: [0.05, 0.05, 0.05])
    mass: float = 0.5


@dataclass
class WorldConfig:
    n_steps: int = 20000
    ee_link_name: str = "wrist_3_link"
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)
    boxes: List[BoxCfgEntry] = field(default_factory=list)
    steps: List[dict] = field(default_factory=list)


def _load_config(yaml_path: str) -> WorldConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    cfg = WorldConfig()
    cfg.n_steps      = raw.get("n_steps",      cfg.n_steps)
    cfg.ee_link_name = raw.get("ee_link_name", cfg.ee_link_name)
    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(dt=il.get("dt", 1.0 / 60.0))
    cfg.boxes = [
        BoxCfgEntry(
            init_pos=b.get("init_pos", [0.4, 0.0, 0.025]),
            size=b.get("size", [0.05, 0.05, 0.05]),
            mass=b.get("mass", 0.5),
        )
        for b in raw.get("boxes", [])
    ]
    cfg.steps = raw.get("steps", [])
    return cfg


# ===========================================================================
# 4. Rollout + goal visualisation
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
# 5. Round-trip timing monitor
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
        ax.set_title("Planner round-trip latency (stacked blocks)")
        ax.legend()
        fig.tight_layout()
        img_path = self._save_path.replace(".csv", ".png")
        fig.savefig(img_path, dpi=150)
        plt.show(block=True)


# ===========================================================================
# 6. Control loop
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)
    cfg.n_steps = args_cli.n_steps
    headless = getattr(args_cli, "headless", False)
    n_rollouts_draw = 0 if headless else args_cli.n_rollouts_draw

    assert len(cfg.boxes) == 4, "config.yaml must define exactly 4 boxes"
    assert len(cfg.steps) > 0,  "config.yaml must define at least one step"

    object_cfgs = [make_box_cfg(b.size, b.mass, b.init_pos) for b in cfg.boxes]

    world = IsaacLabWrapper(
        cfg=IsaacLabConfig(dt=cfg.isaaclab.dt, device="cuda:0", render=not headless),
        robot_cfg=UR16E_CFG,
        num_envs=1,
        ee_link_name=cfg.ee_link_name,
        goal=cfg.steps[0]["end_pos"],
        object_cfgs=object_cfgs,
    )
    device = world.device
    DOF    = world.num_dof

    vis = RolloutVisualiser() if not headless else None

    print(f"[world] Connecting to planner at {args_cli.planner_addr} …", flush=True)
    planner = zerorpc.Client(timeout=60, heartbeat=None)
    planner.connect(args_cli.planner_addr)
    planner.test("world connected")
    print("[world] Connected.", flush=True)

    total_steps_n = int(bytes_to_torch(planner.get_total_steps()).item())
    print(f"[world] Push sequence ({total_steps_n} steps):")
    for i, s in enumerate(cfg.steps):
        print(f"  Step {i}: obj_idx={s['obj_idx']} → {s['end_pos']}")

    q  = world.get_joint_pos()[0].clone()
    dq = world.get_joint_vel()[0].clone()

    log_dir  = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"timing_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    monitor  = TimingMonitor(csv_path)

    boxes_world = world.objects

    for step in range(cfg.n_steps):
        if not simulation_app.is_running():
            break

        # 1. Check if all steps are done
        current_step_idx = int(bytes_to_torch(planner.get_current_step()).item())
        if current_step_idx >= total_steps_n:
            print(f"\n[world] All {total_steps_n} push steps completed!")
            break

        # 2. Build dof_state: [q(DOF), dq(DOF), b0_pos(3), b0_quat(4), ...]
        parts = [q, dq]
        for box in boxes_world:
            if box is not None:
                bpos  = box.data.root_link_pos_w[0]  - world.scene.env_origins[0]
                bquat = box.data.root_link_quat_w[0]
                parts.extend([bpos, bquat])
        dof_state = torch.cat(parts)

        # 3. Call MPPI planner (timed)
        t_rtt  = time.perf_counter()
        u_bytes = planner.compute_action_tensor(torch_to_bytes(dof_state), b"")
        rtt_ms  = (time.perf_counter() - t_rtt) * 1e3
        u = bytes_to_torch(u_bytes).to(device)

        monitor.record(step, rtt_ms)

        # 4. Get current step goal from planner (advances automatically)
        goal_now = bytes_to_torch(planner.get_goal()).to(device)

        # 5. Visualise rollouts + current goal
        if vis is not None:
            rollout_bytes = planner.get_rollouts()
            origin        = world.scene.env_origins[0]
            ee_quat       = world.get_ee_quat()[0]
            vis.update(rollout_bytes, goal_now, ee_quat, origin, n_rollouts_draw)

        # 6. Apply command and step
        world.apply_robot_cmd(u.view(1, DOF))
        world.step()

        # 7. Read new joint state
        q  = world.get_joint_pos()[0].clone()
        dq = world.get_joint_vel()[0].clone()

        # 8. Logging: report active object distance to current goal
        active_obj_idx = cfg.steps[current_step_idx]["obj_idx"]
        active_box = boxes_world[active_obj_idx]
        if active_box is not None:
            box_pos_now = (
                active_box.data.root_link_pos_w[0] - world.scene.env_origins[0]
            )
            dist = torch.linalg.norm(box_pos_now.cpu() - goal_now.cpu()).item()
        else:
            box_pos_now = torch.zeros(3, device=device)
            dist = float("nan")

        print(
            f"\r[{step:05d}] "
            f"step {current_step_idx}/{total_steps_n}  "
            f"obj{active_obj_idx} "
            f"[{box_pos_now[0]:.3f}, {box_pos_now[1]:.3f}, {box_pos_now[2]:.3f}]  "
            f"goal [{goal_now[0]:.2f}, {goal_now[1]:.2f}, {goal_now[2]:.2f}]  "
            f"dist {dist:.4f} m  "
            f"rtt {rtt_ms:.1f} ms",
            end="", flush=True,
        )

    monitor.save()
    print("\n[world] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
