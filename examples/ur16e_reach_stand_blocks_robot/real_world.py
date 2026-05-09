"""Real-robot visualisation client for ur16e_reach_stand.

Mirrors the real UR16e joint state in the Isaac Lab viewer by reading it from
the MPPI planner server (which caches the state received from the ROS bridge).
No ROS dependency — runs in the Isaac Lab conda environment.

The Isaac Lab sim here is a *visualiser only*: physics are not stepped with
velocity commands.  Each frame we write the real robot's joint positions
directly, render the viewer, draw rollouts + goal, and forward any
keyboard-adjusted goal back to the planner.

Usage:
    # Terminal 1 — MPPI planner (headless):
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_reach_stand/planner.py

    # Terminal 2 — ROS bridge (real robot):
    rosrun aurmr_tasks mppi_bridge_node.py _planner_address:=tcp://127.0.0.1:4242

    # Terminal 3 — this viewer:
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_reach_stand/real_world.py

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

parser = argparse.ArgumentParser(description="UR16e reach-stand real-world viewer")
parser.add_argument("--n_steps", type=int, default=1000000)
parser.add_argument("--planner_addr", type=str, default="tcp://localhost:4242")
parser.add_argument("--n_rollouts_draw", type=int, default=50,
                    help="Number of MPPI rollout trajectories to draw (0 = off)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# Always render — this script exists only to show the viewer
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. All other imports (safe after AppLauncher)
# ===========================================================================
import os
import sys
import time
import threading

import torch
import zerorpc
import yaml
from dataclasses import dataclass, field
from typing import List

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.sim import RigidBodyPropertiesCfg
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabWrapper, IsaacLabConfig
from isaaclab_mpc.utils.transport import torch_to_bytes, bytes_to_torch
from assets.robots.ur16e import make_ur16e_cfg
from examples.ur16e_reach_stand_blocks.scene import make_static_cfgs, make_block_cfgs

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
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)
    stand_urdf: str = ""
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
# 3. Keyboard goal control
# ===========================================================================

class GoalController:
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

    def __init__(self, tcp_offset_local: torch.Tensor):
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self.tcp_offset_local = tcp_offset_local

    def update(self, rollouts_bytes, goal, ee_quat_world, env_origin, n_draw):
        self._draw.clear_lines()
        self._draw.clear_points()

        origin = env_origin.cpu()
        tcp_offset_world = _quat_apply(ee_quat_world.cpu(), self.tcp_offset_local)

        # Goal marker (local → world)
        gp = tuple((goal.cpu() + origin).tolist())
        self._draw.draw_points([gp], [self.GOAL_COLOR], [self.GOAL_SIZE])

        if n_draw <= 0:
            return

        rollouts = bytes_to_torch(rollouts_bytes)   # (horizon, num_envs, 3)
        if rollouts.shape[0] < 1 or rollouts.shape[1] < 1:
            return

        rollouts = rollouts.permute(1, 0, 2).cpu() + origin + tcp_offset_world
        stride = max(1, rollouts.shape[0] // n_draw)
        for traj in rollouts[::stride]:
            pts = [tuple(p.tolist()) for p in traj]
            self._draw.draw_lines_spline(pts, self.ROLLOUT_COLOR, self.ROLLOUT_WIDTH, False)


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    DOF = 6
    dt  = 1.0 / 60.0

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)

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
    # Viewer-only sim: single env, rendered, no physics commands
    # ------------------------------------------------------------------
    world = IsaacLabWrapper(
        cfg=IsaacLabConfig(dt=dt, device="cuda:0", render=True),
        robot_cfg=robot_cfg,
        num_envs=1,
        ee_link_name="wrist_3_link",
        goal=[0.4, 0.2, 0.6],
        object_cfgs=make_block_cfgs(),
        static_cfgs=make_static_cfgs(stand_urdf=cfg.stand_urdf),
    )
    device = world.device

    GoalController(world._goal, world._goal_lock)

    tcp_offset_local = torch.tensor([0.0, 0.0, 0.115])
    vis = RolloutVisualiser(tcp_offset_local)

    # ------------------------------------------------------------------
    # Connect to MPPI planner
    # ------------------------------------------------------------------
    print(f"[real_world] Connecting to planner at {args_cli.planner_addr} …", flush=True)
    planner = zerorpc.Client(timeout=60, heartbeat=None)
    planner.connect(args_cli.planner_addr)
    planner.test("real_world connected")
    print("[real_world] Connected.  Arrow keys / PgUp / PgDn to move goal.", flush=True)

    zeros = torch.zeros(DOF, device=device)
    t_prev = time.time()

    for step in range(args_cli.n_steps):
        if not simulation_app.is_running():
            break

        # ------------------------------------------------------------------
        # 1. Read real robot state cached by the planner from the bridge
        # ------------------------------------------------------------------
        try:
            dof_state = bytes_to_torch(planner.get_robot_state()).to(device)
        except Exception as e:
            print(f"\n[real_world] planner unreachable: {e}", flush=True)
            time.sleep(0.1)
            continue

        q  = dof_state[:DOF]
        dq = dof_state[DOF:DOF * 2] if dof_state.numel() > DOF else zeros

        # ------------------------------------------------------------------
        # 2. Mirror joint state + block positions in the viewer
        # ------------------------------------------------------------------
        try:
            obj_bytes = planner.get_object_states()
            obj_data  = bytes_to_torch(obj_bytes)
            print(obj_data)
            if obj_data.numel() >= 7:
                n = obj_data.numel() // 7
                for i in range(min(n, len(world.objects))):
                    pos  = obj_data[i * 7:     i * 7 + 3].to(device)
                    quat = obj_data[i * 7 + 3: i * 7 + 7].to(device)
                    world._reset_object(world.objects[i], pos, quat)
        except Exception:
            pass

        q_exp  = q.view(1, DOF).expand(world.num_envs, -1).contiguous()
        dq_exp = dq.view(1, DOF).expand(world.num_envs, -1).contiguous()
        world.robot.write_joint_state_to_sim(q_exp, dq_exp)
        world.scene.write_data_to_sim()
        world.sim_context.step(render=True)
        world.scene.update(dt)

        # ------------------------------------------------------------------
        # 3. Forward keyboard-adjusted goal to the planner
        # ------------------------------------------------------------------
        with world._goal_lock:
            goal_now = world._goal.clone()
        planner.set_goal(torch_to_bytes(goal_now.cpu()))

        # ------------------------------------------------------------------
        # 4. Draw rollouts + goal
        # ------------------------------------------------------------------
        try:
            rollout_bytes = planner.get_rollouts()
            ee_quat = world.get_ee_quat()[0]
            origin  = world.scene.env_origins[0]
            vis.update(rollout_bytes, goal_now, ee_quat, origin, args_cli.n_rollouts_draw)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # 5. Logging
        # ------------------------------------------------------------------
        ee_pos = world.get_ee_pos()[0]
        tcp_offset_world = _quat_apply(world.get_ee_quat()[0].cpu(), tcp_offset_local)
        tcp_pos = ee_pos.cpu() + tcp_offset_world

        try:
            cur_step   = int(bytes_to_torch(planner.get_current_step()).item())
            tot_steps  = int(bytes_to_torch(planner.get_total_steps()).item())
            step_label = f"step {cur_step}/{tot_steps}"
        except Exception:
            step_label = ""

        elapsed = time.time() - t_prev
        t_prev = time.time()
        print(
            f"\r[{step:06d}] "
            f"TCP [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]  "
            f"{step_label}  "
            f"{elapsed*1000:.0f} ms/step",
            # f"{obj_data}",
            end="",
            flush=True,
        )

    print("\n[real_world] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
