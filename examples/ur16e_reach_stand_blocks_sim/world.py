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
from examples.ur16e_reach_stand_blocks_sim.scene import make_static_cfgs, make_block_cfgs


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
# 4. Rollout + goal visualisation
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
    ROLLOUT_COLOR = (0.1, 0.9, 0.1, 0.25)
    ROLLOUT_WIDTH = 1
    TARGET_COLOR = (1.0, 0.1, 0.1, 1.0)
    TARGET_SIZE  = 20.0

    def __init__(self, tcp_offset_local: torch.Tensor):
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self.tcp_offset_local = tcp_offset_local

    def update(
        self,
        rollouts_bytes: bytes,
        ee_quat_world: torch.Tensor,
        env_origin: torch.Tensor,
        n_draw: int,
        target: torch.Tensor = None,
    ):
        self._draw.clear_lines()
        self._draw.clear_points()

        origin = env_origin.cpu()

        # Rotate TCP offset into world frame using current EE orientation
        tcp_offset_world = _quat_apply(ee_quat_world.cpu(), self.tcp_offset_local)

        if target is not None:
            tp = tuple((target.cpu() + origin).tolist())
            self._draw.draw_points([tp], [self.TARGET_COLOR], [self.TARGET_SIZE])

        # ---- rollout trajectories ----
        if n_draw <= 0:
            return

        rollouts = bytes_to_torch(rollouts_bytes)   # (horizon, num_envs, 3)
        if rollouts.shape[0] < 1 or rollouts.shape[1] < 1:
            return

        # local → world, then shift to TCP tip
        rollouts = rollouts.permute(1, 0, 2).cpu() + origin + tcp_offset_world  # (num_envs, H, 3)
        num_envs = rollouts.shape[0]
        # print(rollouts[0, 0, :])
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

    if not headless:
        world.sim_context.set_camera_view(
            eye=[1.0305, 1.0702, 1.882],
            target=[0.0437, -0.0436, 0.7],
        )

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

    print(f"[world] Body names: {list(world.robot.body_names)}")

    t_prev = time.time()

    for step in range(cfg.n_steps):
        if not simulation_app.is_running():
            break

        if not headless:
            try:
                from omni.kit.viewport.utility import get_active_viewport
                from pxr import UsdGeom
                import omni.usd
                vp = get_active_viewport()
                stage = omni.usd.get_context().get_stage()
                cam_prim = stage.GetPrimAtPath(vp.camera_path)
                xf = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(0)
                cam_pos = (xf[3][0], xf[3][1], xf[3][2])
                fwd = (-xf[2][0], -xf[2][1], -xf[2][2])  # camera -Z in world frame
                t = (0.7 - cam_pos[2]) / fwd[2] if abs(fwd[2]) > 1e-6 else 1.0
                cam_target = (cam_pos[0] + t * fwd[0], cam_pos[1] + t * fwd[1], 0.7)
                print(f"[world] Camera pos: {[round(v, 4) for v in cam_pos]}  target: {[round(v, 4) for v in cam_target]}", flush=True)
            except Exception as e:
                print(f"[world] Could not read camera pose: {e}")

        # ------------------------------------------------------------------
        # 1. Call MPPI planner — returns optimal joint-velocity command
        #    Append block states [pos(3), quat(4)] × 4 so the planner can
        #    reset parallel envs to the current real block positions.
        # ------------------------------------------------------------------
        block_states = []
        for i in range(len(world.objects)):
            pos  = world.get_object_pos(i)[0]   # (3,)
            quat = world.get_object_quat(i)[0]  # (4,) w,x,y,z
            block_states.append(pos)
            block_states.append(quat)
        dof_state = torch.cat([q, dq] + block_states)
        u_bytes = planner.compute_action_tensor(torch_to_bytes(dof_state), b"")
        u = bytes_to_torch(u_bytes).to(device)   # (DOF,)

        # ------------------------------------------------------------------
        # 3. Visualise rollouts + goal (before stepping so viewer is current)
        # ------------------------------------------------------------------
        goal = bytes_to_torch(planner.get_goal())
        if vis is not None:
            rollout_bytes = planner.get_rollouts()
            origin = world.scene.env_origins[0]
            ee_quat = world.get_ee_quat()[0]      # (4,) w,x,y,z world frame
            vis.update(rollout_bytes, ee_quat, origin, n_rollouts_draw, target=goal)
            # goal_now = bytes_to_torch(planner.get_current_goal_pos())
            # vis.update(rollout_bytes, goal_now, ee_quat, origin, n_rollouts_draw)

        # # ------------------------------------------------------------------
        # # 2. Visualise rollouts + goal (before stepping so viewer is current)
        # # ------------------------------------------------------------------
        # if vis is not None:
        #     rollout_bytes = planner.get_rollouts()
        #     origin = world.scene.env_origins[0]
        #     ee_quat = world.get_ee_quat()[0]      # (4,) w,x,y,z world frame
        #     goal_now = bytes_to_torch(planner.get_current_goal_pos())
        #     vis.update(rollout_bytes, goal_now, ee_quat, origin, n_rollouts_draw)

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
        # 6. Logging — show TCP tip distance to current block goal
        # ------------------------------------------------------------------
        try:
            step_info = bytes_to_torch(planner.get_current_step()).item()
            total_steps = bytes_to_torch(planner.get_total_steps()).item()
            step_label = f"step {int(step_info)}/{int(total_steps)}"
        except Exception:
            step_label = ""

        block_pos_strs = "  ".join(
            f"b{i}[{world.get_object_pos(i)[0, 0]:.3f},{world.get_object_pos(i)[0, 1]:.3f},{world.get_object_pos(i)[0, 2]:.3f}]"
            for i in range(len(world.objects))
        )
        elapsed = time.time() - t_prev
        t_prev = time.time()
        print(
            f"\r[{step:05d}] "
            # f"EE [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]  "
            f"{block_pos_strs}  "
            f"goal [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]  ",
            # f"{step_label}  "
            # f"{elapsed*1000:.0f} ms/step",
            end="",
            flush=True,
        )

    print("\n[world] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
