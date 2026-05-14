"""Demo collector for ur16e_reach_stand_blocks_sim.

Runs the MPPI planner (planner.py) to solve the block-push task and records
observations + actions as numpy episode files.

Run convert_to_lerobot.py in the openpi env afterward to build the LeRobot
dataset for fine-tuning pi0.5.

Architecture
------------
  Terminal 1 — headless MPPI planner:
      python examples/ur16e_reach_stand_blocks_sim/planner.py --headless

  Terminal 2 — demo collector:
      python examples/ur16e_reach_stand_blocks_sim/collect_demos.py

Output (one .npz per episode):
  --out_dir/episode_0000.npz, episode_0001.npz, ...

Each .npz contains:
  base_rgb   (T, 224, 224, 3) uint8  — overhead camera
  wrist_rgb  (T, 224, 224, 3) uint8  — side camera
  joints     (T, 6)           float32 — joint positions [rad]
  actions    (T, 7)           float32 — joint delta [vel*dt, 0.0 gripper]
  task       str                      — language instruction
"""

# ===========================================================================
# 1. Simulator bootstrap
# ===========================================================================
import sys
import argparse
from isaaclab.app import AppLauncher

if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

parser = argparse.ArgumentParser(description="Demo collector — ur16e reach blocks")
parser.add_argument("--n_episodes",    type=int, default=50,
                    help="Number of successful episodes to collect")
parser.add_argument("--max_steps_ep",  type=int, default=700,
                    help="Max steps per episode before giving up")
parser.add_argument("--planner_addr",  type=str, default="tcp://localhost:4242")
parser.add_argument("--out_dir",       type=str,
                    default="/tmp/ur16e_reach_blocks_demos",
                    help="Directory to save episode .npz files")
parser.add_argument("--success_dist",  type=float, default=0.04,
                    help="EE-to-goal distance threshold for episode success [m]")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. Imports
# ===========================================================================
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
import zerorpc
from dataclasses import dataclass, field
from typing import List
from PIL import Image

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import RigidBodyPropertiesCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabWrapper, IsaacLabConfig
from isaaclab_mpc.utils.transport import torch_to_bytes, bytes_to_torch
from assets.robots.ur16e import make_ur16e_cfg
from examples.ur16e_reach_stand_blocks_sim.scene import make_static_cfgs, make_block_cfgs

# ===========================================================================
# 3. Config loader
# ===========================================================================

@dataclass
class IsaacLabCfg:
    dt: float = 1.0 / 60.0

@dataclass
class WorldConfig:
    n_steps: int = 100000
    goal: List[float] = field(default_factory=lambda: [0.40, 0.10, 0.92])
    ee_link_name: str = "wrist_3_link"
    stand_urdf: str = ""
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)
    robot_init_pos: List[float] = field(default_factory=lambda: [0.208, 0.0, 2.075])
    robot_init_joints: List[float] = field(
        default_factory=lambda: [0.549, -2.2557, 1.0872, 0.8265, 1.5802, 0.5275]
    )

def _load_config(yaml_path: str) -> WorldConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    cfg = WorldConfig()
    cfg.n_steps           = raw.get("n_steps",           cfg.n_steps)
    cfg.goal              = raw.get("goal",               cfg.goal)
    cfg.ee_link_name      = raw.get("ee_link_name",       cfg.ee_link_name)
    cfg.stand_urdf        = raw.get("stand_urdf",         cfg.stand_urdf)
    cfg.robot_init_pos    = raw.get("robot_init_pos",     cfg.robot_init_pos)
    cfg.robot_init_joints = raw.get("robot_init_joints",  cfg.robot_init_joints)
    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(dt=il.get("dt", 1.0 / 60.0))
    return cfg

# ===========================================================================
# 4. Camera helpers
# ===========================================================================

def _look_at_quat(eye, target, world_up=(0., 0., 1.)):
    eye    = np.asarray(eye,      float)
    target = np.asarray(target,   float)
    up     = np.asarray(world_up, float); up /= np.linalg.norm(up)
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    if abs(np.dot(fwd, up)) > 0.99:
        up = np.array([1., 0., 0.])
    right  = np.cross(up, fwd);    right  /= np.linalg.norm(right)
    cam_up = np.cross(fwd, right); cam_up /= np.linalg.norm(cam_up)
    R = np.column_stack([fwd, right, cam_up])
    t = R[0,0] + R[1,1] + R[2,2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25/s; x=(R[2,1]-R[1,2])*s; y=(R[0,2]-R[2,0])*s; z=(R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])
        w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])
        w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s = 2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])
        w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return (float(w), float(x), float(y), float(z))


_WS_CENTER = (0.52, 0.0, 1.26)
_BASE_EYE  = (0.52, 1.4, 1.5)    # +y side view looking toward workspace

# Wrist camera: local offset from wrist_3_link origin.
# Mounted slightly forward (+Z) and to the side, looking along the tool axis.
# Convention "ros": camera optical axis = +Z of this local frame.
_WRIST_CAM_POS = (0.0, 0.05, 0.01)
_WRIST_CAM_ROT = (1.0, 0.0, 0.0, 0.0)  # +45° around local Y: looks forward-down

def _make_camera_cfgs() -> list:
    base_rot = _look_at_quat(_BASE_EYE, _WS_CENTER)
    print(f"[cameras] base rot={tuple(round(v,4) for v in base_rot)}")
    _spawn = sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 20.0),
    )
    return [
        # cam_idx=0 — fixed side camera (base_rgb)
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/BaseCam",
            update_period=0.0, height=224, width=224, data_types=["rgb"],
            spawn=_spawn,
            offset=CameraCfg.OffsetCfg(pos=_BASE_EYE, rot=base_rot, convention="world"),
        ),
        # cam_idx=1 — wrist-mounted camera (wrist_rgb), moves with robot
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/WristCam",
            update_period=0.0, height=224, width=224, data_types=["rgb"],
            spawn=_spawn,
            offset=CameraCfg.OffsetCfg(pos=_WRIST_CAM_POS, rot=_WRIST_CAM_ROT, convention="ros"),
        ),
    ]

# ===========================================================================
# 5. Episode saving (numpy, no lerobot dependency)
# ===========================================================================

def _make_task_prompt(steps: list) -> str:
    parts = []
    for step in steps:
        x, y = step["end_pos"][0], step["end_pos"][1]
        parts.append(f"push the {step['obj_name']} to ({x:.2f}, {y:.2f})")
    return ", then ".join(parts)


def _save_episode(out_dir: str, ep_idx: int, frames: list, task: str):
    os.makedirs(out_dir, exist_ok=True)
    base_rgb  = np.stack([f["base_rgb"]  for f in frames], axis=0)  # (T,224,224,3)
    wrist_rgb = np.stack([f["wrist_rgb"] for f in frames], axis=0)
    joints    = np.stack([f["joints"]    for f in frames], axis=0)   # (T,6)
    actions   = np.stack([f["actions"]   for f in frames], axis=0)   # (T,7)
    path = os.path.join(out_dir, f"episode_{ep_idx:04d}.npz")
    np.savez_compressed(
        path,
        base_rgb=base_rgb,
        wrist_rgb=wrist_rgb,
        joints=joints,
        actions=actions,
        task=np.array(task),
    )
    return path

# ===========================================================================
# 6. Randomised episode configuration
# ===========================================================================

from examples.ur16e_reach_stand_blocks_sim.scene import _BLOCK_SPECS

_TABLE_Z = 1.26
_MIN_SEP  = 0.10   # minimum distance between any two block centres [m]

# Region where blocks may start (visible from base camera)
_START_REGION = dict(x=(0.4, 0.6), y=(-0.3, -0.1))
# Region where goals may land
_GOAL_REGION  = dict(x=(0.4, 0.6), y=(-0.3, -0.1))

# Original solution step structure (obj indices / names preserved)
_BASE_STEPS = [
    {"obj_idx": 1, "obj_name": "blue block"},
    {"obj_idx": 0, "obj_name": "red block"},
]


def _make_region_viz_cfgs() -> list:
    """Thin colored slabs on the table surface marking the start/goal regions."""
    _TABLE_TOP_Z = 1.236  # table center 1.2 + half-height 0.035 + thin slab offset

    def _slab(region, color, z=_TABLE_TOP_Z):
        cx = (region["x"][0] + region["x"][1]) / 2
        cy = (region["y"][0] + region["y"][1]) / 2
        sx = region["x"][1] - region["x"][0]
        sy = region["y"][1] - region["y"][0]
        return AssetBaseCfg(
            prim_path="PLACEHOLDER",
            spawn=sim_utils.CuboidCfg(
                size=(sx, sy, 0.002),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, z)),
        )

    return [
        _slab(_START_REGION, color=(0.2, 0.85, 0.2)),            # green  — start
        _slab(_GOAL_REGION,  color=(0.9, 0.5, 0.1), z=_TABLE_TOP_Z + 0.001),  # orange — goal
    ]


def _sample_pos(region: dict, rng: np.random.Generator) -> list:
    return [
        float(rng.uniform(*region["x"])),
        float(rng.uniform(*region["y"])),
        _TABLE_Z,
    ]


def _far_enough(pos, others, min_sep=_MIN_SEP) -> bool:
    for o in others:
        if np.linalg.norm(np.array(pos[:2]) - np.array(o[:2])) < min_sep:
            return False
    return True


def _sample_episode(rng: np.random.Generator):
    """Sample random start positions and goal positions for one episode.

    Returns:
        block_states: list of (pos_tensor, quat_tensor) — one per block
        steps: list of step dicts with end_pos filled in
    """
    n_blocks = len(_BASE_STEPS)

    # --- start positions ---
    starts = []
    for _ in range(n_blocks):
        for _ in range(200):
            pos = _sample_pos(_START_REGION, rng)
            if _far_enough(pos, starts):
                starts.append(pos)
                break
        else:
            # fallback: use original position from _BLOCK_SPECS
            starts.append(list(_BLOCK_SPECS[len(starts)][0]))

    # --- goal positions ---
    goals = []
    for _ in range(n_blocks):
        for _ in range(200):
            pos = _sample_pos(_GOAL_REGION, rng)
            if _far_enough(pos, goals + starts):
                goals.append(pos)
                break
        else:
            goals.append(_sample_pos(_GOAL_REGION, rng))

    block_states = [
        (
            torch.tensor(starts[i], dtype=torch.float32),
            torch.tensor([1., 0., 0., 0.], dtype=torch.float32),
        )
        for i in range(n_blocks)
    ]

    steps = [
        {**_BASE_STEPS[i], "end_pos": goals[i]}
        for i in range(n_blocks)
    ]

    return block_states, steps

# ===========================================================================
# 7. Main
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)

    _base_robot_cfg = make_ur16e_cfg(
        pos=cfg.robot_init_pos, joint_pos=cfg.robot_init_joints
    )
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

    headless = getattr(args_cli, "headless", True)
    world = IsaacLabWrapper(
        cfg=IsaacLabConfig(dt=cfg.isaaclab.dt, device="cuda:0", render=not headless),
        robot_cfg=robot_cfg,
        num_envs=1,
        ee_link_name=cfg.ee_link_name,
        goal=cfg.goal,
        object_cfgs=make_block_cfgs(),
        static_cfgs=make_static_cfgs(stand_urdf=cfg.stand_urdf) + _make_region_viz_cfgs(),
        camera_cfgs=_make_camera_cfgs(),
    )
    device = world.device
    DOF    = world.num_dof
    dt     = cfg.isaaclab.dt

    init_q  = torch.tensor(cfg.robot_init_joints, device=device, dtype=torch.float32)
    init_dq = torch.zeros(DOF, device=device)

    print(f"[collector] Connecting to planner at {args_cli.planner_addr} …", flush=True)
    planner = zerorpc.Client(timeout=60, heartbeat=None)
    planner.connect(args_cli.planner_addr)
    planner.test("collector connected")
    print("[collector] Connected.", flush=True)

    out_dir = args_cli.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[collector] Saving episodes to {out_dir}")

    # Camera preview
    Image.fromarray(world.get_camera_rgb(0, 0)).save("/tmp/collect_base.png")
    Image.fromarray(world.get_camera_rgb(1, 0)).save("/tmp/collect_wrist.png")
    print("[collector] Camera previews → /tmp/collect_base.png  /tmp/collect_wrist.png")

    rng = np.random.default_rng()
    n_success  = 0
    n_attempts = 0
    t0 = time.time()

    while n_success < args_cli.n_episodes and simulation_app.is_running():
        n_attempts += 1
        ep_frames: list = []
        done = False

        block_states, solution_steps = _sample_episode(rng)
        task_prompt = _make_task_prompt(solution_steps)

        try:
            planner.reset_episode(json.dumps(solution_steps))
        except Exception as e:
            print(f"\n[collector] Warning: planner.reset_episode() failed: {e}", flush=True)
        world.reset_to_state(init_q, init_dq, object_states=block_states)

        q  = world.get_joint_pos()[0].clone()
        dq = world.get_joint_vel()[0].clone()

        for step in range(args_cli.max_steps_ep):
            if not simulation_app.is_running():
                break

            base_img  = world.get_camera_rgb(0, 0)
            wrist_img = world.get_camera_rgb(1, 0)
            joints_np = q.cpu().numpy().astype(np.float32)

            block_states_flat = []
            for i in range(len(world.objects)):
                block_states_flat.append(world.get_object_pos(i)[0])
                block_states_flat.append(world.get_object_quat(i)[0])
            dof_state = torch.cat([q, dq] + block_states_flat)
            u_bytes = planner.compute_action_tensor(torch_to_bytes(dof_state), b"")
            u = bytes_to_torch(u_bytes).to(device)

            action_np = np.concatenate([
                (u.cpu().numpy() * dt).astype(np.float32),
                np.zeros(1, dtype=np.float32),
            ])

            ep_frames.append({
                "base_rgb":  base_img,
                "wrist_rgb": wrist_img,
                "joints":    joints_np,
                "actions":   action_np,
            })

            world.apply_robot_cmd(u.view(1, DOF))
            world.step()

            q  = world.get_joint_pos()[0].clone()
            dq = world.get_joint_vel()[0].clone()

            dists = [
                torch.linalg.norm(
                    world.get_object_pos(step_def["obj_idx"])[0].cpu()
                    - torch.tensor(step_def["end_pos"])
                ).item()
                for step_def in solution_steps
            ]
            done = all(d < args_cli.success_dist for d in dists)

            planner_step  = int(bytes_to_torch(planner.get_current_step()).item())
            planner_total = int(bytes_to_torch(planner.get_total_steps()).item())
            planner_done  = planner_step >= planner_total

            print(
                f"\r\033[2K[attempt {n_attempts}] step {step:04d}  "
                f"dists={[round(d,3) for d in dists]}  "
                f"planner {planner_step}/{planner_total}  "
                f"saved {n_success}/{args_cli.n_episodes}",
                end="", flush=True,
            )

            if done or planner_done:
                break

        if done:
            path = _save_episode(out_dir, n_success, ep_frames, task_prompt)
            n_success += 1
            elapsed = time.time() - t0
            print(
                f"\n[collector] saved {n_success}/{args_cli.n_episodes} ← attempt {n_attempts}  "
                f"({len(ep_frames)} frames, {elapsed:.0f}s total)  {path}",
                flush=True,
            )
        else:
            print(f"\n[collector] attempt {n_attempts} failed — discarding", flush=True)

    print(f"\n[collector] Done. {n_success} episodes in {out_dir}")
    print(f"  Next: python examples/ur16e_reach_stand_blocks_sim/convert_to_lerobot.py --in_dir {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
