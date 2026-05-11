"""World runner for ur16e_reach_stand_blocks_sim with a fine-tuned pi0.5 policy.

Architecture
------------
  Terminal 1 (openpi env):
      cd ~/Documents/openpi
      python scripts/serve_policy.py policy:checkpoint \\
          --policy.config pi05_ur16e_push \\
          --policy.dir checkpoints/pi05_ur16e_push/run_01/5000

  Terminal 2 (isaaclab env):
      cd /home/paolo/Documents/isaaclabmpc
      python examples/ur16e_reach_stand_blocks_sim/world_pi05.py [--headless]
"""

# ===========================================================================
# 1. Simulator bootstrap
# ===========================================================================
import sys
import argparse
from isaaclab.app import AppLauncher

if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

parser = argparse.ArgumentParser(description="ur16e reach blocks — pi0.5 policy")
parser.add_argument("--n_steps",        type=int,   default=5000)
parser.add_argument("--action_execute", type=int,   default=5,
                    help="Actions from each chunk to execute before re-querying")
parser.add_argument("--action_scale",   type=float, default=1.0)
parser.add_argument("--policy_host",    type=str,   default="localhost")
parser.add_argument("--policy_port",    type=int,   default=8000)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. Imports
# ===========================================================================
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import yaml
from PIL import Image

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.sim import RigidBodyPropertiesCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabWrapper, IsaacLabConfig
from assets.robots.ur16e import make_ur16e_cfg
from examples.ur16e_reach_stand_blocks_sim.scene import make_static_cfgs, make_block_cfgs, _BLOCK_SPECS

# ===========================================================================
# 3. Config
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

_WS_CENTER = (0.25, 0.18, 0.84)
_BASE_EYE  = (1.00, 0.18, 1.10)

_WRIST_CAM_POS = (0.0, 0.05, 0.01)
_WRIST_CAM_ROT = (1.0, 0.0, 0.0, 0.0)

def _make_camera_cfgs() -> list:
    base_rot = _look_at_quat(_BASE_EYE, _WS_CENTER)
    _spawn = sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 20.0),
    )
    return [
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/BaseCam",
            update_period=0.0, height=224, width=224, data_types=["rgb"],
            spawn=_spawn,
            offset=CameraCfg.OffsetCfg(pos=_BASE_EYE, rot=base_rot, convention="world"),
        ),
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/WristCam",
            update_period=0.0, height=224, width=224, data_types=["rgb"],
            spawn=_spawn,
            offset=CameraCfg.OffsetCfg(pos=_WRIST_CAM_POS, rot=_WRIST_CAM_ROT, convention="ros"),
        ),
    ]

# ===========================================================================
# 5. Pi0.5 policy client
# ===========================================================================

class Pi05Policy:
    def __init__(self, host: str, port: int):
        try:
            from openpi_client import websocket_client_policy
        except ImportError:
            raise ImportError(
                "openpi_client not found.\n"
                "  pip install 'git+https://github.com/Physical-Intelligence/openpi.git"
                "#subdirectory=packages/openpi-client'"
            )
        print(f"[pi05] Connecting to {host}:{port} …", flush=True)
        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        print("[pi05] Connected.", flush=True)

    def infer(self, base_img: np.ndarray, wrist_img: np.ndarray,
              joints: np.ndarray, prompt: str) -> np.ndarray:
        obs = {
            "base_rgb":  base_img,
            "wrist_rgb": wrist_img,
            "joints":    joints.astype(np.float32),
            "prompt":    prompt,
        }
        return self._client.infer(obs)["actions"]   # (10, 6)

# ===========================================================================
# 6. Episode randomisation (mirrors collect_demos.py)
# ===========================================================================

_TABLE_Z = 1.0
_MIN_SEP  = 0.10
_START_REGION = dict(x=(0.40, 0.65), y=(-0.10, 0.1))
_GOAL_REGION  = dict(x=(0.45, 0.7),  y=(-0.10, 0.1))
_BASE_STEPS = [
    {"obj_idx": 1, "obj_name": "blue block"},
    {"obj_idx": 0, "obj_name": "red block"},
]


def _sample_pos(region: dict, rng: np.random.Generator) -> list:
    return [float(rng.uniform(*region["x"])), float(rng.uniform(*region["y"])), _TABLE_Z]


def _far_enough(pos, others, min_sep=_MIN_SEP) -> bool:
    return all(np.linalg.norm(np.array(pos[:2]) - np.array(o[:2])) >= min_sep for o in others)


def _sample_episode(rng: np.random.Generator):
    n_blocks = len(_BASE_STEPS)
    starts = []
    for _ in range(n_blocks):
        for _ in range(200):
            pos = _sample_pos(_START_REGION, rng)
            if _far_enough(pos, starts):
                starts.append(pos); break
        else:
            starts.append(list(_BLOCK_SPECS[len(starts)][0]))
    goals = []
    for _ in range(n_blocks):
        for _ in range(200):
            pos = _sample_pos(_GOAL_REGION, rng)
            if _far_enough(pos, goals + starts):
                goals.append(pos); break
        else:
            goals.append(_sample_pos(_GOAL_REGION, rng))
    block_states = [
        (torch.tensor(starts[i], dtype=torch.float32),
         torch.tensor([1., 0., 0., 0.], dtype=torch.float32))
        for i in range(n_blocks)
    ]
    steps = [{**_BASE_STEPS[i], "end_pos": goals[i]} for i in range(n_blocks)]
    return block_states, steps


def _make_task_prompt(steps: list) -> str:
    parts = [f"push the {s['obj_name']} to ({s['end_pos'][0]:.2f}, {s['end_pos'][1]:.2f})"
             for s in steps]
    return ", then ".join(parts)


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
            activate_contact_sensors=False,
        )
    )

    headless = getattr(args_cli, "headless", False)
    world = IsaacLabWrapper(
        cfg=IsaacLabConfig(dt=cfg.isaaclab.dt, device="cuda:0", render=not headless),
        robot_cfg=robot_cfg,
        num_envs=1,
        ee_link_name=cfg.ee_link_name,
        goal=cfg.goal,
        object_cfgs=make_block_cfgs(),
        static_cfgs=make_static_cfgs(stand_urdf=cfg.stand_urdf),
        camera_cfgs=_make_camera_cfgs(),
    )
    device = world.device
    DOF    = world.num_dof
    dt     = cfg.isaaclab.dt

    policy = Pi05Policy(args_cli.policy_host, args_cli.policy_port)
    action_queue: deque = deque()

    rng = np.random.default_rng()
    init_q  = torch.tensor(cfg.robot_init_joints, device=device, dtype=torch.float32)
    init_dq = torch.zeros(DOF, device=device)
    success_dist = 0.04

    ep = 0
    step = 0
    while step < args_cli.n_steps and simulation_app.is_running():
        # --- new episode ---
        ep += 1
        block_states, solution_steps = _sample_episode(rng)
        prompt = _make_task_prompt(solution_steps)
        world.reset_to_state(init_q, init_dq, object_states=block_states)
        action_queue.clear()
        print(f"\n[ep {ep}] {prompt}", flush=True)

        t_prev = time.time()
        for ep_step in range(args_cli.n_steps - step):
            if not simulation_app.is_running():
                break

            q = world.get_joint_pos()[0].cpu().numpy().astype(np.float32)

            if not action_queue:
                base_img  = world.get_camera_rgb(0, 0)
                wrist_img = world.get_camera_rgb(1, 0)
                actions = policy.infer(base_img, wrist_img, q, prompt)
                n_exec = min(args_cli.action_execute, actions.shape[0])
                for i in range(n_exec):
                    action_queue.append(actions[i].copy())

            delta   = action_queue.popleft()
            vel_cmd = args_cli.action_scale * delta / dt
            vel_cmd = np.clip(vel_cmd, -3.14, 3.14)

            u = torch.tensor(vel_cmd, device=device, dtype=torch.float32).view(1, DOF)
            world.apply_robot_cmd(u)
            world.step()
            step += 1

            elapsed = (time.time() - t_prev) * 1000
            t_prev  = time.time()

            dists = [
                torch.linalg.norm(
                    world.get_object_pos(s["obj_idx"])[0].cpu()
                    - torch.tensor(s["end_pos"])
                ).item()
                for s in solution_steps
            ]
            done = all(d < success_dist for d in dists)
            print(
                f"\r[ep {ep} | {step:05d}]  dists={[round(d,3) for d in dists]}  "
                f"queue {len(action_queue)}/{args_cli.action_execute}  {elapsed:.0f} ms",
                end="", flush=True,
            )
            if done:
                print(f"\n[ep {ep}] SUCCESS", flush=True)
                break

    print("\n[world] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
