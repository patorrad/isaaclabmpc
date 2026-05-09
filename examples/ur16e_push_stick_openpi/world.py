"""Isaac Lab world runner for UR16e push-stick task with pi0.5 policy.

Architecture
------------
  Terminal 1 (openpi conda env):
      cd ~/Documents/openpi
      python scripts/serve_policy.py --env droid policy:default --port 8000

  Terminal 2 (isaaclab conda env):
      cd /home/paolo/Documents/isaaclabmpc
      python examples/ur16e_push_stick/world.py [--headless]

Policy interface
----------------
  pi0.5 (DROID config) expects:
    "observation/image"       (224, 224, 3) uint8   — overhead camera
    "observation/wrist_image" (224, 224, 3) uint8   — side/external camera
    "observation/state"       (7,)          float32 — 6 joint pos + 0.0 (gripper pad)
    "prompt"                  str           — language instruction

  pi0.5 returns:
    actions (10, 7) float32 — predicted delta joint positions over 10 steps
                              (columns 0–5: UR16e joints; column 6: gripper, ignored)

Action execution
----------------
  Delta joint positions are converted to velocity commands:
      vel_cmd = action_scale * delta / dt
  The first `action_execute` steps of the horizon are queued; once the queue
  empties the policy is queried again with a fresh observation.
"""

# ===========================================================================
# 1. Simulator bootstrap — must happen first
# ===========================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR16e push-stick world (pi0.5)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

args_cli.enable_cameras = True  # required for Camera sensors
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. All other imports (safe after AppLauncher)
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

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.sim import RigidBodyPropertiesCfg
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabWrapper, IsaacLabConfig
from assets.robots.ur16e import make_ur16e_cfg
from examples.ur16e_push_stick_openpi.scene import (
    make_static_cfgs,
    make_push_object_cfg,
    make_camera_cfgs,
    PUSH_OBJECT_INIT_POS,
    GOAL_MARKER_POS,
)

# ===========================================================================
# 3. Config
# ===========================================================================

@dataclass
class Pi05Cfg:
    host: str = "localhost"
    port: int = 8000
    language_instruction: str = "push the red block to the gray goal marker"
    action_execute: int = 5
    action_scale: float = 0.5


@dataclass
class IsaacLabCfg:
    dt: float = 0.02
    env_spacing: float = 3.0


@dataclass
class WorldConfig:
    n_steps: int = 5000
    robot_init_pos: List[float] = field(default_factory=lambda: [0.208, 0.0, 2.075])
    robot_init_joints: List[float] = field(
        default_factory=lambda: [0.549, -2.2557, 1.0872, 0.8265, 1.5802, 0.5275]
    )
    ee_link_name: str = "wrist_3_link"
    stand_urdf: str = ""
    pi05: Pi05Cfg = field(default_factory=Pi05Cfg)
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)


def _load_config(yaml_path: str) -> WorldConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cfg = WorldConfig()
    cfg.n_steps             = raw.get("n_steps",             cfg.n_steps)
    cfg.robot_init_pos      = raw.get("robot_init_pos",      cfg.robot_init_pos)
    cfg.robot_init_joints   = raw.get("robot_init_joints",   cfg.robot_init_joints)
    cfg.ee_link_name        = raw.get("ee_link_name",        cfg.ee_link_name)
    cfg.stand_urdf          = raw.get("stand_urdf",          cfg.stand_urdf)

    if "pi05" in raw:
        p = raw["pi05"]
        cfg.pi05 = Pi05Cfg(
            host=p.get("host", cfg.pi05.host),
            port=p.get("port", cfg.pi05.port),
            language_instruction=p.get("language_instruction", cfg.pi05.language_instruction),
            action_execute=p.get("action_execute", cfg.pi05.action_execute),
            action_scale=p.get("action_scale", cfg.pi05.action_scale),
        )
    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(
            dt=il.get("dt", cfg.isaaclab.dt),
            env_spacing=il.get("env_spacing", cfg.isaaclab.env_spacing),
        )
    return cfg


# ===========================================================================
# 4. Pi0.5 policy client wrapper
# ===========================================================================

class Pi05Policy:
    """Thin wrapper around the openpi WebSocket policy server."""

    def __init__(self, cfg: Pi05Cfg):
        try:
            from openpi_client import websocket_client_policy
        except ImportError:
            raise ImportError(
                "openpi_client not found. Install it in your isaaclab env:\n"
                "  pip install 'git+https://github.com/Physical-Intelligence/openpi.git"
                "#subdirectory=packages/openpi-client'"
            )
        print(f"[pi05] Connecting to policy server at {cfg.host}:{cfg.port} …", flush=True)
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=cfg.host, port=cfg.port
        )
        self._instruction = cfg.language_instruction
        print("[pi05] Connected.", flush=True)

    def infer(
        self,
        overhead_img: np.ndarray,
        side_img: np.ndarray,
        joint_pos: np.ndarray,
    ) -> np.ndarray:
        """Query pi0.5 and return predicted actions.

        Args:
            overhead_img: (224, 224, 3) uint8 — maps to exterior_image_1_left
            side_img:     (224, 224, 3) uint8 — maps to wrist_image_left
            joint_pos:    (6,) float32 — UR16e joint positions in radians

        Returns:
            actions: (10, 8) float32 — delta joint positions; cols 0-5 are UR16e joints
        """
        # DROID expects 7 joint positions + 1 gripper. Pad UR16e's 6 DOF with zeros.
        joint_pos_7d = np.append(joint_pos.astype(np.float64), 0.0)   # (7,)
        gripper_pos  = np.array([0.0], dtype=np.float64)               # (1,) stick = closed

        obs = {
            "observation/exterior_image_1_left": overhead_img,
            "observation/wrist_image_left":      side_img,
            "observation/joint_position":        joint_pos_7d,
            "observation/gripper_position":      gripper_pos,
            "prompt":                            self._instruction,
        }
        return self._client.infer(obs)["actions"]   # (10, 8) float32


# ===========================================================================
# 5. Logging helper
# ===========================================================================

def _object_to_goal_dist(obj_pos: torch.Tensor, goal_pos: torch.Tensor) -> float:
    """2-D (XY) distance from object to goal, ignoring height."""
    return torch.linalg.norm(obj_pos[:2] - goal_pos[:2]).item()


# ===========================================================================
# 6. Main control loop
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)
    headless = getattr(args_cli, "headless", False)

    # ------------------------------------------------------------------
    # Build robot config (velocity control, gravity disabled for arm)
    # ------------------------------------------------------------------
    _base_robot_cfg = make_ur16e_cfg(
        pos=cfg.robot_init_pos,
        rot=(0, 1, 0, 0),
        joint_pos=cfg.robot_init_joints,
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

    # ------------------------------------------------------------------
    # World: single environment, rendered, with cameras
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
        goal=list(GOAL_MARKER_POS),
        env_spacing=cfg.isaaclab.env_spacing,
        object_cfgs=[make_push_object_cfg()],
        static_cfgs=make_static_cfgs(stand_urdf=cfg.stand_urdf),
        camera_cfgs=make_camera_cfgs(),
    )
    device = world.device
    DOF = world.num_dof   # 6 for UR16e
    dt = cfg.isaaclab.dt

    # ------------------------------------------------------------------
    # Goal position in env-local frame for distance logging
    # ------------------------------------------------------------------
    goal_local = torch.tensor(GOAL_MARKER_POS, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Connect to pi0.5 server
    # ------------------------------------------------------------------
    policy = Pi05Policy(cfg.pi05)

    # ------------------------------------------------------------------
    # Action queue: populated when empty by querying the policy
    # ------------------------------------------------------------------
    action_queue: deque = deque()
    t_prev = time.time()

    print(
        f"[world] Running {cfg.n_steps} steps | "
        f"action_execute={cfg.pi05.action_execute} | "
        f"action_scale={cfg.pi05.action_scale}"
    )
    print(
        f"[world] Push object init: {PUSH_OBJECT_INIT_POS} → "
        f"goal: {GOAL_MARKER_POS}"
    )

    for step in range(cfg.n_steps):
        if not simulation_app.is_running():
            break

        # ------------------------------------------------------------------
        # 1. Read current robot state
        # ------------------------------------------------------------------
        q  = world.get_joint_pos()[0].cpu().numpy()   # (6,)

        # ------------------------------------------------------------------
        # 2. Re-infer when action queue is empty
        # ------------------------------------------------------------------
        if not action_queue:
            overhead_img = world.get_camera_rgb(cam_idx=0, env_idx=0)   # (224,224,3) uint8
            side_img     = world.get_camera_rgb(cam_idx=1, env_idx=0)   # (224,224,3) uint8

            # Write latest frames to /tmp so they can be viewed externally.
            # Open with:  eog /tmp/pi05_overhead.png  (or any auto-refreshing viewer)
            from PIL import Image
            Image.fromarray(overhead_img).save("/tmp/pi05_overhead.png")
            Image.fromarray(side_img).save("/tmp/pi05_side.png")
            if step == 0:
                print("\n[cam] Frames saved to /tmp/pi05_overhead.png and /tmp/pi05_side.png")

            actions = policy.infer(overhead_img, side_img, q)           # (10, 7)

            n_exec = min(cfg.pi05.action_execute, actions.shape[0])
            for i in range(n_exec):
                action_queue.append(actions[i, :DOF].copy())            # (6,) deltas

        # ------------------------------------------------------------------
        # 3. Convert delta joint positions → velocity command
        #    vel = scale * delta / dt
        # ------------------------------------------------------------------
        delta = action_queue.popleft()                                   # (6,) float32
        vel_cmd = cfg.pi05.action_scale * delta / dt
        vel_cmd = np.clip(vel_cmd, -3.14, 3.14)                         # UR16e limit

        u = torch.tensor(vel_cmd, device=device, dtype=torch.float32).view(1, DOF)

        # ------------------------------------------------------------------
        # 4. Step physics
        # ------------------------------------------------------------------
        world.apply_robot_cmd(u)
        world.step()

        # ------------------------------------------------------------------
        # 5. Logging
        # ------------------------------------------------------------------
        obj_pos = world.get_object_pos(idx=0)[0]   # (3,) env-local frame
        dist_xy = _object_to_goal_dist(obj_pos, goal_local)

        elapsed_ms = (time.time() - t_prev) * 1000
        t_prev = time.time()
        print(
            f"\r[{step:05d}]  obj→goal dist {dist_xy:.3f} m  "
            f"queue {len(action_queue)}/{ cfg.pi05.action_execute}  "
            f"{elapsed_ms:.0f} ms/step",
            end="",
            flush=True,
        )

        if dist_xy < 0.05:
            print(f"\n[world] SUCCESS at step {step} — object within 5 cm of goal.")
            break

    print("\n[world] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
