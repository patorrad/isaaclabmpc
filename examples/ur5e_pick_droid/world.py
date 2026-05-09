"""UR5e pick task with pi0.5 DROID policy — in-distribution test.

Architecture
------------
  Terminal 1 (openpi conda env):
      cd ~/Documents/openpi
      python scripts/serve_policy.py \\
          policy:checkpoint \\
          --policy.config pi05_droid \\
          --policy.dir /home/paolo/.cache/openpi/openpi-assets/checkpoints/pi05_base \\
          --port 8000

  Terminal 2 (isaaclab conda env):
      cd /home/paolo/Documents/isaaclabmpc
      python examples/ur5e_pick_droid/world.py [--headless]

DROID observation keys sent to server:
    "observation/exterior_image_1_left"  (224,224,3) uint8 — overhead camera
    "observation/wrist_image_left"       (224,224,3) uint8 — side camera
    "observation/joint_position"         (7,)  float32 — 6 UR5e joints + 1 zero pad
    "observation/gripper_position"       (1,)  float32 — always 0 (no gripper)
    "prompt"                             str

DROID action output (8D):
    [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, ?, gripper]
    Dims 0-5 → DLS IK → joint velocities
    Dim  7   → gripper command (logged, not applied — no physical gripper)
"""

# ===========================================================================
# 1. Simulator bootstrap
# ===========================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR5e pick — DROID policy (pi0.5)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

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
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabWrapper, IsaacLabConfig
from assets.robots.ur5e import make_ur5e_cfg
from examples.ur5e_pick_droid.scene import (
    make_static_cfgs,
    make_pick_object_cfg,
    make_camera_cfgs,
    PICK_OBJECT_INIT_POS,
    PLACE_TARGET_POS,
)

# ===========================================================================
# 3. Config
# ===========================================================================

@dataclass
class DroidCfg:
    host: str = "localhost"
    port: int = 8000
    language_instruction: str = "pick up the red block"
    action_execute: int = 5
    action_scale: float = 0.3
    ee_lambda: float = 0.05


@dataclass
class IsaacLabCfg:
    dt: float = 0.02
    env_spacing: float = 3.0


@dataclass
class WorldConfig:
    n_steps: int = 3000
    robot_init_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.835])
    robot_init_joints: List[float] = field(
        default_factory=lambda: [2.82, -1.63, 0.44, -1.54, -0.39, 0.18]
    )
    ee_link_name: str = "wrist_3_link"
    droid: DroidCfg = field(default_factory=DroidCfg)
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)


def _load_config(path: str) -> WorldConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = WorldConfig()
    cfg.n_steps           = raw.get("n_steps",           cfg.n_steps)
    cfg.robot_init_pos    = raw.get("robot_init_pos",    cfg.robot_init_pos)
    cfg.robot_init_joints = raw.get("robot_init_joints", cfg.robot_init_joints)
    cfg.ee_link_name      = raw.get("ee_link_name",      cfg.ee_link_name)
    if "droid" in raw:
        d = raw["droid"]
        cfg.droid = DroidCfg(
            host=d.get("host", cfg.droid.host),
            port=d.get("port", cfg.droid.port),
            language_instruction=d.get("language_instruction", cfg.droid.language_instruction),
            action_execute=d.get("action_execute", cfg.droid.action_execute),
            action_scale=d.get("action_scale", cfg.droid.action_scale),
            ee_lambda=d.get("ee_lambda", cfg.droid.ee_lambda),
        )
    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(
            dt=il.get("dt", cfg.isaaclab.dt),
            env_spacing=il.get("env_spacing", cfg.isaaclab.env_spacing),
        )
    return cfg


# ===========================================================================
# 4. DROID policy client
# ===========================================================================

class DroidPolicy:
    def __init__(self, cfg: DroidCfg):
        try:
            from openpi_client import websocket_client_policy
        except ImportError:
            raise ImportError(
                "openpi_client not found.\n"
                "  pip install 'git+https://github.com/Physical-Intelligence/openpi.git"
                "#subdirectory=packages/openpi-client'"
            )
        print(f"[droid] Connecting to {cfg.host}:{cfg.port} …", flush=True)
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=cfg.host, port=cfg.port
        )
        self._instruction = cfg.language_instruction
        print("[droid] Connected.", flush=True)

    def infer(self, overhead_img: np.ndarray, side_img: np.ndarray,
              joints: np.ndarray) -> np.ndarray:
        """
        Args:
            overhead_img: (224,224,3) uint8
            side_img:     (224,224,3) uint8
            joints:       (6,) float32 — UR5e joint positions [rad]
        Returns:
            (horizon, 8) float32 — DROID actions
                [:, 0:6] EE delta [Δx,Δy,Δz,Δroll,Δpitch,Δyaw]
                [:, 7]   gripper command (0=close, 1=open)
        """
        # Pad UR5e 6-DOF joints to 7D to match DROID state dimension
        joint_7d = np.concatenate([joints.astype(np.float32), [0.0]])
        obs = {
            "observation/exterior_image_1_left": overhead_img,
            "observation/wrist_image_left":      side_img,
            "observation/joint_position":        joint_7d,
            "observation/gripper_position":      np.zeros(1, dtype=np.float32),
            "prompt":                            self._instruction,
        }
        return self._client.infer(obs)["actions"]   # (horizon, 8)


# ===========================================================================
# 5. DLS Jacobian IK
# ===========================================================================

def _dls_ik(J: torch.Tensor, ee_delta: torch.Tensor, lam: float) -> torch.Tensor:
    """Δq = J^T (J J^T + λ²I)^{-1} Δee.  Shapes: J=(N,6,DOF), ee_delta=(N,6)."""
    k = J.shape[1]
    A = J @ J.transpose(-1, -2) + (lam**2) * torch.eye(k, device=J.device, dtype=J.dtype)
    return (J.transpose(-1, -2) @ torch.linalg.inv(A) @ ee_delta.unsqueeze(-1)).squeeze(-1)


# ===========================================================================
# 6. Main loop
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)
    headless = getattr(args_cli, "headless", False)

    _base_robot_cfg = make_ur5e_cfg(
        pos=cfg.robot_init_pos, rot=(1, 0, 0, 0), joint_pos=cfg.robot_init_joints
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

    world = IsaacLabWrapper(
        cfg=IsaacLabConfig(dt=cfg.isaaclab.dt, device="cuda:0", render=not headless),
        robot_cfg=robot_cfg,
        num_envs=1,
        ee_link_name=cfg.ee_link_name,
        goal=list(PLACE_TARGET_POS),
        env_spacing=cfg.isaaclab.env_spacing,
        object_cfgs=[make_pick_object_cfg()],
        static_cfgs=make_static_cfgs(),
        camera_cfgs=make_camera_cfgs(),
    )
    device = world.device
    DOF    = world.num_dof

    place_target = torch.tensor(PLACE_TARGET_POS, device=device, dtype=torch.float32)

    policy = DroidPolicy(cfg.droid)
    action_queue: deque = deque()
    t_prev  = time.time()
    frame_idx = 0

    print(f"[world] {cfg.n_steps} steps | scale={cfg.droid.action_scale} | "
          f"lambda={cfg.droid.ee_lambda}")
    print(f"[world] pick object at {PICK_OBJECT_INIT_POS}")
    print(f"[world] place target at {PLACE_TARGET_POS}")

    for step in range(cfg.n_steps):
        if not simulation_app.is_running():
            break

        q = world.get_joint_pos()[0].cpu().numpy()   # (6,)

        if not action_queue:
            overhead = world.get_camera_rgb(cam_idx=0, env_idx=0)
            side     = world.get_camera_rgb(cam_idx=1, env_idx=0)

            Image.fromarray(overhead).save("/tmp/droid_overhead.png")
            Image.fromarray(side).save("/tmp/droid_side.png")
            if frame_idx == 0:
                print("\n[cam] frames → /tmp/droid_overhead.png  /tmp/droid_side.png")
            frame_idx += 1

            actions = policy.infer(overhead, side, q)   # (horizon, 8)

            n_exec = min(cfg.droid.action_execute, actions.shape[0])
            for i in range(n_exec):
                action_queue.append(actions[i].copy())

        action    = action_queue.popleft()             # (8,)
        ee_delta  = action[:6]                         # [Δx,Δy,Δz,Δroll,Δpitch,Δyaw]
        gripper   = float(action[7])                   # 0=close, 1=open

        ee_delta_scaled = cfg.droid.action_scale * ee_delta
        ee_t = torch.tensor(ee_delta_scaled, device=device, dtype=torch.float32).view(1, 6)

        J    = world.get_ee_jacobian()                 # (1, 6, 6)
        dq   = _dls_ik(J, ee_t, lam=cfg.droid.ee_lambda)
        dq   = torch.clamp(dq, -0.1, 0.1)

        q_cur = world.get_joint_pos()                  # (1, 6)
        world.apply_position_cmd(q_cur + dq)
        world.step()

        obj_pos = world.get_object_pos(idx=0)[0]
        dist_xy = torch.linalg.norm(obj_pos[:2] - place_target[:2]).item()
        elapsed = (time.time() - t_prev) * 1000
        t_prev  = time.time()
        print(
            f"\r[{step:05d}]  obj_dist_xy {dist_xy:.3f} m  "
            f"gripper {'open' if gripper > 0.5 else 'close'}  "
            f"queue {len(action_queue)}/{cfg.droid.action_execute}  "
            f"{elapsed:.0f} ms/step",
            end="", flush=True,
        )

    print("\n[world] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
