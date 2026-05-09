"""Isaac Lab world runner for UR16e push-stick task — EE delta pose + DLS IK.

Architecture
------------
  Terminal 1 (openpi conda env):
      cd ~/Documents/openpi
      python scripts/serve_policy.py \\
          policy:checkpoint \\
          --policy.config pi05_ur16e_push_ee \\
          --policy.dir /home/paolo/.cache/openpi/openpi-assets/checkpoints/pi05_base \\
          --port 8000

  Terminal 2 (isaaclab conda env):
      cd /home/paolo/Documents/isaaclabmpc
      python examples/ur16e_push_stick_pi05/world_ee.py [--headless]

Policy observation keys (defined in openpi/src/openpi/policies/ur16e_push_policy.py):
    "base_rgb"   (224, 224, 3) uint8  — overhead camera
    "wrist_rgb"  (224, 224, 3) uint8  — side camera
    "joints"     (6,)          float32 — UR16e joint positions [rad]
    "prompt"     str

Policy action output (after UR16ePushEEOutputs):
    (10, 6) float32 — predicted EE position deltas [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]

Low-level controller:
    DLS (Damped Least Squares) Jacobian IK maps EE position delta → joint position delta:
        Δq = J^T (J J^T + λ²I)^{-1} Δee
    Applied as a position target:
        q_des = q_current + Δq
    where J is the 6×6 geometric Jacobian in world frame from PhysX.
"""

# ===========================================================================
# 1. Simulator bootstrap — must happen first
# ===========================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR push-stick world EE (pi0.5-base)")
parser.add_argument("--robot", choices=["ur16e", "ur5e"], default="ur16e",
                    help="Robot model to simulate (default: ur16e)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. All other imports
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
from assets.robots.ur16e import make_ur16e_cfg
from assets.robots.ur5e import make_ur5e_cfg
from examples.ur16e_push_stick_pi05.scene import (
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
    language_instruction: str = "push the red block to the goal"
    action_execute: int = 5
    action_scale: float = 0.3   # EE space: smaller scale (metres/radian per step)
    ee_lambda: float = 0.05     # DLS damping factor


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
    cfg.n_steps           = raw.get("n_steps",           cfg.n_steps)
    cfg.robot_init_pos    = raw.get("robot_init_pos",    cfg.robot_init_pos)
    cfg.robot_init_joints = raw.get("robot_init_joints", cfg.robot_init_joints)
    cfg.ee_link_name      = raw.get("ee_link_name",      cfg.ee_link_name)
    cfg.stand_urdf        = raw.get("stand_urdf",        cfg.stand_urdf)
    if "pi05" in raw:
        p = raw["pi05"]
        cfg.pi05 = Pi05Cfg(
            host=p.get("host", cfg.pi05.host),
            port=p.get("port", cfg.pi05.port),
            language_instruction=p.get("language_instruction", cfg.pi05.language_instruction),
            action_execute=p.get("action_execute", cfg.pi05.action_execute),
            action_scale=p.get("action_scale", cfg.pi05.action_scale),
            ee_lambda=p.get("ee_lambda", cfg.pi05.ee_lambda),
        )
    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(
            dt=il.get("dt", cfg.isaaclab.dt),
            env_spacing=il.get("env_spacing", cfg.isaaclab.env_spacing),
        )
    return cfg


# ===========================================================================
# 4. Pi0.5 policy client
# ===========================================================================

class Pi05Policy:
    def __init__(self, cfg: Pi05Cfg):
        try:
            from openpi_client import websocket_client_policy
        except ImportError:
            raise ImportError(
                "openpi_client not found in this env.\n"
                "  pip install 'git+https://github.com/Physical-Intelligence/openpi.git"
                "#subdirectory=packages/openpi-client'"
            )
        print(f"[pi05] Connecting to {cfg.host}:{cfg.port} …", flush=True)
        self._client = websocket_client_policy.WebsocketClientPolicy(
            host=cfg.host, port=cfg.port
        )
        self._instruction = cfg.language_instruction
        print("[pi05] Connected.", flush=True)

    def infer(self, base_img: np.ndarray, wrist_img: np.ndarray,
              joints: np.ndarray) -> np.ndarray:
        """
        Args:
            base_img:  (224, 224, 3) uint8
            wrist_img: (224, 224, 3) uint8
            joints:    (6,) float32 — joint positions [rad]
        Returns:
            (10, 6) float32 — EE position deltas [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
        """
        obs = {
            "base_rgb":  base_img,
            "wrist_rgb": wrist_img,
            "joints":    joints.astype(np.float32),
            "prompt":    self._instruction,
        }
        return self._client.infer(obs)["actions"]   # (10, 6)


# ===========================================================================
# 5. DLS Jacobian IK
# ===========================================================================

def _dls_ik(J: torch.Tensor, ee_delta: torch.Tensor, lam: float) -> torch.Tensor:
    """Damped Least Squares IK: maps EE position delta → joint position delta.

    Args:
        J:        (N, 6, DOF) geometric Jacobian in world frame.
        ee_delta: (N, 6)      desired EE delta [Δx, Δy, Δz, Δroll, Δpitch, Δyaw].
        lam:      damping factor (larger = more stable near singularities).
    Returns:
        dq: (N, DOF) joint position deltas [rad].
    """
    k = J.shape[1]  # 6
    A = J @ J.transpose(-1, -2) + (lam ** 2) * torch.eye(k, device=J.device, dtype=J.dtype)
    J_dls = J.transpose(-1, -2) @ torch.linalg.inv(A)              # (N, DOF, 6)
    return (J_dls @ ee_delta.unsqueeze(-1)).squeeze(-1)             # (N, DOF)


# ===========================================================================
# 6. Main loop
# ===========================================================================

def main():
    headless = getattr(args_cli, "headless", False)
    robot_name = getattr(args_cli, "robot", "ur16e")
    cfg_file = f"config_{robot_name}.yaml" if robot_name != "ur16e" else "config.yaml"
    cfg_path = os.path.join(os.path.dirname(__file__), cfg_file)
    cfg = _load_config(cfg_path)

    if robot_name == "ur5e":
        _base_robot_cfg = make_ur5e_cfg(
            pos=cfg.robot_init_pos, rot=(1, 0, 0, 0), joint_pos=cfg.robot_init_joints
        )
    else:
        _base_robot_cfg = make_ur16e_cfg(
            pos=cfg.robot_init_pos, rot=(0, 1, 0, 0), joint_pos=cfg.robot_init_joints
        )
    print(f"[world_ee] robot = {robot_name}")
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
        goal=list(GOAL_MARKER_POS),
        env_spacing=cfg.isaaclab.env_spacing,
        object_cfgs=[make_push_object_cfg()],
        static_cfgs=make_static_cfgs(stand_urdf=cfg.stand_urdf),
        camera_cfgs=make_camera_cfgs(),
    )
    device = world.device
    DOF    = world.num_dof   # 6

    goal_local = torch.tensor(GOAL_MARKER_POS, device=device, dtype=torch.float32)

    policy = Pi05Policy(cfg.pi05)
    action_queue: deque = deque()
    t_prev = time.time()

    print(f"[world_ee] {cfg.n_steps} steps | action_execute={cfg.pi05.action_execute} | "
          f"scale={cfg.pi05.action_scale} | lambda={cfg.pi05.ee_lambda}")
    print(f"[world_ee] object {PUSH_OBJECT_INIT_POS} → goal {GOAL_MARKER_POS}")

    frame_idx = 0

    for step in range(cfg.n_steps):
        if not simulation_app.is_running():
            break

        q = world.get_joint_pos()[0].cpu().numpy()   # (6,)

        if not action_queue:
            base_img  = world.get_camera_rgb(cam_idx=0, env_idx=0)
            wrist_img = world.get_camera_rgb(cam_idx=1, env_idx=0)

            Image.fromarray(base_img).save("/tmp/pi05_base.png")
            Image.fromarray(wrist_img).save("/tmp/pi05_wrist.png")
            if frame_idx == 0:
                print("\n[cam] frames → /tmp/pi05_base.png  /tmp/pi05_wrist.png")
            frame_idx += 1

            actions = policy.infer(base_img, wrist_img, q)   # (10, 6) EE deltas

            n_exec = min(cfg.pi05.action_execute, actions.shape[0])
            for i in range(n_exec):
                action_queue.append(actions[i].copy())

        ee_delta = action_queue.popleft()                              # (6,) EE position delta

        # DLS IK: Δee → Δq, then apply as position target
        ee_delta_scaled = cfg.pi05.action_scale * ee_delta
        ee_delta_t = torch.tensor(ee_delta_scaled, device=device, dtype=torch.float32).view(1, 6)

        J    = world.get_ee_jacobian()                                 # (1, 6, 6)
        dq   = _dls_ik(J, ee_delta_t, lam=cfg.pi05.ee_lambda)         # (1, 6) joint delta
        dq   = torch.clamp(dq, -0.1, 0.1)                             # limit per-step joint change

        q_cur = world.get_joint_pos()                                  # (1, 6)
        q_des = q_cur + dq

        world.apply_position_cmd(q_des)
        world.step()

        obj_pos  = world.get_object_pos(idx=0)[0]
        dist_xy  = torch.linalg.norm(obj_pos[:2] - goal_local[:2]).item()
        elapsed  = (time.time() - t_prev) * 1000
        t_prev   = time.time()
        print(
            f"\r[{step:05d}]  dist {dist_xy:.3f} m  "
            f"queue {len(action_queue)}/{cfg.pi05.action_execute}  "
            f"{elapsed:.0f} ms/step",
            end="", flush=True,
        )
        if dist_xy < 0.05:
            print(f"\n[world_ee] SUCCESS at step {step}.")
            break

    print("\n[world_ee] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
