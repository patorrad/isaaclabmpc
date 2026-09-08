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
parser.add_argument("--output_path", type=str, default=None,
                    help="Path to write result JSON (success, steps_completed, "
                         "total_steps, elapsed_time_s, ee_trajectory, block_positions_final).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# Always render — this script exists only to show the viewer
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. All other imports (safe after AppLauncher)
# ===========================================================================
import json
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
from assets.robots.ur16e import make_ur16e_cfg, get_tool_length
from examples.ur16e_stacked_robot.scene import make_static_cfgs, make_block_cfgs, make_bin_wall_rigid_cfgs

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
    viewer_lookat: List[float] = field(default_factory=lambda: [0.25, 0.0, 0.04])
    viewer_eye:    List[float] = field(default_factory=lambda: [1.50, 0.0, 0.60])
    bin_size: float | None = None
    bin_center: List[float] = field(default_factory=lambda: [0.55, 0.275])


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
    cfg.bin_size          = raw.get("bin_size",          None)
    cfg.bin_center        = raw.get("bin_center",        cfg.bin_center)

    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(dt=il.get("dt", 1.0 / 60.0))
    if "viewer" in raw:
        v = raw["viewer"]
        cfg.viewer_lookat = v.get("lookat", cfg.viewer_lookat)
        cfg.viewer_eye    = v.get("eye",    cfg.viewer_eye)
    return cfg


# ===========================================================================
# 4. Camera helpers
# ===========================================================================

def _get_table_top_z() -> float:
    """Query the USD stage for the table prim's world bounding-box top."""
    from pxr import UsdGeom, Usd
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/envs/env_0/Static0")
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim)
    return float(bbox.GetRange().GetMax()[2])

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
    TARGET_COLOR = (1.0, 0.1, 0.1, 1.0)
    TARGET_SIZE  = 20.0

    def __init__(self, tcp_offset_local: torch.Tensor):
        from isaacsim.util.debug_draw import _debug_draw
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self.tcp_offset_local = tcp_offset_local

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

        # Rotate TCP offset into world frame using current EE orientation
        tcp_offset_world = _quat_apply(ee_quat_world.cpu(), self.tcp_offset_local)

        # ---- goal marker (local → world) ----
        gp = tuple((goal.cpu() + origin).tolist())
        self._draw.draw_points([gp], [self.TARGET_COLOR], [self.TARGET_SIZE])

        # ---- rollout trajectories ----
        if n_draw <= 0:
            return

        rollouts = bytes_to_torch(rollouts_bytes)   # (horizon, num_envs, 3)
        if rollouts.shape[0] < 1 or rollouts.shape[1] < 1:
            return

        # local → world, then shift to TCP tip
        rollouts = rollouts.permute(1, 0, 2).cpu() + origin + tcp_offset_world  # (num_envs, H, 3)
        num_envs = rollouts.shape[0]

        stride = max(1, num_envs // n_draw)
        rollouts_sub = rollouts[::stride]

        for traj in rollouts_sub:
            pts = [tuple(p.tolist()) for p in traj]
            self._draw.draw_lines_spline(pts, self.ROLLOUT_COLOR, self.ROLLOUT_WIDTH, False)


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    DOF = 6
    dt  = 1.0 / 60.0

    # Hard-coded calibration: shift the bin in MPPI world frame.
    # Positive = further from robot (+x), positive y = further right.
    BIN_CALIB_X =  -0.05   # viewer bin was 5 cm too far in +x → nudge back
    BIN_CALIB_Y =  0.0

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)
    cfg.bin_center[0] += BIN_CALIB_X
    cfg.bin_center[1] += BIN_CALIB_Y
    n_rollouts_draw = args_cli.n_rollouts_draw

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
    # Connect to MPPI planner *before* building the scene, so the viewer
    # spawns exactly as many blocks as the real bridge is actually tracking
    # instead of falling back to the hardcoded _BLOCK_SPECS count.
    # ------------------------------------------------------------------
    print(f"[real_world] Connecting to planner at {args_cli.planner_addr} …", flush=True)
    planner = zerorpc.Client(timeout=60, heartbeat=None)
    planner.connect(args_cli.planner_addr)
    planner.test("real_world connected")
    print("[real_world] Connected.  Arrow keys / PgUp / PgDn to move goal.", flush=True)

    print("[real_world] Waiting for object states from the bridge …", flush=True)
    block_positions = None
    bin_tracker_pos = None
    while block_positions is None:
        try:
            obj_data = bytes_to_torch(planner.get_object_states())
            # Last 2 entries are bin trackers (back-wall, side), not puzzle blocks.
            if obj_data.numel() >= 14:
                n = obj_data.numel() // 7 - 2
                block_positions = [obj_data[i * 7: i * 7 + 3].tolist() for i in range(n)]
                bin_tracker_pos = obj_data[n * 7: n * 7 + 3].tolist()
        except Exception as e:
            print(f"\n[real_world] waiting for object states: {e}", flush=True)
        if block_positions is None:
            time.sleep(0.2)
    print(f"[real_world] {len(block_positions)} objects reported by bridge.", flush=True)

    # Bin-exit detection (same geometry + convention as pipeline.py's
    # _run_real_robot_scenario and world.py's _run_episode): derive the real
    # bin position from the live bin-tracker object, not the static config —
    # the physical bin doesn't sit at the config.yaml default. Target (block 0)
    # counts as "exited" once past the -X opening; any other block leaving the
    # bin footprint counts as an intruder.
    if cfg.bin_size is not None and bin_tracker_pos is not None:
        _bsize = cfg.bin_size
        _bx0 = bin_tracker_pos[0] - _bsize  # MPPI x at the -X exit boundary
        print(f"[real_world] Bin tracker={bin_tracker_pos[:2]} → "
              f"exit boundary x0={_bx0:.4f}", flush=True)

        def _target_exited(pos):
            return pos[0] < _bx0 #+ 0.02
    else:
        _target_exited = None

    # Update bin_center from live tracker so visual walls match the physical bin.
    if cfg.bin_size is not None and bin_tracker_pos is not None:
        cfg.bin_center = [bin_tracker_pos[0] - cfg.bin_size / 2, bin_tracker_pos[1]]

    # ------------------------------------------------------------------
    # Viewer-only sim: single env, rendered, no physics commands
    # ------------------------------------------------------------------
    world = IsaacLabWrapper(
        cfg=IsaacLabConfig(dt=dt, device="cuda:0", render=True),
        robot_cfg=robot_cfg,
        num_envs=1,
        ee_link_name="wrist_3_link",
        goal=[0.4, 0.2, 0.6],
        object_cfgs=make_block_cfgs(positions=block_positions) + (make_bin_wall_rigid_cfgs(cfg.bin_size, cfg.bin_center) if cfg.bin_size is not None else []),
        static_cfgs=make_static_cfgs(stand_urdf=cfg.stand_urdf,
                                     bin_size=cfg.bin_size,
                                     bin_center=cfg.bin_center,
                                     skip_bin_walls=True),
    )
    device = world.device

    # Set viewer camera — z offsets are above the table surface queried from USD
    table_top_z = _get_table_top_z()
    lookat = (cfg.viewer_lookat[0], cfg.viewer_lookat[1], table_top_z + cfg.viewer_lookat[2])
    eye    = (cfg.viewer_eye[0],    cfg.viewer_eye[1],    table_top_z + cfg.viewer_eye[2])
    world.sim_context.set_camera_view(eye, lookat)

    tcp_offset_local = torch.tensor([0,0, get_tool_length()])
    vis = RolloutVisualiser(tcp_offset_local)

    zeros = torch.zeros(DOF, device=device)
    _fixed_bin_states = None   # (pos, quat) pairs for each bin wall; set once on first read
    t_start = time.time()
    t_prev = t_start
    steps_completed = 0
    total_steps = 0
    ee_trajectory = []
    target_exited = False
    intruder_exited = False

    try:
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
                if planner.is_done():
                    target_exited = True
                    print(f"\n[real_world] planner.is_done() — target exited.", flush=True)
            except Exception:
                pass

            if target_exited:
                print(f"\n[real_world] Target exited the bin (intruder_exited={intruder_exited}).")
                break

            try:
                obj_bytes = planner.get_object_states()
                obj_data  = bytes_to_torch(obj_bytes)

                # Last 2 entries are bin trackers, not puzzle blocks — exclude them.
                if obj_data.numel() >= 14:
                    n = obj_data.numel() // 7 - 2
                    for i in range(min(n, len(world.objects))):
                        pos  = obj_data[i * 7:     i * 7 + 3].to(device)
                        quat = obj_data[i * 7 + 3: i * 7 + 7].to(device)
                        world._reset_object(world.objects[i], pos, quat)
                    # Fix bin walls at the position from the first received frame.
                    if _fixed_bin_states is None:
                        _fixed_bin_states = []
                        for t_idx in range(2):
                            w_idx = n + t_idx
                            if len(world.objects) > w_idx and obj_data.numel() >= (w_idx + 1) * 7:
                                t_pos  = obj_data[w_idx * 7:     w_idx * 7 + 3].to(device)
                                t_quat = obj_data[w_idx * 7 + 3: w_idx * 7 + 7].to(device)
                                _fixed_bin_states.append((w_idx, t_pos.clone(), t_quat.clone()))
                    for w_idx, t_pos, t_quat in _fixed_bin_states:
                        world._reset_object(world.objects[w_idx], t_pos, t_quat)

                    target_pos = obj_data[0:3]
                    print(
                        f"\r[real_world] target block x={target_pos[0]:.4f} "
                        f"y={target_pos[1]:.4f} z={target_pos[2]:.4f}  "
                        f"exit_x={_bx0:.4f}",
                        end="", flush=True,
                    )

                    if _target_exited is not None and not intruder_exited:
                        for i in range(1, n):
                            blk_pos = obj_data[i * 7: i * 7 + 3]
                            if _target_exited(blk_pos):
                                intruder_exited = True
                                # break

            except Exception:
                pass

            q_exp  = q.view(1, DOF).expand(world.num_envs, -1).contiguous()
            dq_exp = dq.view(1, DOF).expand(world.num_envs, -1).contiguous()
            world.robot.write_joint_state_to_sim(q_exp, dq_exp)
            world.scene.write_data_to_sim()
            world.sim_context.step(render=True)
            world.scene.update(dt)

            # ------------------------------------------------------------------
            # 3. Visualise rollouts + goal
            # ------------------------------------------------------------------
            goal_now = None
            if vis is not None:
                try:
                    goal_now = bytes_to_torch(planner.get_current_goal_pos())
                except Exception as _e:
                    print(f"\n[real_world] get_current_goal_pos failed: {_e}", flush=True)
                    goal_now = torch.zeros(3)
                rollout_bytes = planner.get_rollouts()
                origin = world.scene.env_origins[0]
                ee_quat = world.get_ee_quat()[0]
                vis.update(rollout_bytes, goal_now, ee_quat, origin, n_rollouts_draw)

            # ------------------------------------------------------------------
            # 4. Logging
            # ------------------------------------------------------------------
            ee_pos = world.get_ee_pos()[0]
            tcp_offset_world = _quat_apply(world.get_ee_quat()[0].cpu(), tcp_offset_local)
            tcp_pos = ee_pos.cpu() + tcp_offset_world

            try:
                steps_completed = int(bytes_to_torch(planner.get_current_step()).item())
                total_steps     = int(bytes_to_torch(planner.get_total_steps()).item())
                step_label = f"step {steps_completed}/{total_steps}"
            except Exception:
                step_label = ""

            ee_trajectory.append(ee_pos.tolist())

            elapsed = time.time() - t_prev
            t_prev = time.time()
            goal_str = (f"G [{goal_now[0]:.3f},{goal_now[1]:.3f},{goal_now[2]:.3f}]"
                        if goal_now is not None else "G None")
            # print(
            #     f"\r[{step:06d}] "
            #     f"TCP [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]  "
            #     f"{goal_str}  "
            #     f"{step_label}  "
            #     f"{elapsed*1000:.0f} ms/step",
            #     end="",
            #     flush=True,
            # )

    except KeyboardInterrupt:
        print("\n[real_world] Stopped by user (Ctrl+C).", flush=True)

    elapsed_time_s = time.time() - t_start
    if _target_exited is not None:
        success = target_exited and not intruder_exited
    else:
        success = total_steps > 0 and steps_completed >= total_steps
    print(f"\n[real_world] Done: {steps_completed}/{total_steps} steps "
          f"in {elapsed_time_s:.1f}s (success={success})."
          + (f" target_exited={target_exited} intruder_exited={intruder_exited}"
             if _target_exited is not None else ""), flush=True)

    if args_cli.output_path:
        reported_steps = total_steps if (target_exited and total_steps > 0) else steps_completed
        result = {
            "success":               success,
            "target_exited":         target_exited,
            "intruder_exited":       intruder_exited,
            "steps_completed":       reported_steps,
            "total_steps":           total_steps,
            "elapsed_time_s":        round(elapsed_time_s, 3),
            "ee_trajectory":         ee_trajectory,
            "block_positions_final": [world.get_object_pos(i)[0].tolist()
                                      for i in range(len(world.objects))],
        }
        os.makedirs(os.path.dirname(args_cli.output_path), exist_ok=True)
        with open(args_cli.output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[real_world] Result written to {args_cli.output_path}", flush=True)

    # Exit immediately — simulation_app.close() can hang on viewer cleanup
    # and would block pipeline.py from reading the result and writing CSV.
    import sys
    sys.exit(0)


if __name__ == "__main__":
    main()
    simulation_app.close()
