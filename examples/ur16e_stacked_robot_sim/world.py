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

parser = argparse.ArgumentParser(description="UR16e world runner")
parser.add_argument("--n_steps", type=int, default=100000)
parser.add_argument("--save_video", action="store_true", default=False,
                    help="Record a video of each episode and save alongside the result JSON.")
parser.add_argument("--solution_timeout_s", type=float, default=None,
                    help="Wall-clock timeout per episode in seconds (None = no limit).")
parser.add_argument("--planner_addr", type=str, default="tcp://localhost:4242")
parser.add_argument("--n_rollouts_draw", type=int, default=50,
                    help="Number of MPPI rollout trajectories to visualise (0 = off)")
parser.add_argument("--scenario", type=str, default=None,
                    help="Path to a scenario YAML file. Overrides hardcoded block positions.")
parser.add_argument("--output_path", type=str, default=None,
                    help="Path to write result JSON (success, steps_completed, elapsed_time_s, …).")
parser.add_argument("--manifest", type=str, default=None,
                    help="JSON manifest listing episodes to run sequentially without relaunching. "
                         "Each entry: {name, scenario_yaml, solution_path, output_path}.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. All other imports
# ===========================================================================
import json
import os
import sys
import time
import threading
from pathlib import Path

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
from examples.ur16e_stacked_robot_sim.scene import make_static_cfgs, make_block_cfgs, _bin_to_mppi_local


# ===========================================================================
# 3. Multi-episode helpers
# ===========================================================================

def _object_states_from_scenario(scenario_yaml: str) -> list:
    """Return [(pos_tensor, quat_tensor), …] for each block, ready for reset_to_state()."""
    with open(scenario_yaml) as f:
        sc = yaml.safe_load(f)
    is_ = sc["initial_state"]
    bin_positions = [is_["target_pos"]] + [o["pos"] for o in is_["obstacles"]]
    quats         = [is_.get("target_quat", [1., 0., 0., 0.])] + [
        o.get("quat", [1., 0., 0., 0.]) for o in is_["obstacles"]
    ]
    return [
        (torch.tensor(_bin_to_mppi_local(pos), dtype=torch.float32),
         torch.tensor(quat,                    dtype=torch.float32))
        for pos, quat in zip(bin_positions, quats)
    ]


def _resolve_steps(solution_path: str, scenario_yaml: str) -> list:
    """Load solution steps and convert positions to MPPI world frame."""
    with open(solution_path) as f:
        sol = json.load(f)
    steps = [dict(s) for s in sol["steps"]]
    frame = sol.get("coordinate_frame", "robot")

    if frame == "bin":
        for step in steps:
            step["end_pos"] = _bin_to_mppi_local(step["end_pos"])
            if "start_pos" in step:
                step["start_pos"] = _bin_to_mppi_local(step["start_pos"])
    elif frame == "target":
        with open(scenario_yaml) as f:
            sc = yaml.safe_load(f)
        t = _bin_to_mppi_local(sc["initial_state"]["target_pos"])
        for step in steps:
            ep = step["end_pos"]
            step["end_pos"] = [t[j] + ep[j] for j in range(3)]
            if "start_pos" in step:
                sp = step["start_pos"]
                step["start_pos"] = [t[j] + sp[j] for j in range(3)]
    # "robot" frame: positions already in MPPI world frame

    return steps


def _setup_episode_video(video_path: str, resolution=(1280, 720), fps: int = 15):
    """Attach an RGB annotator to the perspective viewport and open a streaming mp4 writer.
    Returns (render_product, annotator, video_writer) or (None, None, None) on failure.
    """
    try:
        import omni.replicator.core as rep
        import imageio
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        rp = rep.create.render_product("/OmniverseKit_Persp", resolution)
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        vw = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=7)
        print(f"[world] Video recording: {video_path}")
        return rp, ann, vw
    except Exception as e:
        print(f"[world] Video setup failed: {e}")
        return None, None, None


def _capture_video_frame(ann, vw):
    """Trigger replicator, read RGB frame from annotator, append to video writer."""
    try:
        import omni.replicator.core as rep
        rep.orchestrator.step(rt_subframes=0, delta_time=0.0)
        data = ann.get_data()
        if data is not None and hasattr(data, "shape") and data.size > 0:
            vw.append_data(data[..., :3])  # drop alpha channel
    except Exception:
        pass


def _finalize_episode_video(rp, ann, vw, video_path: str):
    """Close the video writer and detach the annotator."""
    n_frames = 0
    try:
        n_frames = vw.count_frames() if hasattr(vw, "count_frames") else "?"
        vw.close()
    except Exception:
        pass
    try:
        ann.detach()
        rp.destroy()
    except Exception:
        pass
    print(f"[world] Video saved: {video_path} ({n_frames} frames)")


def _run_episode(world, planner, cfg, n_steps: int, vis, n_rollouts_draw: int,
                 stall_steps: int = 700, bin_bounds=None,
                 wall_timeout_s: float | None = None,
                 video_path: str | None = None) -> dict:
    """Run one episode control loop. Returns result dict.

    bin_bounds: (x0, y0, bin_size) in MPPI world frame, or None to use legacy
    step-completion success. When provided, success = target exits through -X
    opening (pos_x < x0) without any non-target block leaving the bin.
    """
    DOF = world.num_dof
    device = world.device

    # Unpack bin geometry for exit detection.
    if bin_bounds is not None:
        _bx0, _by0, _bsize = bin_bounds
        def _target_exited(pos):
            return pos[0] < _bx0 - 0.02
        def _in_bin(pos):
            return (pos[0] >= _bx0 - 0.055) and (pos[0] <= _bx0 + _bsize) \
                   and (_by0 <= pos[1] <= _by0 + _bsize)
    else:
        _target_exited = None
        _in_bin = None

    # Video recording setup.
    _vid_rp, _vid_ann, _vid_vw = None, None, None
    if video_path is not None:
        _vid_rp, _vid_ann, _vid_vw = _setup_episode_video(video_path)

    q  = world.get_joint_pos()[0].clone()
    dq = world.get_joint_vel()[0].clone()
    steps_completed  = 0
    total_steps      = 0
    last_progress_at = 0
    ee_trajectory    = []
    t_start = time.time()
    t_prev  = t_start
    exit_reason    = "step_limit"
    target_exited  = False
    intruder_exited = False

    RECOVERY_STALL    = max(120, stall_steps // 6)   # floor: ~2 s at 60 Hz
    RECOVERY_DURATION = max(70, stall_steps // 10)   # floor: ~1 s — brief face attempt then resume
    _in_recovery      = False
    _recovery_start   = 0
    _recovery_count   = 0  # cycles through offset directions

    # TCP-checkpoint progress: reset stall counter when TCP has moved > threshold
    # over a window, so approach-phase movement counts as progress.
    _TCP_WINDOW     = RECOVERY_STALL // 2
    _TCP_THRESHOLD  = 0.08  # 8 cm net displacement over the window
    _tcp_ckpt_pos   = None
    _tcp_ckpt_step  = 0
    _prev_total_steps = 0

    for step in range(n_steps):
        if not simulation_app.is_running():
            exit_reason = "app_closed"
            break

        block_states = []
        for i in range(len(world.objects)):
            block_states.append(world.get_object_pos(i)[0])
            block_states.append(world.get_object_quat(i)[0])
        dof_state = torch.cat([q, dq] + block_states)
        try:
            u_bytes = planner.compute_action_tensor(torch_to_bytes(dof_state), b"")
        except Exception as e:
            print(f"\n[world] Planner disconnected: {e}. Ending episode.")
            exit_reason = "planner_error"
            break
        u = bytes_to_torch(u_bytes).to(device)

        try:
            new_completed = int(bytes_to_torch(planner.get_current_step()).item())
            total_steps   = int(bytes_to_torch(planner.get_total_steps()).item())
            # First time we learn total_steps, anchor the stall clock here so
            # startup latency doesn't count as stall time.
            if total_steps > 0 and _prev_total_steps == 0:
                last_progress_at = step
            _prev_total_steps = total_steps
            if new_completed > steps_completed:
                last_progress_at = step
                if _in_recovery:
                    _in_recovery = False
                    try:
                        planner.set_goal_offset(0.0, 0.0, 0.0)
                    except Exception:
                        pass
            steps_completed = new_completed
        except Exception:
            pass

        # TCP-checkpoint: if TCP has moved > threshold since last checkpoint,
        # the robot is still making progress (approach or push phase).
        _cur_tcp = world.get_ee_pos()[0]
        if _tcp_ckpt_pos is None:
            _tcp_ckpt_pos  = _cur_tcp.clone()
            _tcp_ckpt_step = step
        elif (step - _tcp_ckpt_step) >= _TCP_WINDOW:
            if torch.linalg.norm(_cur_tcp - _tcp_ckpt_pos).item() > _TCP_THRESHOLD:
                last_progress_at = step
            _tcp_ckpt_pos  = _cur_tcp.clone()
            _tcp_ckpt_step = step

        # Exit recovery mode once duration elapsed; reset last_progress_at so
        # the full stall counter gets a fresh start.
        if _in_recovery and (step - _recovery_start) >= RECOVERY_DURATION:
            _in_recovery = False
            last_progress_at = step
            try:
                planner.set_goal_offset(0.0, 0.0, 0.0)
                print(f"\n[world] Recovery done at step {step}, resuming normal goal.")
            except Exception:
                pass

        # Bin-exit detection (overrides step-completion logic when bin_bounds given).
        if _target_exited is not None and len(world.objects) > 0:
            target_pos = world.get_object_pos(0)[0]
            if _target_exited(target_pos):
                target_exited = True
                # Check all non-target blocks.
                for i in range(1, len(world.objects)):
                    if not _in_bin(world.get_object_pos(i)[0]):
                        intruder_exited = True
                        break
                exit_reason = "target_exited"
                break
        elif total_steps > 0 and steps_completed >= total_steps:
            exit_reason = "success"
            break

        steps_since_progress = step - last_progress_at
        if total_steps > 0 and not _in_recovery and steps_since_progress >= RECOVERY_STALL:
            _in_recovery = True
            _recovery_start = step
            _recovery_count += 1
            # Rotate 90° each recovery so successive attempts approach different faces.
            # Magnitude 0.25 m dominates the typical 0.10-0.15 m X-component of
            # obj_to_goal, which flips push_align to the intended face.
            _FACE_OFFSETS = [
                ( 0.0,  0.25, 0.0),   # south face  (+Y)
                (-0.25, 0.0,  0.0),   # east face   (-X, toward exit)
                ( 0.0, -0.25, 0.0),   # north face  (-Y)
                ( 0.25, 0.0,  0.0),   # west face   (+X, away from exit)
            ]
            rx, ry, rz = _FACE_OFFSETS[(_recovery_count - 1) % 4]
            try:
                planner.set_goal_offset(rx, ry, rz)
                print(f"\n[world] RECOVERY #{_recovery_count}: step {steps_completed}/{total_steps} "
                      f"stuck for {RECOVERY_STALL} steps — goal offset ({rx:+.2f}, {ry:+.2f}, {rz:+.2f}).")
            except Exception:
                pass
        if total_steps > 0 and steps_since_progress >= stall_steps:
            print(f"\n[world] STALLED: step {steps_completed}/{total_steps} "
                  f"unchanged for {stall_steps} world steps.")
            exit_reason = "stalled"
            break
        if wall_timeout_s is not None and (time.time() - t_start) >= wall_timeout_s:
            print(f"\n[world] TIMEOUT: {wall_timeout_s:.0f}s wall-clock limit reached.")
            exit_reason = "timeout"
            break

        if vis is not None:
            rollout_bytes = planner.get_rollouts()
            origin   = world.scene.env_origins[0]
            ee_quat  = world.get_ee_quat()[0]
            goal_now = bytes_to_torch(planner.get_current_goal_pos())
            vis.update(rollout_bytes, goal_now, ee_quat, origin, n_rollouts_draw)

        world.apply_robot_cmd(u.view(1, DOF))
        world.step()

        if _vid_ann is not None and step % 5 == 0:
            _capture_video_frame(_vid_ann, _vid_vw)

        q  = world.get_joint_pos()[0].clone()
        dq = world.get_joint_vel()[0].clone()
        ee_pos = world.get_ee_pos()[0]
        ee_trajectory.append(ee_pos.tolist())

        step_label = f"step {steps_completed}/{total_steps}" if total_steps > 0 else ""
        elapsed = time.time() - t_prev
        t_prev  = time.time()
        print(f"\r[{step:05d}] EE [{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}]"
              f"  {step_label}  {elapsed*1000:.0f}ms/step",
              end="", flush=True)

    if _vid_vw is not None:
        _finalize_episode_video(_vid_rp, _vid_ann, _vid_vw, video_path)

    # Success: target exited AND no intruder left the bin.
    if bin_bounds is not None:
        success = target_exited and not intruder_exited
    else:
        success = exit_reason == "success"

    print(f"\n[world] Episode done: {exit_reason} "
          f"({steps_completed}/{total_steps} steps, {time.time()-t_start:.1f}s)"
          + (f" | target_exited={target_exited} intruder_exited={intruder_exited}"
             if bin_bounds is not None else ""))

    return {
        "success":               success,
        "exit_reason":           exit_reason,
        "target_exited":         target_exited,
        "intruder_exited":       intruder_exited,
        "steps_completed":       steps_completed,
        "total_steps":           total_steps,
        "elapsed_time_s":        round(time.time() - t_start, 3),
        "ee_trajectory":         ee_trajectory,
        "block_positions_final": [world.get_object_pos(i)[0].tolist()
                                  for i in range(len(world.objects))],
    }


# ===========================================================================
# 4. Config
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
# 5. Rollout + goal visualisation
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
    """Draws MPPI rollout trajectories and the goal marker each step."""

    # Rollout line colour (RGBA) and thickness
    ROLLOUT_COLOR = (0.1, 0.9, 0.1, 0.25)
    ROLLOUT_WIDTH = 1
    # Goal sphere colour and size
    GOAL_COLOR = (1.0, 0.4, 0.0, 1.0)
    GOAL_SIZE = 15.0

    def __init__(self, tcp_offset_local: torch.Tensor):
        """
        Args:
            tcp_offset_local: (3,) tool-tip offset in the EE link frame.
                              Set to zeros if tracking the link origin directly.
        """
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
        """Refresh debug geometry for this control step.

        The planner tracks the EE link origin.  To shift rollouts to the
        actual tool tip, we rotate tcp_offset_local into world frame using
        the current EE quaternion and add it to every rollout point — the
        same pattern as genesismpc/examples/ur5_suction/world.py.

        Args:
            rollouts_bytes:  serialised (horizon, num_envs, 3) tensor from planner.
                             Positions are in local (env-relative) frame.
            goal:            (3,) goal position in local frame.
            ee_quat_world:   (4,) current EE quaternion in world frame (w, x, y, z).
            env_origin:      (3,) world-frame offset of env 0 (scene.env_origins[0]).
            n_draw:          how many rollout trajectories to draw (subsampled).
        """
        self._draw.clear_lines()
        self._draw.clear_points()

        origin = env_origin.cpu()

        # Rotate TCP offset into world frame using current EE orientation
        tcp_offset_world = _quat_apply(ee_quat_world.cpu(), self.tcp_offset_local)

        # ---- goal marker (local → world) ----
        gp = tuple((goal.cpu() + origin).tolist())
        self._draw.draw_points([gp], [self.GOAL_COLOR], [self.GOAL_SIZE])

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

    # Determine the initial scenario: manifest first episode, then --scenario, then hardcoded.
    manifest = None
    if args_cli.manifest is not None:
        with open(args_cli.manifest) as f:
            manifest = json.load(f)
        first_scenario = manifest["episodes"][0].get("scenario_yaml")
    else:
        first_scenario = args_cli.scenario

    block_positions = None
    bin_size = None
    if first_scenario is not None:
        with open(first_scenario) as f:
            sc = yaml.safe_load(f)
        is_ = sc["initial_state"]
        bin_positions = [is_["target_pos"]] + [o["pos"] for o in is_["obstacles"]]
        block_positions = [_bin_to_mppi_local(p) for p in bin_positions]
        bin_size = sc.get("bin_size")

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
        object_cfgs=make_block_cfgs(positions=block_positions),
        static_cfgs=make_static_cfgs(stand_urdf=cfg.stand_urdf, bin_size=bin_size),
    )

    if not headless:
        table_top_z = _get_table_top_z()
        lookat = (cfg.viewer_lookat[0], cfg.viewer_lookat[1], table_top_z + cfg.viewer_lookat[2])
        eye    = (cfg.viewer_eye[0],    cfg.viewer_eye[1],    table_top_z + cfg.viewer_eye[2])
        world.sim_context.set_camera_view(eye, lookat)

    tcp_offset_local = torch.tensor([0.0, 0.0, 0.12])
    vis = RolloutVisualiser(tcp_offset_local) if not headless else None

    # ------------------------------------------------------------------
    # Connect to the MPPI planner server
    # ------------------------------------------------------------------
    print(f"[world] Connecting to planner at {args_cli.planner_addr} …", flush=True)
    planner = zerorpc.Client(timeout=60, heartbeat=None)
    planner.connect(args_cli.planner_addr)
    planner.test("world connected")
    print("[world] Connected.", flush=True)
    print(f"[world] Body names: {list(world.robot.body_names)}")

    q_init  = torch.tensor(cfg.robot_init_joints, dtype=torch.float32)
    dq_zero = torch.zeros_like(q_init)

    # ------------------------------------------------------------------
    # Multi-episode mode (--manifest)
    # ------------------------------------------------------------------
    if manifest is not None:
        episodes = manifest["episodes"]
        print(f"[world] Multi-episode mode: {len(episodes)} episodes.")

        for ep_idx, episode in enumerate(episodes):
            if not simulation_app.is_running():
                break
            print(f"\n[world] ── Episode {ep_idx+1}/{len(episodes)}: {episode['name']} ──")

            # Reset robot and blocks to this episode's initial state.
            object_states = _object_states_from_scenario(episode["scenario_yaml"])
            world.reset_to_state(q_init, dq_zero, object_states)
            # Settle physics for a few steps before handing control to MPPI.
            for _ in range(30):
                world.step()

            # Tell the planner which steps to execute.
            steps = _resolve_steps(episode["solution_path"], episode["scenario_yaml"])
            try:
                planner.reset_episode(json.dumps(steps))
            except Exception as e:
                print(f"[world] Planner unavailable for episode {ep_idx+1}: {e}")
                result = {"success": False, "exit_reason": "planner_error",
                          "target_exited": False, "intruder_exited": False,
                          "steps_completed": 0, "total_steps": 0, "elapsed_time_s": 0.0,
                          "ee_trajectory": [], "block_positions_final": []}
                out_path = episode.get("output_path")
                if out_path:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "w") as f:
                        json.dump(result, f)
                continue

            _bin_bounds = (0.35, 0.075, bin_size) if bin_size is not None else None
            _video_path = episode.get("video_path") if args_cli.save_video else None
            result = _run_episode(world, planner, cfg, cfg.n_steps, vis, n_rollouts_draw,
                                  bin_bounds=_bin_bounds,
                                  wall_timeout_s=args_cli.solution_timeout_s,
                                  video_path=_video_path)

            out_path = episode.get("output_path")
            if out_path:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(result, f)
                print(f"[world] Result written to {out_path}")

        print("\n[world] All episodes done.")
        os._exit(0)  # skip simulation_app.close() which hangs on Isaac Lab shutdown

    # ------------------------------------------------------------------
    # Single-episode mode (--scenario / --output_path)
    # ------------------------------------------------------------------
    _bin_bounds = (0.35, 0.075, bin_size) if bin_size is not None else None
    _video_path = args_cli.output_path.replace(".json", ".mp4") \
                  if (args_cli.save_video and args_cli.output_path) else None
    result = _run_episode(world, planner, cfg, cfg.n_steps, vis, n_rollouts_draw,
                          bin_bounds=_bin_bounds,
                          wall_timeout_s=args_cli.solution_timeout_s,
                          video_path=_video_path)

    if args_cli.output_path:
        os.makedirs(os.path.dirname(args_cli.output_path), exist_ok=True)
        with open(args_cli.output_path, "w") as f:
            json.dump(result, f)
        print(f"[world] Result written to {args_cli.output_path}")

    print("\n[world] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
