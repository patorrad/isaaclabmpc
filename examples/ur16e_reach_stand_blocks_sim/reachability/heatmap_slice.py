"""IK reachability heatmap for the UR16e — task-constrained orientations.

Tests only end-effector orientations that satisfy the planner's robot_ori
cost (world-frame yaw = pitch = 0).  These are the orientations the MPPI
controller considers "good", so the heatmap directly shows which positions
are reachable under the planner's actual orientation constraint.

robot_ori cost: ||[yaw, pitch]||  where yaw/pitch are ZYX Euler angles of
the EE quaternion in world frame.  Zero cost ↔ only X-roll is non-zero.
In URDF frame (robot mounted 180° around world X) this family is:
    q_urdf = [sin(θ/2), −cos(θ/2), 0, 0]   for θ ∈ [0, 2π)

Usage:
    conda run -n genesistest2 python heatmap_slice.py
    conda run -n genesistest2 python heatmap_slice.py --z 1.2 --res 60 --viser
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR      = Path(__file__).parent / "results"
ROBOT_BASE_WORLD = np.array([0.208, 0.0, 2.075])

TASK_POSITIONS_WORLD = {
    "red_block":  np.array([0.3127, 0.1797, 0.90]),
    "blue_block": np.array([0.1825, 0.1874, 0.90]),
    "goal":       np.array([0.40,   0.10,   0.92]),
}

TASK_COLORS = {
    "red_block":  "#e03030",
    "blue_block": "#3060e0",
    "goal":       "#ff9900",
}

# Orientations satisfying robot_ori = 0  (world-frame yaw = pitch = 0).
# q_urdf = [sin(θ/2), −cos(θ/2), 0, 0]  for 12 evenly-spaced roll angles.
# These are the ONLY orientations the planner's robot_ori cost rewards.
_THETAS = np.linspace(0, 2 * np.pi, 12, endpoint=False)
TEST_ORIENTATIONS_WXYZ = np.stack([
    np.sin(_THETAS / 2),
    -np.cos(_THETAS / 2),
    np.zeros_like(_THETAS),
    np.zeros_like(_THETAS),
], axis=1).astype(np.float32)


# ── Coordinate transform ───────────────────────────────────────────────────────

def world_to_urdf(pos_world: np.ndarray) -> np.ndarray:
    p = np.atleast_2d(pos_world) - ROBOT_BASE_WORLD
    return np.stack([p[:, 0], -p[:, 1], -p[:, 2]], axis=1)


# ── IK sweep ──────────────────────────────────────────────────────────────────

def solve_heatmap(ik: InverseKinematics,
                  xs: np.ndarray, ys: np.ndarray,
                  z_world: float) -> np.ndarray:
    """Return success-rate array of shape (len(xs),) in [0, 1]."""
    n_pos = len(xs)
    n_ori = len(TEST_ORIENTATIONS_WXYZ)

    pos_world = np.stack([xs, ys, np.full(n_pos, z_world)], axis=1).astype(np.float32)
    pos_urdf  = world_to_urdf(pos_world).astype(np.float32)

    chunk = ik.config.max_batch_size
    success_counts = np.zeros(n_pos, dtype=np.int32)

    for ori in TEST_ORIENTATIONS_WXYZ:
        q_rep = np.tile(ori, (n_pos, 1)).astype(np.float32)

        for start in range(0, n_pos, chunk):
            end = min(start + chunk, n_pos)
            pos_t = torch.tensor(pos_urdf[start:end], device="cuda")
            q_t   = torch.tensor(q_rep[start:end],   device="cuda")
            result = ik.solve_pose(
                GoalToolPose.from_poses(
                    {"tool0": Pose(position=pos_t, quaternion=q_t)},
                    num_goalset=1,
                )
            )
            s = result.success.squeeze().cpu().numpy()
            if s.ndim == 0:
                s = np.array([s])
            success_counts[start:end] += s.astype(np.int32)

    return success_counts / n_ori   # success rate ∈ [0, 1]


# ── Matplotlib heatmap ────────────────────────────────────────────────────────

def plot_heatmap(rate_grid: np.ndarray,
                 x_vals: np.ndarray, y_vals: np.ndarray,
                 z_world: float, res: int,
                 out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    cmap = plt.cm.RdYlGn
    im = ax.pcolormesh(x_vals, y_vals, rate_grid,
                       cmap=cmap, vmin=0.0, vmax=1.0, shading="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("IK success rate  (fraction of orientations)", fontsize=11)

    # Robot base (projected onto XY)
    ax.plot(ROBOT_BASE_WORLD[0], ROBOT_BASE_WORLD[1],
            "k+", markersize=16, markeredgewidth=2.5, label="Robot base (projected)")

    # Task positions (mark any within 5 cm of slice height)
    for name, pos in TASK_POSITIONS_WORLD.items():
        if abs(pos[2] - z_world) < 0.05:
            ax.plot(pos[0], pos[1], "o",
                    color=TASK_COLORS[name], markersize=11,
                    markeredgecolor="white", markeredgewidth=1.5,
                    label=name, zorder=5)

    reachable_pct = (rate_grid > 0).mean() * 100
    ax.set_xlabel("World X (m)", fontsize=12)
    ax.set_ylabel("World Y (m)", fontsize=12)
    ax.set_title(
        f"UR16e IK Reachability  —  z = {z_world:.3f} m  "
        f"({len(TEST_ORIENTATIONS_WXYZ)} task orientations,  {res}×{res} grid)\n"
        f"Orientations: world yaw=pitch=0  (robot_ori cost = 0)\n"
        f"{reachable_pct:.1f}% of positions reachable by at least one task orientation",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Viser viewer ──────────────────────────────────────────────────────────────

def launch_viser(xs: np.ndarray, ys: np.ndarray,
                 z_world: float, success_rate: np.ndarray,
                 port: int = 8080) -> None:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf
    from curobo._src.state.state_joint import JointState

    URDF_PATH       = Path("/home/paolo/Documents/curobo/curobo/content/assets/robot/ur_description/urdf/ur16e.urdf")
    MESH_BASE_DIR   = URDF_PATH.parent
    JOINT_NAMES     = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    HOME_JOINTS     = np.array([0.549, -2.2557, 1.0872, 0.8265, 1.5802, 0.5275], dtype=np.float32)
    ROBOT_BASE_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])
    BLOCK_SIZE      = (0.05, 0.05, 0.05)
    _Q_ROBOT_INV    = np.array([0.0, -1.0, 0.0, 0.0])

    def mesh_filename_handler(fname: str) -> str:
        fname = fname.replace("package://", "")
        return str((MESH_BASE_DIR / fname).resolve())

    def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1;  w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def world_to_urdf_pos(p_world: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p_world) - ROBOT_BASE_WORLD
        return np.stack([p[:, 0], -p[:, 1], -p[:, 2]], axis=1)

    def urdf_to_world_pos(p_urdf: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p_urdf)
        return np.stack([p[:, 0] + ROBOT_BASE_WORLD[0],
                         -p[:, 1] + ROBOT_BASE_WORLD[1],
                         -p[:, 2] + ROBOT_BASE_WORLD[2]], axis=1)

    def make_js(joints: np.ndarray) -> JointState:
        return JointState.from_position(
            torch.tensor(joints[None], device="cuda", dtype=torch.float32),
            joint_names=JOINT_NAMES,
        )

    # ── Differential IK solver ────────────────────────────────────────────────
    print("Building differential IK solver ...")
    ik_cfg = InverseKinematicsCfg.create(
        robot="ur16e.yml",
        optimizer_configs=["ik/lbfgs_ik.yml"],
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        use_cuda_graph=True,
        num_seeds=1,
        seed_solver_num_seeds=1,
        acceleration_regularization_weight=100.0,
        velocity_regularization_weight=1.0,
        seed_velocity_weight=1.0,
        seed_acceleration_weight=1.0,
        optimization_dt=0.1,
        success_requires_convergence=False,
        seed_position_weight=1.0,
        seed_orientation_weight=0.1,
        max_batch_size=1,
    )
    ik_cfg.exit_early = False
    ik_dik = InverseKinematics(ik_cfg)

    current_joints = HOME_JOINTS.copy()
    home_js  = make_js(current_joints)
    kin      = ik_dik.compute_kinematics(home_js)
    from curobo.types import GoalToolPose as _GTP, Pose as _Pose
    ik_dik.solve_pose(
        _GTP.from_poses(kin.tool_poses.to_dict(),
                        ordered_tool_frames=ik_dik.tool_frames, num_goalset=1),
        current_state=make_js(current_joints), return_seeds=1,
    )
    print("Differential IK ready.")

    home_pos_urdf  = kin.tool_poses["tool0"].position.squeeze().cpu().numpy()
    home_quat_urdf = kin.tool_poses["tool0"].quaternion.squeeze().cpu().numpy()
    home_pos_world  = urdf_to_world_pos(home_pos_urdf[None])[0]
    home_quat_world = _qmul(ROBOT_BASE_WXYZ, home_quat_urdf)

    # ── Heatmap colours ───────────────────────────────────────────────────────
    cmap   = plt.cm.RdYlGn
    colors = (cmap(success_rate)[:, :3] * 255).astype(np.uint8)
    pts_world = np.stack([xs, ys, np.full(len(xs), z_world)], axis=1)

    # ── Viser server ──────────────────────────────────────────────────────────
    print(f"Starting Viser at http://localhost:{port} ...")
    server = viser.ViserServer(host="0.0.0.0", port=port)

    @server.on_client_connect
    def on_connect(client: viser.ClientHandle):
        client.camera.position = np.array([2.0, -2.0, 1.5])
        client.camera.look_at  = np.array([0.3,  0.1,  1.0])

    # Robot mesh in world frame
    server.scene.add_frame("/robot_world",
                           position=tuple(ROBOT_BASE_WORLD.tolist()),
                           wxyz=tuple(ROBOT_BASE_WXYZ.tolist()),
                           show_axes=False)
    urdf = yourdfpy.URDF.load(str(URDF_PATH), load_meshes=True,
                               build_scene_graph=True,
                               filename_handler=mesh_filename_handler)
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot_world")
    viser_urdf.update_cfg(dict(zip(JOINT_NAMES, HOME_JOINTS)))

    # EE gizmo (world frame)
    ee_gizmo = server.scene.add_transform_controls(
        "/ee_target", scale=0.12,
        position=tuple(home_pos_world.tolist()),
        wxyz=tuple(home_quat_world.tolist()),
    )

    # Heatmap point cloud
    server.scene.add_point_cloud("/heatmap", points=pts_world,
                                 colors=colors, point_size=0.018)

    # Task blocks
    TASK_COLORS_RGB = {
        "red_block":  (220,  50,  50),
        "blue_block": ( 60, 120, 220),
        "goal":       (255, 160,   0),
    }
    for name, pos in TASK_POSITIONS_WORLD.items():
        server.scene.add_box(f"/scene/{name}", dimensions=BLOCK_SIZE,
                             position=tuple(pos.tolist()),
                             color=TASK_COLORS_RGB[name])
        server.scene.add_label(f"/scene/{name}/label", text=name,
                               position=tuple((pos + np.array([0, 0, 0.06])).tolist()))

    # Table
    server.scene.add_box("/scene/table", dimensions=(1.40, 2.50, 0.07),
                         position=(1.1, 0.0, 0.95), color=(200, 200, 200), opacity=0.5)

    print(f"Viewer ready at http://localhost:{port}")
    print("Drag the gizmo — robot follows via differential IK.")
    print("Green = reachable with task orientations.  Press Ctrl+C to exit.\n")

    while True:
        cur_pos_world  = np.array(ee_gizmo.position, dtype=np.float32)
        cur_quat_world = np.array(ee_gizmo.wxyz,     dtype=np.float32)

        cur_pos_urdf  = world_to_urdf_pos(cur_pos_world[None])[0].astype(np.float32)
        cur_quat_urdf = _qmul(_Q_ROBOT_INV, cur_quat_world).astype(np.float32)

        from curobo.types import Pose as _P
        goal = _P(position=torch.tensor(cur_pos_urdf[None], device="cuda"),
                  quaternion=torch.tensor(cur_quat_urdf[None], device="cuda"))

        result = ik_dik.solve_pose(
            _GTP.from_poses({"tool0": goal},
                            ordered_tool_frames=ik_dik.tool_frames,
                            num_goalset=1),
            current_state=make_js(current_joints),
            return_seeds=1,
        )

        if result.success.any():
            current_joints = result.js_solution.position.squeeze().cpu().numpy()
            viser_urdf.update_cfg(dict(zip(JOINT_NAMES, current_joints)))

        time.sleep(0.005)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UR16e IK reachability heatmap")
    parser.add_argument("--z",     type=float, default=0.90,
                        help="Slice height in world frame (default 0.90 m)")
    parser.add_argument("--x-min", type=float, default=-0.15)
    parser.add_argument("--x-max", type=float, default=0.90)
    parser.add_argument("--y-min", type=float, default=-0.55)
    parser.add_argument("--y-max", type=float, default=0.55)
    parser.add_argument("--res",   type=int,   default=50,
                        help="Grid resolution per axis (default 50 → 50×50=2500 pts)")
    parser.add_argument("--viser", action="store_true",
                        help="Open Viser 3D viewer after saving the plot")
    parser.add_argument("--port",  type=int,   default=8080)
    args = parser.parse_args()

    # Build IK solver
    print("Building IK solver ...")
    config = InverseKinematicsCfg.create(
        robot="ur16e.yml",
        num_seeds=16,
        self_collision_check=True,
        max_batch_size=512,
    )
    ik = InverseKinematics(config)
    ik.exit_early = False
    print("IK solver ready.\n")

    # Build grid
    x_vals = np.linspace(args.x_min, args.x_max, args.res)
    y_vals = np.linspace(args.y_min, args.y_max, args.res)
    xx, yy = np.meshgrid(x_vals, y_vals)   # shape (res, res)
    xs = xx.ravel()
    ys = yy.ravel()

    n_total = len(xs) * len(TEST_ORIENTATIONS_WXYZ)
    print(f"Grid: {args.res}×{args.res} = {len(xs)} positions  "
          f"× {len(TEST_ORIENTATIONS_WXYZ)} orientations = {n_total} IK queries")
    print(f"Slice at z = {args.z:.3f} m (world frame)\n")

    t0 = time.time()
    success_rate = solve_heatmap(ik, xs, ys, args.z)
    elapsed = time.time() - t0

    reachable_pct = (success_rate > 0).mean() * 100
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Reachable (any orientation): {reachable_pct:.1f}%")
    print(f"Mean success rate (reachable only): "
          f"{success_rate[success_rate > 0].mean() * 100:.1f}%\n")

    rate_grid = success_rate.reshape(args.res, args.res)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"heatmap_z{args.z:.2f}.png"
    plot_heatmap(rate_grid, x_vals, y_vals, args.z, args.res, out_path)

    if args.viser:
        launch_viser(xs, ys, args.z, success_rate, port=args.port)


if __name__ == "__main__":
    main()
