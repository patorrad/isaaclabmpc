"""Interactive Viser visualizer for the UR16e reachability analysis.

Displays the saved reachability point cloud in world frame together with
task positions and a live IK slice plane that re-solves on the fly.

Usage:
    conda run -n curobo_analysis python visualize_viser.py
    # then open http://localhost:8080 in your browser
"""

import time
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as vtf

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
ROBOT_BASE_WORLD = np.array([0.208, 0.0, 2.075])
PORT = 8080

TASK_POSITIONS_WORLD = {
    "red_block":  (np.array([0.3127, 0.1797, 0.90]), (0.9, 0.2, 0.2)),
    "blue_block": (np.array([0.1825, 0.1874, 0.90]), (0.3, 0.5, 0.9)),
    "goal":       (np.array([0.40,   0.10,   0.92]), (1.0, 0.6, 0.0)),
}

# ── Coordinate helpers ────────────────────────────────────────────────────────
def world_to_urdf(positions_world: np.ndarray) -> np.ndarray:
    """World → robot URDF frame (robot has 180° rotation around world X)."""
    p = np.atleast_2d(positions_world) - ROBOT_BASE_WORLD
    return np.stack([p[:, 0], -p[:, 1], -p[:, 2]], axis=1)


def urdf_to_world(positions_urdf: np.ndarray) -> np.ndarray:
    """Robot URDF frame → world frame."""
    p = np.atleast_2d(positions_urdf)
    world = np.stack([p[:, 0], -p[:, 1], -p[:, 2]], axis=1)
    return world + ROBOT_BASE_WORLD


# ── IK slice helper ───────────────────────────────────────────────────────────
def solve_slice(ik: InverseKinematics, center_world: np.ndarray,
                normal_world: np.ndarray, extent: float,
                n_per_axis: int = 20) -> tuple:
    """Solve IK for a grid on a plane defined by center+normal (world frame).

    Returns (positions_world, success_bool) arrays of shape (N, 3) and (N,).
    """
    # Build a local XY grid on the plane
    z_ax = normal_world / (np.linalg.norm(normal_world) + 1e-9)
    x_ax = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(x_ax, z_ax)) > 0.9:
        x_ax = np.array([0.0, 1.0, 0.0])
    x_ax = x_ax - np.dot(x_ax, z_ax) * z_ax
    x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)

    ts = np.linspace(-extent / 2, extent / 2, n_per_axis)
    xs, ys = np.meshgrid(ts, ts)
    offsets = xs.ravel()[:, None] * x_ax + ys.ravel()[:, None] * y_ax
    pos_world = center_world + offsets

    pos_urdf = world_to_urdf(pos_world).astype(np.float32)

    # Use the FK-home quaternion as seed orientation (best coverage for this robot)
    home_q = np.array([0.703, 0.576, 0.002, 0.417], dtype=np.float32)
    n = len(pos_urdf)
    pos_t = torch.tensor(pos_urdf, device="cuda")
    q_t = torch.tensor(np.tile(home_q, (n, 1)), device="cuda", dtype=torch.float32)

    result = ik.solve_pose(
        GoalToolPose.from_poses({"tool0": Pose(position=pos_t, quaternion=q_t)},
                                num_goalset=1)
    )
    success = result.success.squeeze().cpu().numpy()
    if success.ndim == 0:
        success = np.array([success])
    return pos_world, success


def main():
    # ── Load saved reachability data ─────────────────────────────────────────
    pos_world = np.load(RESULTS_DIR / "positions_world.npy")
    reachable = np.load(RESULTS_DIR / "reachable.npy")

    r_pos = pos_world[reachable]
    u_pos = pos_world[~reachable]

    # ── Build IK solver for live slice ───────────────────────────────────────
    print("Loading IK solver for live slice ...")
    config = InverseKinematicsCfg.create(
        robot="ur16e.yml",
        num_seeds=16,
        self_collision_check=True,
        max_batch_size=500,
    )
    ik = InverseKinematics(config)
    ik.exit_early = False
    print("IK solver ready.")

    # ── Viser server ─────────────────────────────────────────────────────────
    server = viser.ViserServer(host="0.0.0.0", port=PORT)
    print(f"\nViser running at http://localhost:{PORT}")
    print("Open that URL in your browser.\n")

    # World-frame axes helper (swap Y/Z for Viser's Y-up convention)
    # Viser uses Y-up; our world uses Z-up → rotate by -90° around X for display
    # We work around this by placing everything in Viser's frame directly.

    # ── Static point cloud: reachable (green) ────────────────────────────────
    if len(r_pos):
        server.scene.add_point_cloud(
            "/reachability/reachable",
            points=r_pos,
            colors=np.tile([0, 200, 80], (len(r_pos), 1)).astype(np.uint8),
            point_size=0.025,
        )
    # Unreachable (red, sparser)
    if len(u_pos):
        # Subsample to keep it light
        idx = np.random.choice(len(u_pos), min(len(u_pos), 1500), replace=False)
        server.scene.add_point_cloud(
            "/reachability/unreachable",
            points=u_pos[idx],
            colors=np.tile([200, 40, 40], (len(idx), 1)).astype(np.uint8),
            point_size=0.020,
        )

    # ── Task positions ────────────────────────────────────────────────────────
    for name, (pos, color) in TASK_POSITIONS_WORLD.items():
        c = tuple(int(v * 255) for v in color)
        server.scene.add_icosphere(
            f"/task/{name}",
            radius=0.04,
            position=tuple(pos.tolist()),
            color=c,
        )
        server.scene.add_label(f"/task/{name}/label", text=name,
                               position=tuple((pos + [0, 0, 0.06]).tolist()))

    # ── Robot base marker ─────────────────────────────────────────────────────
    server.scene.add_icosphere(
        "/robot/base",
        radius=0.05,
        position=tuple(ROBOT_BASE_WORLD.tolist()),
        color=(30, 30, 30),
    )
    server.scene.add_label("/robot/base/label", text="robot base",
                           position=tuple((ROBOT_BASE_WORLD + [0, 0, 0.08]).tolist()))

    # ── GUI controls ──────────────────────────────────────────────────────────
    with server.gui.add_folder("Live IK Slice"):
        btn = server.gui.add_button("Solve slice at current gizmo")
        extent_slider = server.gui.add_slider(
            "Extent (m)", min=0.2, max=1.5, step=0.05, initial_value=0.6
        )
        resolution_slider = server.gui.add_slider(
            "Resolution (n per axis)", min=8, max=30, step=1, initial_value=15
        )

    with server.gui.add_folder("Visibility"):
        show_reachable = server.gui.add_checkbox("Show reachable (green)", initial_value=True)
        show_unreachable = server.gui.add_checkbox("Show unreachable (red)", initial_value=True)

    # Gizmo for the slice plane, initialised at block height
    slice_center_world = np.array([0.25, 0.0, 0.90])
    gizmo = server.scene.add_transform_controls(
        "/slice_gizmo",
        scale=0.12,
        position=tuple(slice_center_world.tolist()),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    @show_reachable.on_update
    def _toggle_reachable(_):
        server.scene.add_point_cloud(
            "/reachability/reachable",
            points=r_pos if show_reachable.value else np.zeros((0, 3)),
            colors=np.tile([0, 200, 80], (len(r_pos) if show_reachable.value else 0, 1)).astype(np.uint8),
            point_size=0.025,
        )

    @show_unreachable.on_update
    def _toggle_unreachable(_):
        idx2 = np.random.choice(len(u_pos), min(len(u_pos), 1500), replace=False)
        pts = u_pos[idx2] if show_unreachable.value else np.zeros((0, 3))
        server.scene.add_point_cloud(
            "/reachability/unreachable",
            points=pts,
            colors=np.tile([200, 40, 40], (len(pts), 1)).astype(np.uint8),
            point_size=0.020,
        )

    @btn.on_click
    def _solve_slice(_):
        center = np.array(gizmo.position, dtype=np.float64)
        # Normal from gizmo orientation (Z axis of the gizmo frame)
        wxyz = np.array(gizmo.wxyz, dtype=np.float64)
        rot = vtf.SO3(wxyz_numpy=wxyz)
        normal = rot.as_matrix()[:, 2]

        extent = extent_slider.value
        n = int(resolution_slider.value)

        print(f"Solving {n}x{n} IK slice at world {center}  normal={normal.round(2)} ...")
        t0 = time.time()
        pos_w, success = solve_slice(ik, center, normal, extent, n_per_axis=n)
        print(f"  Done in {time.time()-t0:.1f}s  ({success.sum()}/{len(success)} reachable)")

        reach_pts = pos_w[success]
        unreach_pts = pos_w[~success]

        server.scene.add_point_cloud(
            "/slice/reachable",
            points=reach_pts if len(reach_pts) else np.zeros((0, 3)),
            colors=np.tile([50, 255, 100], (len(reach_pts), 1)).astype(np.uint8),
            point_size=0.030,
        )
        server.scene.add_point_cloud(
            "/slice/unreachable",
            points=unreach_pts if len(unreach_pts) else np.zeros((0, 3)),
            colors=np.tile([255, 60, 60], (len(unreach_pts), 1)).astype(np.uint8),
            point_size=0.025,
        )

    print("Controls:")
    print("  - Drag the gizmo (white axes) to move the slice plane")
    print("  - Click 'Solve slice at current gizmo' to re-run IK on that plane")
    print("  - Toggle green/red point clouds with the checkboxes")
    print("  - Task positions: red=target block, blue=obstacle block, orange=goal")
    print("\nPress Ctrl+C to exit.\n")

    while True:
        time.sleep(0.05)


if __name__ == "__main__":
    main()
