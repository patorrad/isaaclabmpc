"""UR16e reachability analysis using CuRobo batch IK.

Robot base in world frame: [0.20, 0.0, 2.075]
All target positions are converted to robot-base frame before IK.

Usage:
    conda run -n curobo_analysis python reachability_analysis.py
    conda run -n curobo_analysis python reachability_analysis.py --orientations 4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose

# ── Scene constants ────────────────────────────────────────────────────────────
# Robot base in world frame.  rot=(0,1,0,0) [w,x,y,z] = 180° around world X
# → rotation matrix R_world = diag(1, -1, -1).
# To go from world frame to robot URDF frame:
#   p_urdf = R_world^T @ (p_world - base_world)
#          = diag(1,-1,-1) @ (p_world - base_world)   (R is its own inverse)
# Concretely: x_urdf = p_x - 0.208
#             y_urdf = -(p_y - 0.0)
#             z_urdf = -(p_z - 2.075)  =  2.075 - p_z
# So task positions AT z_world=0.90 are at z_urdf=+1.175 (ABOVE base in URDF)!
ROBOT_BASE_WORLD = np.array([0.208, 0.0, 2.075])   # world-frame base position

# Exact task positions from config.yaml / extraction_robot_sim.json (world frame)
TASK_POSITIONS_WORLD = {
    "red_block":  np.array([0.3127, 0.1797, 0.90]),
    "blue_block": np.array([0.1825, 0.1874, 0.90]),
    "goal":       np.array([0.40,   0.10,   0.92]),
}

# Tool-down orientations to test (w, x, y, z).
# Primary: tool pointing straight down (-Z world ≡ -Z robot base).
# Additional slight tilts (+/-15 deg around X and Y) to capture approach flexibility.
def make_orientations(n: int):
    """Return n candidate tool0 quaternions in robot URDF frame.

    The robot is mounted with rot=(0,1,0,0) [w,x,y,z] in Isaac Lab (180 deg
    around world X, so URDF +Z points world -Z = downward).  Task positions are
    above the robot in URDF frame.  We sweep orientations from the known-good
    home pose FK result and add yaw variants for the push direction.
    """
    from scipy.spatial.transform import Rotation as R
    quats = []
    # FK at home joints gives quaternion wxyz ≈ [0.703, 0.576, 0.002, 0.417]
    # Use this as the base orientation (arm reaching upward/forward in URDF frame)
    home_q_xyzw = np.array([0.576, 0.002, 0.417, 0.703])
    base_rot = R.from_quat(home_q_xyzw)
    quats.append(base_rot.as_quat()[[3, 0, 1, 2]])  # wxyz

    if n >= 2:
        # Yaw sweeps around robot Z (in URDF frame) to allow different push directions
        for angle in [-30, 30, 60, -60, 90, 180]:
            rot = R.from_euler("z", angle, degrees=True) * base_rot
            quats.append(rot.as_quat()[[3, 0, 1, 2]])

    return np.array(quats[:n], dtype=np.float32)


def world_to_base(positions_world: np.ndarray) -> np.ndarray:
    """Convert (N,3) world-frame positions to robot URDF frame.

    In Isaac Lab, the robot has rot=(0,1,0,0) [w,x,y,z] = 180 deg around world X.
    Rotation matrix R = diag(1,-1,-1).  Inverse:
        p_urdf = R^T @ (p_world - base) = diag(1,-1,-1) @ (p_world - base)
    Concretely: x_urdf = p_x - base_x
                y_urdf = -(p_y - base_y)
                z_urdf = -(p_z - base_z)  =  base_z - p_z
    Blocks at z_world=0.90 are at z_urdf = 2.075 - 0.90 = +1.175 (above base in URDF).
    """
    p = positions_world - ROBOT_BASE_WORLD
    return np.stack([p[:, 0], -p[:, 1], -p[:, 2]], axis=1)


def run_ik(ik: InverseKinematics, positions_base: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    """Batch IK over positions x orientations. Returns bool array (N, O)."""
    n_pos = len(positions_base)
    n_ori = len(orientations)
    results = np.zeros((n_pos, n_ori), dtype=bool)

    for oi, quat in enumerate(orientations):
        pos_t = torch.tensor(positions_base, device="cuda", dtype=torch.float32)
        q_t = torch.tensor(
            np.tile(quat, (n_pos, 1)), device="cuda", dtype=torch.float32
        )
        goal_pose = Pose(position=pos_t, quaternion=q_t)
        result = ik.solve_pose(
            GoalToolPose.from_poses({"tool0": goal_pose}, num_goalset=1)
        )
        success = result.success.squeeze().cpu().numpy()
        if success.ndim == 0:
            success = np.array([success])
        results[:, oi] = success

    return results


def spot_check(ik: InverseKinematics, orientations: np.ndarray):
    """Test exact task positions and report per-position, per-orientation results."""
    print("\n" + "=" * 60)
    print("SPOT CHECK — exact task positions")
    print("=" * 60)
    print(f"{'Position':15s}  {'World (x,y,z)':28s}  {'Base (x,y,z)':28s}  Reach%")
    print("-" * 60)
    for name, pos_w in TASK_POSITIONS_WORLD.items():
        pos_b = world_to_base(pos_w[None])[0]
        results = run_ik(ik, pos_b[None], orientations)  # (1, n_ori)
        frac = results[0].mean() * 100
        dist = np.linalg.norm(pos_b)
        print(
            f"{name:15s}  [{pos_w[0]:.3f},{pos_w[1]:.3f},{pos_w[2]:.3f}]"
            f"  [{pos_b[0]:+.3f},{pos_b[1]:+.3f},{pos_b[2]:+.3f}]"
            f"  {frac:.0f}%  (dist_from_base={dist:.3f}m)"
        )
        for oi, ok in enumerate(results[0]):
            print(f"   ori {oi}: {'OK' if ok else 'FAIL'}")


def grid_sweep(ik: InverseKinematics, orientations: np.ndarray, step: float = 0.05):
    """3D grid sweep over task workspace. Returns positions (N,3) and reachability (N,)."""
    # World-frame bounds: task workspace + wider margin to show full reachable envelope
    # z_world=0.60..2.40 corresponds to z_urdf=2.075-0.60=1.475 .. 2.075-2.40=-0.325
    x_w = np.arange(-0.10, 0.80 + step, step)
    y_w = np.arange(-0.50, 0.50 + step, step)
    z_w = np.arange(0.60, 2.40 + step, step)

    xs, ys, zs = np.meshgrid(x_w, y_w, z_w, indexing="ij")
    pos_world = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    pos_base = world_to_base(pos_world)

    n_total = len(pos_base)
    print(f"\n{'='*60}")
    print(f"GRID SWEEP — {n_total} positions ({len(x_w)}x{len(y_w)}x{len(z_w)})")
    print(f"World bounds: x={x_w[0]:.2f}..{x_w[-1]:.2f}, "
          f"y={y_w[0]:.2f}..{y_w[-1]:.2f}, z={z_w[0]:.2f}..{z_w[-1]:.2f}")
    print("=" * 60)

    # Solve in chunks to stay within GPU memory / max_batch_size
    chunk = ik.config.max_batch_size if hasattr(ik.config, "max_batch_size") else 512
    reachable = np.zeros(n_total, dtype=bool)

    t0 = time.time()
    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        batch_pos = pos_base[start:end]
        res = run_ik(ik, batch_pos, orientations)   # (B, n_ori)
        reachable[start:end] = res.any(axis=1)      # reachable by ANY orientation
        if start % (chunk * 10) == 0:
            pct = 100 * end / n_total
            elapsed = time.time() - t0
            eta = elapsed / (end / n_total) - elapsed
            print(f"  {pct:.0f}%  ({end}/{n_total})  ETA {eta:.0f}s")

    print(f"Done in {time.time()-t0:.1f}s")
    print(f"Reachable: {reachable.sum()} / {n_total} ({100*reachable.mean():.1f}%)")
    return pos_world, reachable


def visualize(pos_world: np.ndarray, reachable: np.ndarray, out_dir: Path):
    """Save 3D scatter plot and per-slice XY heatmaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    task_pts = {k: v for k, v in TASK_POSITIONS_WORLD.items()}

    # ── 3D scatter ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    r_pos = pos_world[reachable]
    u_pos = pos_world[~reachable]
    if len(u_pos):
        ax.scatter(u_pos[:, 0], u_pos[:, 1], u_pos[:, 2],
                   c="red", alpha=0.05, s=4, label="unreachable")
    if len(r_pos):
        ax.scatter(r_pos[:, 0], r_pos[:, 1], r_pos[:, 2],
                   c="green", alpha=0.15, s=4, label="reachable")
    # Task positions
    colors = {"red_block": "r", "blue_block": "b", "goal": "orange"}
    for name, pt in task_pts.items():
        ax.scatter(*pt, c=colors[name], s=200, marker="*", zorder=10, label=name)
    # Robot base
    ax.scatter(*ROBOT_BASE_WORLD, c="black", s=300, marker="^", zorder=10, label="robot base")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("UR16e Reachability Map")
    ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    path = out_dir / "reachability_3d.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved 3D plot → {path}")

    # ── XY slices at key Z heights ──────────────────────────────────────────────
    z_unique = np.unique(np.round(pos_world[:, 2], 3))
    target_zs = [0.90, 0.92, 1.00, 1.10]
    slice_zs = [z_unique[np.argmin(np.abs(z_unique - tz))] for tz in target_zs]

    fig, axes = plt.subplots(1, len(slice_zs), figsize=(4 * len(slice_zs), 4))
    for ax, sz in zip(axes, slice_zs):
        mask = np.abs(pos_world[:, 2] - sz) < 1e-3
        sub_pos = pos_world[mask]
        sub_reach = reachable[mask]
        x_u = np.unique(np.round(sub_pos[:, 0], 4))
        y_u = np.unique(np.round(sub_pos[:, 1], 4))
        img = np.zeros((len(y_u), len(x_u)))
        for p, r in zip(sub_pos, sub_reach):
            xi = np.argmin(np.abs(x_u - p[0]))
            yi = np.argmin(np.abs(y_u - p[1]))
            img[yi, xi] = 1.0 if r else 0.0
        ax.imshow(img, origin="lower", aspect="auto",
                  extent=[x_u[0], x_u[-1], y_u[0], y_u[-1]],
                  cmap="RdYlGn", vmin=0, vmax=1)
        # Overlay task points
        for name, pt in task_pts.items():
            if abs(pt[2] - sz) < 0.06:
                ax.plot(pt[0], pt[1], marker="*", markersize=12,
                        color=colors[name], label=name)
        ax.set_title(f"z = {sz:.2f} m")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.legend(fontsize=7)
    plt.suptitle("UR16e XY Reachability Slices", fontsize=12)
    plt.tight_layout()
    path = out_dir / "reachability_slices.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved XY slices → {path}")

    # ── Save raw data ───────────────────────────────────────────────────────────
    np.save(out_dir / "positions_world.npy", pos_world)
    np.save(out_dir / "reachable.npy", reachable)
    print(f"Saved raw arrays → {out_dir}/positions_world.npy, reachable.npy")


def main():
    parser = argparse.ArgumentParser(description="UR16e reachability analysis via CuRobo")
    parser.add_argument("--orientations", type=int, default=3,
                        help="Number of tool-down orientations to test (1/3/5)")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Grid step size in metres (default 0.05)")
    parser.add_argument("--out-dir", type=str,
                        default=str(Path(__file__).parent / "results"),
                        help="Output directory for plots and numpy arrays")
    parser.add_argument("--spot-only", action="store_true",
                        help="Only run spot-check (no 3D grid sweep)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    orientations = make_orientations(args.orientations)
    print(f"Testing {args.orientations} orientation(s):")
    for i, q in enumerate(orientations):
        print(f"  [{i}] wxyz = [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]")

    # ── Build IK solver ─────────────────────────────────────────────────────────
    # max_batch_size must accommodate grid chunks; we use 512 internally
    print("\nLoading UR16e IK solver ...")
    config = InverseKinematicsCfg.create(
        robot="ur16e.yml",
        num_seeds=16,
        self_collision_check=True,
        max_batch_size=512,
    )
    ik = InverseKinematics(config)
    print("IK solver ready.")

    # ── Spot check ─────────────────────────────────────────────────────────────
    spot_check(ik, orientations)

    if args.spot_only:
        return

    # ── Grid sweep ─────────────────────────────────────────────────────────────
    pos_world, reachable = grid_sweep(ik, orientations, step=args.step)

    # ── Visualize ──────────────────────────────────────────────────────────────
    visualize(pos_world, reachable, out_dir)
    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
