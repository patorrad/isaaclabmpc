"""Differential IK viewer for the UR16e with reachability workspace overlay.

All geometry is displayed in world frame. The robot URDF is attached to a
parent frame at its world-space pose so it appears right-side up.

Usage:
    conda run -n curobo_analysis python differential_ik_viewer.py
    # open http://localhost:8080
"""

import time
from pathlib import Path

import numpy as np
import torch
import viser
from viser.extras import ViserUrdf
import yourdfpy

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose
from curobo._src.state.state_joint import JointState

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path(__file__).parent / "results"
URDF_PATH     = Path("/home/paolo/Documents/curobo/curobo/content/assets/robot/ur_description/urdf/ur16e.urdf")
MESH_BASE_DIR = URDF_PATH.parent
PORT = 8080

# ── World-frame robot pose ─────────────────────────────────────────────────────
# Robot is mounted inverted: 180° rotation around world X axis.
ROBOT_BASE_POS  = np.array([0.208, 0.0, 2.075])
ROBOT_BASE_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])   # wxyz

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
HOME_JOINTS = np.array([0.549, -2.2557, 1.0872, 0.8265, 1.5802, 0.5275], dtype=np.float32)

# ── Scene objects (world frame) ────────────────────────────────────────────────
BLOCK_SIZE = (0.05, 0.05, 0.05)

TASK_BLOCKS = {
    "red_block":  (np.array([0.3127, 0.1797, 0.90]), (220,  50,  50)),
    "blue_block": (np.array([0.1825, 0.1874, 0.90]), ( 60, 120, 220)),
    "goal":       (np.array([0.40,   0.10,   0.92]), (255, 160,   0)),
}

TABLE_POS  = np.array([1.1, 0.0, 0.95])
TABLE_DIMS = (1.40, 2.50, 0.07)


# ── Coordinate transforms ──────────────────────────────────────────────────────

def world_to_urdf_pos(pos_world: np.ndarray) -> np.ndarray:
    """World frame → URDF base frame (position only)."""
    p = np.atleast_2d(pos_world) - ROBOT_BASE_POS
    return np.stack([p[:, 0], -p[:, 1], -p[:, 2]], axis=1)


def urdf_to_world_pos(pos_urdf: np.ndarray) -> np.ndarray:
    """URDF base frame → world frame (position only)."""
    p = np.atleast_2d(pos_urdf)
    return np.stack([
        p[:, 0] + ROBOT_BASE_POS[0],
        -p[:, 1] + ROBOT_BASE_POS[1],
        -p[:, 2] + ROBOT_BASE_POS[2],
    ], axis=1)


def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (wxyz convention)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# q_robot_inv = conjugate([0,1,0,0]) = [0,-1,0,0]  (same rotation, double cover)
_Q_ROBOT_INV = np.array([0.0, -1.0, 0.0, 0.0])


def world_to_urdf_quat(q_world: np.ndarray) -> np.ndarray:
    return _qmul(_Q_ROBOT_INV, q_world)


def urdf_to_world_quat(q_urdf: np.ndarray) -> np.ndarray:
    return _qmul(ROBOT_BASE_WXYZ, q_urdf)


# ── Helpers ───────────────────────────────────────────────────────────────────

def mesh_filename_handler(fname: str) -> str:
    fname = fname.replace("package://", "")
    return str((MESH_BASE_DIR / fname).resolve())


def make_js(joints_np: np.ndarray) -> JointState:
    """Position-only JointState with shape [1, dof]."""
    return JointState.from_position(
        torch.tensor(joints_np[None], device="cuda", dtype=torch.float32),
        joint_names=JOINT_NAMES,
    )


def main():
    # ── Load saved reachability data ─────────────────────────────────────────
    print("Loading reachability data ...")
    pos_world = np.load(RESULTS_DIR / "positions_world.npy")
    reachable  = np.load(RESULTS_DIR / "reachable.npy")
    r_pos = pos_world[reachable]
    u_pos = pos_world[~reachable]

    # ── IK solver ────────────────────────────────────────────────────────────
    print("Building IK solver ...")
    config = InverseKinematicsCfg.create(
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
    )
    config.exit_early = False
    ik = InverseKinematics(config)

    current_joints = HOME_JOINTS.copy()

    # Warm-up CUDA graph with home pose
    home_js  = make_js(current_joints)
    kin      = ik.compute_kinematics(home_js)
    ik.solve_pose(
        GoalToolPose.from_poses(
            kin.tool_poses.to_dict(),
            ordered_tool_frames=ik.tool_frames,
            num_goalset=1,
        ),
        current_state=make_js(current_joints),
        return_seeds=1,
    )
    print("IK solver ready.")

    # Home EE pose in world frame (for gizmo initial placement)
    home_pos_urdf  = kin.tool_poses["tool0"].position.squeeze().cpu().numpy()
    home_quat_urdf = kin.tool_poses["tool0"].quaternion.squeeze().cpu().numpy()
    home_pos_world  = urdf_to_world_pos(home_pos_urdf[None])[0]
    home_quat_world = urdf_to_world_quat(home_quat_urdf)

    # ── Viser server ──────────────────────────────────────────────────────────
    print(f"Starting Viser at http://localhost:{PORT} ...")
    server = viser.ViserServer(host="0.0.0.0", port=PORT)

    @server.on_client_connect
    def on_connect(client: viser.ClientHandle):
        client.camera.position = np.array([2.0, -2.0, 1.5])
        client.camera.look_at  = np.array([0.3,  0.1,  1.0])

    # ── Robot: parent frame in world space ────────────────────────────────────
    robot_frame = server.scene.add_frame(
        "/robot_world",
        position=tuple(ROBOT_BASE_POS.tolist()),
        wxyz=tuple(ROBOT_BASE_WXYZ.tolist()),
        show_axes=False,
    )

    print("Loading robot mesh ...")
    urdf = yourdfpy.URDF.load(
        str(URDF_PATH),
        load_meshes=True,
        build_scene_graph=True,
        filename_handler=mesh_filename_handler,
    )
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot_world")
    viser_urdf.update_cfg(dict(zip(JOINT_NAMES, HOME_JOINTS)))
    print(f"Robot loaded ({len(viser_urdf._meshes)} meshes).")

    # ── EE target gizmo (world frame) ─────────────────────────────────────────
    ee_gizmo = server.scene.add_transform_controls(
        "/ee_target",
        scale=0.12,
        position=tuple(home_pos_world.tolist()),
        wxyz=tuple(home_quat_world.tolist()),
    )

    # ── Reachability point cloud (world frame, no transform needed) ────────────
    server.scene.add_point_cloud(
        "/workspace/reachable",
        points=r_pos,
        colors=np.tile([30, 200, 80], (len(r_pos), 1)).astype(np.uint8),
        point_size=0.022,
    )
    idx = np.random.choice(len(u_pos), min(len(u_pos), 2000), replace=False)
    server.scene.add_point_cloud(
        "/workspace/unreachable",
        points=u_pos[idx],
        colors=np.tile([190, 40, 40], (len(idx), 1)).astype(np.uint8),
        point_size=0.018,
    )

    # ── Blocks ────────────────────────────────────────────────────────────────
    block_handles = {}
    label_handles = {}
    for name, (pos, color) in TASK_BLOCKS.items():
        block_handles[name] = server.scene.add_box(
            f"/scene/{name}",
            dimensions=BLOCK_SIZE,
            position=tuple(pos.tolist()),
            color=color,
        )
        label_handles[name] = server.scene.add_label(
            f"/scene/{name}/label",
            text=name,
            position=tuple((pos + np.array([0, 0, 0.06])).tolist()),
        )

    # ── Table ─────────────────────────────────────────────────────────────────
    table_handle = server.scene.add_box(
        "/scene/table",
        dimensions=TABLE_DIMS,
        position=tuple(TABLE_POS.tolist()),
        color=(200, 200, 200),
        opacity=0.6,
    )

    # ── GUI ───────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Workspace"):
        cb_reach   = server.gui.add_checkbox("Reachable (green)",  initial_value=True)
        cb_unreach = server.gui.add_checkbox("Unreachable (red)",  initial_value=False)
        cb_scene   = server.gui.add_checkbox("Scene objects",       initial_value=True)

    @cb_reach.on_update
    def _tr(_):
        pts  = r_pos if cb_reach.value else np.zeros((0, 3))
        cols = np.tile([30, 200, 80], (len(pts), 1)).astype(np.uint8)
        server.scene.add_point_cloud("/workspace/reachable",
                                     points=pts, colors=cols, point_size=0.022)

    @cb_unreach.on_update
    def _tu(_):
        idx2 = np.random.choice(len(u_pos), min(len(u_pos), 2000), replace=False)
        pts  = u_pos[idx2] if cb_unreach.value else np.zeros((0, 3))
        cols = np.tile([190, 40, 40], (len(pts), 1)).astype(np.uint8)
        server.scene.add_point_cloud("/workspace/unreachable",
                                     points=pts, colors=cols, point_size=0.018)

    @cb_scene.on_update
    def _ts(_):
        vis = cb_scene.value
        table_handle.visible = vis
        for h in block_handles.values():
            h.visible = vis
        for h in label_handles.values():
            h.visible = vis

    # ── Differential IK loop ──────────────────────────────────────────────────
    print(f"\nViewer ready at http://localhost:{PORT}")
    print("Drag the white gizmo (EE target) — the robot follows with differential IK.")
    print("Green cloud = reachable workspace.  Press Ctrl+C to exit.\n")

    while True:
        cur_pos_world  = np.array(ee_gizmo.position, dtype=np.float32)
        cur_quat_world = np.array(ee_gizmo.wxyz,     dtype=np.float32)

        # Convert world-frame gizmo pose → URDF frame for CuRobo
        cur_pos_urdf  = world_to_urdf_pos(cur_pos_world[None])[0].astype(np.float32)
        cur_quat_urdf = world_to_urdf_quat(cur_quat_world).astype(np.float32)

        goal = Pose(
            position=torch.tensor(cur_pos_urdf[None],   device="cuda"),
            quaternion=torch.tensor(cur_quat_urdf[None], device="cuda"),
        )

        result = ik.solve_pose(
            GoalToolPose.from_poses(
                {"tool0": goal},
                ordered_tool_frames=ik.tool_frames,
                num_goalset=1,
            ),
            current_state=make_js(current_joints),
            return_seeds=1,
        )

        if result.success.any():
            current_joints = result.js_solution.position.squeeze().cpu().numpy()
            viser_urdf.update_cfg(dict(zip(JOINT_NAMES, current_joints)))

        time.sleep(0.005)


if __name__ == "__main__":
    main()
