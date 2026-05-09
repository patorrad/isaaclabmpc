"""Static scene objects and camera configs for ur16e_push_stick_openpi.

Scene layout (world frame, robot base at (0.208, 0.0, 2.075)):
  - Stand + table
  - Push object: red 5 cm cube on the table
  - Goal marker: green flat pad on table surface (static)
  - Overhead camera: directly above workspace, looking down
  - Side camera: from the right, angled in toward workspace

Camera setup
------------
Isaac Lab Camera with convention="world" uses:
    forward = +X_cam,  up = +Z_cam

look_at_quat(eye, target) computes the (w,x,y,z) quaternion so the camera
at `eye` looks exactly at `target`. Adjust only `eye` and `target` to move
the cameras — no quaternion math needed.
"""

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg

# ── Scene geometry ───────────────────────────────────────────────────────────
_TABLE_SURFACE_Z = 0.72   # table center z=0.65 + half-height 0.07

PUSH_OBJECT_INIT_POS = (0.55, 0.0, 0.80)

# Goal marker: 25 cm in +Y from start position
GOAL_MARKER_POS = (0.55, 0.25, _TABLE_SURFACE_Z + 0.002)

# Workspace centre (midpoint between object start and goal)
_WS_CENTER = (0.55, 0.125, _TABLE_SURFACE_Z + 0.025)


# ── look_at helper ───────────────────────────────────────────────────────────

def look_at_quat(eye, target, world_up=(0., 0., 1.)):
    """Return (w, x, y, z) quaternion for a camera at `eye` looking at `target`.

    Convention: Isaac Lab Camera with convention="world"
        Camera +X = forward (into scene)
        Camera +Y = right
        Camera +Z = up (image top direction)

    `world_up` biases which way is "up" in the rendered image.
    Automatically switches to (1,0,0) when forward is nearly parallel to
    world_up (gimbal-lock guard, e.g. overhead cameras).
    """
    eye    = np.asarray(eye,    float)
    target = np.asarray(target, float)
    up     = np.asarray(world_up, float); up /= np.linalg.norm(up)

    fwd = target - eye; fwd /= np.linalg.norm(fwd)   # camera +X

    if abs(np.dot(fwd, up)) > 0.99:                   # gimbal-lock guard
        up = np.array([1., 0., 0.])

    right   = np.cross(up, fwd);   right   /= np.linalg.norm(right)    # camera +Y
    cam_up  = np.cross(fwd, right); cam_up /= np.linalg.norm(cam_up)   # camera +Z

    # Rotation matrix: columns = [cam+X, cam+Y, cam+Z] expressed in world
    R = np.column_stack([fwd, right, cam_up])

    # Matrix → quaternion (w, x, y, z)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s;  x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s;  z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s;  x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s;                  z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s;  x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s;  z = 0.25 * s

    return (float(w), float(x), float(y), float(z))


# ── Static scene objects ─────────────────────────────────────────────────────

def make_static_cfgs(stand_urdf: str) -> list:
    stand_cfg = AssetBaseCfg(
        prim_path="PLACEHOLDER",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=stand_urdf,
            fix_base=True,
            merge_fixed_joints=True,
            self_collision=False,
            joint_drive=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.14)),
    )

    table_cfg = AssetBaseCfg(
        prim_path="PLACEHOLDER",
        spawn=sim_utils.CuboidCfg(
            size=(1.40, 2.50, 0.14),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.5
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, 0.0, 0.65)),
    )

    goal_marker_cfg = AssetBaseCfg(
        prim_path="PLACEHOLDER",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.004),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.9, 0.2), opacity=0.9
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=GOAL_MARKER_POS),
    )

    return [stand_cfg, table_cfg, goal_marker_cfg]


def make_push_object_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="PLACEHOLDER",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=2.0,
                angular_damping=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.5
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PUSH_OBJECT_INIT_POS),
    )


# ── Camera configs ───────────────────────────────────────────────────────────
# Tune cameras by changing eye/target tuples only — look_at_quat handles rotation.

# Overhead camera: directly above, looking straight down at workspace centre.
_OVERHEAD_EYE    = (0.55, 0.125, 1.80)
_OVERHEAD_TARGET = _WS_CENTER

# Side camera: from the right (+X) and slightly above, angled in.
_SIDE_EYE    = (1.30, 0.125, 1.10)
_SIDE_TARGET = _WS_CENTER


def make_camera_cfgs() -> list:
    overhead_rot = look_at_quat(_OVERHEAD_EYE, _OVERHEAD_TARGET)
    side_rot     = look_at_quat(_SIDE_EYE,     _SIDE_TARGET)

    print(f"[cameras] overhead  eye={_OVERHEAD_EYE}  rot={tuple(round(v,4) for v in overhead_rot)}")
    print(f"[cameras] side      eye={_SIDE_EYE}  rot={tuple(round(v,4) for v in side_rot)}")

    _cam_spawn = sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 20.0),
    )

    overhead_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverheadCam",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=_cam_spawn,
        offset=CameraCfg.OffsetCfg(
            pos=_OVERHEAD_EYE,
            rot=overhead_rot,
            convention="world",
        ),
    )

    side_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/SideCam",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=_cam_spawn,
        offset=CameraCfg.OffsetCfg(
            pos=_SIDE_EYE,
            rot=side_rot,
            convention="world",
        ),
    )

    return [overhead_cam, side_cam]
