"""Scene for UR5e pick task with DROID policy.

Layout (robot base at robot_init_pos in config.yaml):
  - White table
  - Red 5 cm cube (pick object)
  - Green flat marker (place target)
  - Overhead camera above workspace
  - Side camera from the right
"""

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg

_TABLE_SURFACE_Z = 0.72   # table center z=0.65 + half-height 0.07

PICK_OBJECT_INIT_POS = (0.45, 0.0,  0.775)   # cube resting on table
PLACE_TARGET_POS     = (0.45, 0.25, _TABLE_SURFACE_Z + 0.002)

_WS_CENTER = (0.45, 0.125, _TABLE_SURFACE_Z + 0.025)


def look_at_quat(eye, target, world_up=(0., 0., 1.)):
    """Return (w,x,y,z) quaternion for Isaac Lab camera convention="world"."""
    eye    = np.asarray(eye,      float)
    target = np.asarray(target,   float)
    up     = np.asarray(world_up, float); up /= np.linalg.norm(up)

    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    if abs(np.dot(fwd, up)) > 0.99:
        up = np.array([1., 0., 0.])

    right  = np.cross(up, fwd);    right  /= np.linalg.norm(right)
    cam_up = np.cross(fwd, right); cam_up /= np.linalg.norm(cam_up)

    R = np.column_stack([fwd, right, cam_up])
    t = R[0,0] + R[1,1] + R[2,2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s; x = (R[2,1]-R[1,2])*s; y = (R[0,2]-R[2,0])*s; z = (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1]-R[1,2])/s; x = 0.25*s; y = (R[0,1]+R[1,0])/s; z = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2]-R[2,0])/s; x = (R[0,1]+R[1,0])/s; y = 0.25*s; z = (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0]-R[0,1])/s; x = (R[0,2]+R[2,0])/s; y = (R[1,2]+R[2,1])/s; z = 0.25*s

    return (float(w), float(x), float(y), float(z))


def make_static_cfgs() -> list:
    table_cfg = AssetBaseCfg(
        prim_path="PLACEHOLDER",
        spawn=sim_utils.CuboidCfg(
            size=(1.20, 2.00, 0.14),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.5
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.65)),
    )

    place_marker_cfg = AssetBaseCfg(
        prim_path="PLACEHOLDER",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.004),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.9, 0.2), opacity=0.9
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=PLACE_TARGET_POS),
    )

    return [table_cfg, place_marker_cfg]


def make_pick_object_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="PLACEHOLDER",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=1.0,
                angular_damping=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.6
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PICK_OBJECT_INIT_POS),
    )


def make_camera_cfgs() -> list:
    _OVERHEAD_EYE    = (0.45, 0.125, 1.60)
    _SIDE_EYE        = (1.10, 0.125, 1.00)

    overhead_rot = look_at_quat(_OVERHEAD_EYE, _WS_CENTER)
    side_rot     = look_at_quat(_SIDE_EYE,     _WS_CENTER)

    print(f"[cameras] overhead eye={_OVERHEAD_EYE} rot={tuple(round(v,4) for v in overhead_rot)}")
    print(f"[cameras] side     eye={_SIDE_EYE} rot={tuple(round(v,4) for v in side_rot)}")

    _cam_spawn = sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 20.0),
    )

    return [
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/OverheadCam",
            update_period=0.0, height=224, width=224,
            data_types=["rgb"], spawn=_cam_spawn,
            offset=CameraCfg.OffsetCfg(pos=_OVERHEAD_EYE, rot=overhead_rot, convention="world"),
        ),
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/SideCam",
            update_period=0.0, height=224, width=224,
            data_types=["rgb"], spawn=_cam_spawn,
            offset=CameraCfg.OffsetCfg(pos=_SIDE_EYE, rot=side_rot, convention="world"),
        ),
    ]
