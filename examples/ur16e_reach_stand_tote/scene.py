"""Shared static scene objects for ur16e_reach_stand_tote.

Kept in a separate module (no AppLauncher bootstrap) so both planner.py
and world.py can import it safely after SimulationApp is already running.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.converters import UrdfConverterCfg

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from assets.ycb import get_spec, make_ycb_object_cfg, yaw_quat

# All puzzle blocks are 8 cm cubes, 1 kg
_BLOCK_SIZE = (0.05, 0.05, 0.05)
_BLOCK_MASS = 0.2
_BLOCK_FRICTION = 0.2

# (init_pos, diffuse_color) — order matches solution JSON obj_idx
_BLOCK_SPECS = [
    ([0.6127,  0.0797, 0.8], (0.9, 0.2, 0.2)),   # 0: target      red
    ([0.4825,  0.0874, 0.8], (0.3, 0.5, 0.9)),   # 1: obstacle_0  blue
    ([0.4712, -0.0893, 0.8], (0.3, 0.9, 0.2)),   # 2: obstacle_1  green
    ([0.6095, -0.0735, 0.8], (0.9, 0.9, 0.2)),   # 3: obstacle_2  yellow
]

# Table: 0.14 m thick slab centred at z = 0.65.
TABLE_TOP_Z = 0.65 + 0.14 / 2
# Tote (model.usd): origin at its base, 0.62 x 0.42 x 0.28 m; floor top 1.4 cm above the base.
TOTE_POS = (0.4, 0.0, TABLE_TOP_Z)
TOTE_FLOOR_Z = TABLE_TOP_Z + 0.0143
_YCB_DROP = 0.002   # spawn this far above the floor so contacts start clean
# Tote interior (world x/y) and rim height, from the tote's collision boxes (assets/.../yellow_tote).
TOTE_INTERIOR = ((TOTE_POS[0] - 0.238, TOTE_POS[0] + 0.238), (TOTE_POS[1] - 0.17, TOTE_POS[1] + 0.17))
TOTE_RIM_Z = TABLE_TOP_Z + 0.287
_TOTE_WALL_T = 0.03


def tote_wall_boxes() -> list:
    """The four tote walls as upright boxes (search occlusion / collision model)."""
    import numpy as np
    from isaaclab_mpc.search.belief import Box
    (x0, x1), (y0, y1) = TOTE_INTERIOR
    zc, hz, t = (TABLE_TOP_Z + TOTE_RIM_Z) / 2, (TOTE_RIM_Z - TABLE_TOP_Z) / 2, _TOTE_WALL_T
    xm, ym, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 + t, (y1 - y0) / 2 + t
    return [Box(np.array([x0 - t / 2, ym, zc]), np.array([t / 2, hy, hz])),
            Box(np.array([x1 + t / 2, ym, zc]), np.array([t / 2, hy, hz])),
            Box(np.array([xm, y0 - t / 2, zc]), np.array([hx, t / 2, hz])),
            Box(np.array([xm, y1 + t / 2, zc]), np.array([hx, t / 2, hz]))]


def tote_static_boxes() -> list:
    """Known static structure for perception: the tote walls plus its floor slab."""
    import numpy as np
    from isaaclab_mpc.search.belief import Box
    (x0, x1), (y0, y1) = TOTE_INTERIOR
    floor = Box(np.array([(x0 + x1) / 2, (y0 + y1) / 2, (TABLE_TOP_Z + TOTE_FLOOR_Z) / 2]),
                np.array([(x1 - x0) / 2 + _TOTE_WALL_T, (y1 - y0) / 2 + _TOTE_WALL_T, (TOTE_FLOOR_Z - TABLE_TOP_Z) / 2]))
    return tote_wall_boxes() + [floor]


def known_objects(objects: list) -> list:
    """Config `objects` minus the hidden target (the planner must not know about it)."""
    return [o for o in objects if not o.get("target")]


def target_index(objects: list):
    """Index of the `target: true` entry in config `objects`, or None."""
    idx = [i for i, o in enumerate(objects) if o.get("target")]
    if len(idx) > 1:
        raise ValueError("only one object may be the target")
    return idx[0] if idx else None


def object_halves(objects: list) -> list:
    """Bounding-box half extents (rest pose) of each config object."""
    return [tuple(v / 2 for v in get_spec(o["name"]).size) for o in objects]


def ycb_object_poses(objects: list) -> list:
    """[(pos, quat_wxyz)] resting pose for each config entry {name, pos: [x, y(, z)], yaw: deg}.

    Without z, the object rests on the tote floor (bounding-box centre half its height above it).
    """
    poses = []
    for o in objects:
        spec = get_spec(o["name"])
        xy = list(o["pos"])
        z = xy[2] if len(xy) > 2 else TOTE_FLOOR_Z + spec.height / 2 + _YCB_DROP
        poses.append(((float(xy[0]), float(xy[1]), float(z)),
                      yaw_quat(math.radians(float(o.get("yaw", 0.0))))))
    return poses


def make_ycb_cfgs(objects: list, friction: float = 0.5, rollout_model: bool = False) -> list:
    """RigidObjectCfg per YCB entry in config.yaml `objects`, in list order.

    rollout_model: for the planner's copies. Gravity off (poses are reset from the world every
    step, so a known object resting on an unknown one keeps its true height in rollouts instead
    of falling) and high linear damping in place of the floor friction that gravity provided.
    """
    kw = dict(disable_gravity=True, linear_damping=8.0) if rollout_model else {}
    return [make_ycb_object_cfg(o["name"], pos, math.radians(float(o.get("yaw", 0.0))), friction, **kw)
            for o, (pos, _) in zip(objects, ycb_object_poses(objects))]


def make_tof_cfgs(sensors: list, rays_per_zone: int = 2, debug_vis: bool = False,
                  hit_radius: float = 0.02) -> list:
    """VL53L5CX ray casters from config.yaml `tof_sensors` entries {link, pos, rpy_deg}.

    Rays hit the static scene (table, stand, tote) and the YCB objects; the robot is not a target.
    """
    from isaaclab.utils.math import quat_from_euler_xyz
    import torch
    from isaaclab_mpc.sensors.tof import make_tof_cfg

    cfgs = []
    for s in sensors:
        r, p, y = (math.radians(float(a)) for a in s.get("rpy_deg", (0.0, 0.0, 0.0)))
        rot = quat_from_euler_xyz(torch.tensor(r), torch.tensor(p), torch.tensor(y)).tolist()
        cfgs.append(make_tof_cfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{s.get('link', 'wrist_3_link')}",
            static_targets=["{ENV_REGEX_NS}/Static.*"],
            moving_targets=["{ENV_REGEX_NS}/Object.*"],
            pos=tuple(float(v) for v in s.get("pos", (0.0, 0.0, 0.0))),
            rot=tuple(rot),
            rays_per_zone=rays_per_zone,
            debug_vis=debug_vis,
            hit_radius=hit_radius,
        ))
    return cfgs


def make_static_cfgs(stand_urdf: str) -> list:
    """Build AssetBaseCfg entries for the stand (URDF) and table (box)."""
    stand_cfg = AssetBaseCfg(
        prim_path="PLACEHOLDER",  # replaced by _make_scene_cfg
        spawn=sim_utils.UrdfFileCfg(
            asset_path=stand_urdf,
            fix_base=True,
            merge_fixed_joints=True,
            self_collision=False,
            joint_drive=None,  # single-link URDF — no joints to drive
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.14)),
    )

    table_cfg = AssetBaseCfg(
        prim_path="PLACEHOLDER",
        spawn=sim_utils.CuboidCfg(
            size=(1.40, 2.50, 0.14),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.2,
                                                            dynamic_friction=0.2),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, 0.0, 0.65)),
    )

    # The USD's root is a dynamic rigid body; make it kinematic so objects and
    # rollouts can't shove it (it isn't a tracked object, so it is never reset).
    tote_cfg = AssetBaseCfg(
        prim_path="PLACEHOLDER",  # replaced by _make_scene_cfg
        spawn=sim_utils.UsdFileCfg(
            usd_path='/home/paolo/Documents/isaaclabmpc/assets/fixed_assets/yellow_tote/model.usd',
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TOTE_POS),
    )

    return [stand_cfg, table_cfg, tote_cfg]

def make_block_cfgs() -> list:
    """Build RigidObjectCfg entries for the four puzzle blocks.

    Returns a list ordered by obj_idx so it maps directly to the solution JSON.
    """
    cfgs = []
    for init_pos, color in _BLOCK_SPECS:
        cfgs.append(RigidObjectCfg(
            prim_path="PLACEHOLDER",  # replaced by IsaacLabWrapper
            spawn=sim_utils.CuboidCfg(
                size=_BLOCK_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.5,
                    angular_damping=0.5,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=_BLOCK_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=_BLOCK_FRICTION,
                    dynamic_friction=_BLOCK_FRICTION,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(init_pos)),
        ))
    return cfgs
