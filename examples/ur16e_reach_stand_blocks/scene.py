"""Shared scene objects for ur16e_reach_stand_blocks.

Kept in a separate module (no AppLauncher bootstrap) so both planner.py
and world.py can import it safely after SimulationApp is already running.

Block index → puzzle role (matches solution JSON obj_idx):
  0  puzzle_target     red,  init [0.6127,  0.0797, 0.76]
  1  puzzle_obstacle_0 blue, init [0.4825,  0.0874, 0.76]
  2  puzzle_obstacle_1 blue, init [0.4712, -0.0893, 0.76]
  3  puzzle_obstacle_2 blue, init [0.6095, -0.0735, 0.76]
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

_STAND_URDF = "/home/paolo/Documents/genesismpc/assets/stand/stand.urdf"

# All puzzle blocks are 8 cm cubes, 1 kg
_BLOCK_SIZE = (0.05, 0.05, 0.05)
_BLOCK_MASS = 0.2
_BLOCK_FRICTION = 0.2

# (init_pos, diffuse_color) — order matches solution JSON obj_idx
_BLOCK_SPECS = [
    ([0.3127,  0.1797, 0.8], (0.9, 0.2, 0.2)),   # 0: target      red
    ([0.1825,  0.1874, 0.8], (0.3, 0.5, 0.9)),   # 1: obstacle_0  blue
    ([0.1712, -0.1893, 0.8], (0.3, 0.9, 0.2)),   # 2: obstacle_1  green
    ([0.3095, -0.1735, 0.8], (0.9, 0.9, 0.2)),   # 3: obstacle_2  yellow
]


def make_static_cfgs() -> list:
    """Build AssetBaseCfg entries for the stand (URDF) and table (box)."""
    stand_cfg = AssetBaseCfg(
        prim_path="PLACEHOLDER",  # replaced by _make_scene_cfg
        spawn=sim_utils.UrdfFileCfg(
            asset_path=_STAND_URDF,
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
            size=(1.40, 2.50, 0.07),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)), # opacity=0.35),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.2,
                                                            dynamic_friction=0.2),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, 0.0, 0.775)),
    )

    return [stand_cfg, table_cfg]


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
