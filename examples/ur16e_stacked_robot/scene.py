"""Shared scene objects for ur16e_stacked_robot.

Kept in a separate module (no AppLauncher bootstrap) so both planner.py
and world.py can import it safely after SimulationApp is already running.

Block index → puzzle role (matches solution JSON obj_idx).
Hardcoded _BLOCK_SPECS positions are MPPI world-frame (fallback only);
scenario-driven runs convert bin-frame positions via _bin_to_mppi_local().
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

# All puzzle blocks are 5 cm cubes, 0.2 kg
_BLOCK_SIZE = (0.05, 0.05, 0.05)
_BLOCK_MASS = 0.2
_BLOCK_FRICTION = 0.2

def _bin_to_mppi_local(bin_pos: list) -> list:
    """Constant linear transform: bin frame → Isaac Lab world frame (stacked robot rig).

    R = [[0,1,0],[1,0,0],[0,0,1]]  (swap X↔Y)
    t = [0.10, 0.10, 1.225]        (near-robot offset + stacked-rig table surface)

    Table is at Z=1.19; top surface = 1.19 + 0.035 = 1.225.
    All blocks land at MPPI Y ∈ [0.10, bin_size+0.10] — positive-Y side only.
    Matches puzzles/main.py:_bin_to_mppi_local() except for the Z offset.
    """
    x, y, z = bin_pos
    return [y + 0.10, x + 0.10, z + 1.225]


# Bin-frame source positions from ur16e_stand_blocks.yaml (bin_size=0.2).
# _BLOCK_SPECS is derived from these so the fallback stays in sync with the transform.
_BIN_BLOCK_SPECS = [
    ([0.2297, 0.2127, 0.025], (0.9, 0.2, 0.2)),   # 0: target      red
    ([0.2374, 0.0825, 0.025], (0.3, 0.5, 0.9)),   # 1: obstacle_0  blue
    ([0.0608, 0.0712, 0.025], (0.3, 0.9, 0.2)),   # 2: obstacle_1  green
    # ([0.0765, 0.2095, 0.025], (0.9, 0.9, 0.2)),   # 3: obstacle_2  yellow
]
_BLOCK_SPECS = [(_bin_to_mppi_local(pos), color) for pos, color in _BIN_BLOCK_SPECS]

# Obstacle colour cycle used when loading from a scenario file.
_OBSTACLE_COLORS = [
    (0.3, 0.5, 0.9),  # blue
    (0.3, 0.9, 0.2),  # green
    (0.9, 0.9, 0.2),  # yellow
    (0.9, 0.5, 0.2),  # orange
]


def make_static_cfgs(stand_urdf: str) -> list:
    """Build AssetBaseCfg entries for the stand (URDF) and table (box)."""
    # stand_cfg = AssetBaseCfg(
    #     prim_path="PLACEHOLDER",  # replaced by _make_scene_cfg
    #     spawn=sim_utils.UrdfFileCfg(
    #         asset_path=stand_urdf,
    #         fix_base=True,
    #         merge_fixed_joints=True,
    #         self_collision=False,
    #         joint_drive=None,  # single-link URDF — no joints to drive
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.14)),
    # )

    table_cfg = AssetBaseCfg(
        prim_path="PLACEHOLDER",
        spawn=sim_utils.CuboidCfg(
            size=(1.40, 2.50, 0.07),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)), # opacity=0.35),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.2,
                                                            dynamic_friction=0.2),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, 0.0, 1.19)), 
    )

    stand2_cfg = AssetBaseCfg(
        prim_path="PLACEHOLDER",
        spawn=sim_utils.CuboidCfg(
            size=(0.07, 3.0, 5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)), # opacity=0.35),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.2,
                                                            dynamic_friction=0.2),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.3, 0.0, 0.0)),
    )

    return [table_cfg, stand2_cfg]


def make_block_cfgs(positions: list | None = None) -> list:
    """Build RigidObjectCfg entries for the puzzle blocks.

    Parameters
    ----------
    positions : list of [x, y, z] in MPPI world frame, ordered [target, obs_0, ...].
        If None, uses hardcoded _BLOCK_SPECS (backwards compatible).
        Convert from bin frame first via _bin_to_mppi_local() if needed.
    """
    if positions is None:
        specs = _BLOCK_SPECS
    else:
        target_color = (0.9, 0.2, 0.2)
        specs = [
            (pos, target_color if i == 0 else _OBSTACLE_COLORS[(i - 1) % len(_OBSTACLE_COLORS)])
            for i, pos in enumerate(positions)
        ]
    cfgs = []
    for init_pos, color in specs:
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
