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

from robots import STAND_URDF_PATH as _STAND_URDF

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

# Obstacle colour cycle used when loading from a scenario file.
_OBSTACLE_COLORS = [
    (0.3, 0.5, 0.9),  # blue
    (0.3, 0.9, 0.2),  # green
    (0.9, 0.9, 0.2),  # yellow
    (0.9, 0.5, 0.2),  # orange
]


def _bin_to_mppi_local(bin_pos: list) -> list:
    """Convert bin-frame [x, y, z] to MPPI scene-local frame.

    Matches puzzles/main.py:_bin_to_mppi_local() — keep in sync.
    Verified against the four hardcoded positions in _BLOCK_SPECS.
    """
    x, y, z = bin_pos
    return [
        y + 0.10,
        (x - 0.15) + 0.10 * (1.0 if x >= 0.15 else -1.0),
        z + 0.810,
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.160)),
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


def make_block_cfgs(positions: list | None = None) -> list:
    """Build RigidObjectCfg entries for the puzzle blocks.

    Parameters
    ----------
    positions : list of [x, y, z] in MPPI local frame, ordered [target, obs_0, ...].
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
