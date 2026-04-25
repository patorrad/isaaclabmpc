"""Shared static scene objects for ur16e_reach_stand.

Kept in a separate module (no AppLauncher bootstrap) so both planner.py
and world.py can import it safely after SimulationApp is already running.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.converters import UrdfConverterCfg

_STAND_URDF = "/home/paolo/Documents/genesismpc/assets/stand/stand.urdf"


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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
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

    return [stand_cfg, table_cfg]
