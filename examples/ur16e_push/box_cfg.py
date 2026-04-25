"""Shared helper: build an IsaacLab RigidObjectCfg for a simple box."""

from typing import List

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


def make_box_cfg(
    size: List[float],
    mass: float,
    init_pos: List[float],
) -> RigidObjectCfg:
    """Return a RigidObjectCfg for a box with given half-extents, mass and initial position.

    Args:
        size:     [sx, sy, sz] full side lengths in metres.
        mass:     mass in kg.
        init_pos: [x, y, z] initial position in local env frame.
    """
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",   # replaced by IsaacLabWrapper
        spawn=sim_utils.CuboidCfg(
            size=(size[0], size[1], size[2]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.5,
                angular_damping=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.4, 0.8),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(init_pos[0], init_pos[1], init_pos[2]),
        ),
    )
