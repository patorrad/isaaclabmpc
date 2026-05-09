"""UR5e robot configuration for Isaac Lab.

Uses the UR5e URDF already present in ur_description/urdf/ur5e.urdf.
Actuator model: implicit velocity-tracking (stiffness=0, damping=20),
matching the UR16e setup so the same world.py applies_robot_cmd works.

Default joint positions are taken from the centre of the ur5e norm_stats
training distribution (pi05_base/assets/ur5e/norm_stats.json state mean),
rounded to 2 dp: [2.82, -1.63, 0.44, -1.54, -0.39, 0.18].
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
UR5E_URDF_PATH = os.path.normpath(
    os.path.join(_THIS_DIR, "ur_description", "urdf", "ur5e.urdf")
)

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=UR5E_URDF_PATH,
        fix_base=True,
        merge_fixed_joints=True,
        self_collision=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="velocity",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=100.0,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Near the centre of the ur5e training distribution
        joint_pos={
            "shoulder_pan_joint":  2.82,
            "shoulder_lift_joint": -1.63,
            "elbow_joint":          0.44,
            "wrist_1_joint":       -1.54,
            "wrist_2_joint":       -0.39,
            "wrist_3_joint":        0.18,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1, 0, 0, 0),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=3.14,
            # stiffness=0.0,
            # damping=20.0,
            stiffness=100.0,   #0.0                 # no position tracking
            damping=0.0,    #20.0 
        ),
    },
)


def make_ur5e_cfg(pos=(0.0, 0.0, 0.0), rot=(1, 0, 0, 0), joint_pos=None):
    """Return an ArticulationCfg with the robot base and joints at the given state."""
    jp = None
    if joint_pos is not None:
        jp = dict(zip(JOINT_NAMES, joint_pos))

    new_init = UR5E_CFG.init_state.replace(pos=tuple(pos), rot=tuple(rot))
    if jp is not None:
        new_init = new_init.replace(joint_pos=jp)

    return UR5E_CFG.replace(init_state=new_init)
