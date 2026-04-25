"""UR16e robot configuration for Isaac Lab.

Uses the UR16e URDF from the genesismpc assets directory.  Isaac Lab converts
it to USD automatically on first run (cached in /tmp/isaaclab/).

The URDF at UR16E_URDF_PATH references meshes via relative paths under
``../meshes/``, so the file must stay in its original location inside
``ur_description/urdf/``.

Kinematic summary
-----------------
Joints (revolute, DOF = 6):
    shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
    wrist_1_joint,      wrist_2_joint,       wrist_3_joint

End-effector link used for MPPI cost: ``wrist_3_link``
(tool0 / flange are merged into it when merge_fixed_joints=True)
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

# ---------------------------------------------------------------------------
# Path to the UR16e URDF (relative to this file so the package is portable)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(
    _THIS_DIR,
    "ur_description",
    "urdf",
)
UR16E_URDF_PATH = os.path.normpath(os.path.join(_ASSETS_DIR, "ur16e.urdf"))

# ---------------------------------------------------------------------------
# ArticulationCfg
# ---------------------------------------------------------------------------
UR16E_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=UR16E_URDF_PATH,
        # Fix root link to world (mounted robot, same as genesismpc "fixed=True")
        fix_base=True,
        # Merge fixed joints (flange, tool0, base → merged into their parents)
        merge_fixed_joints=True,
        # Keep self-collision off for speed
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
        # Drive type for the USD conversion — we override at runtime via
        # ImplicitActuatorCfg below, but "velocity" hints the converter to
        # set up the correct drive schema.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="velocity",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=100.0,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Home configuration (arm pointing upward, clear of singularities)
        joint_pos={ #-1.8730, -2.2557,  1.0872,  5.4254,  2.0986, -0.5222
            "shoulder_pan_joint": -1.5708,
            "shoulder_lift_joint": -2.2557, #-2.0977, #-1.5708,   # -90°
            "elbow_joint":         1.0872, #1.5708,    # +90°
            "wrist_1_joint":      0.8265, #-1.5708,    # -90°
            "wrist_2_joint":      1.5802, #-1.5708,    # -90°
            "wrist_3_joint":       0.5275, #0.0,
        },
        pos=(0.208, 0.0, 1.867), # (0., -0., 1.2),
        rot=(0, 1, 0, 0)
    ),
    actuators={
        # Pure velocity-tracking actuator (stiffness=0 → torque = damping*(v_cmd - v))
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint"],
            # UR16e continuous-current torques: ~330 Nm (shoulder), ~56 Nm (wrist)
            effort_limit_sim=330.0,
            velocity_limit_sim=3.14,          # ~180 deg/s
            stiffness=0.0,                    # no position tracking
            damping=200.0,                    # velocity-tracking gain
        ),
    },
)
