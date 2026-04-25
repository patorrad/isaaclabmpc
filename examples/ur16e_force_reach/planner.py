"""MPPI planner server for the UR16e force-control reach task.

Like ur16e_reach but the robot is controlled in joint-torque (effort) space
rather than joint-velocity space.  This gives compliant behaviour: the arm
reaches the goal while yielding to unexpected contact forces instead of
rigidly tracking a velocity command.

Key differences from ur16e_reach
---------------------------------
* Robot actuator: stiffness=0, damping=10 Nm/(rad/s) — passive damping only,
  no position tracking.  The robot would fall under gravity without the
  compensation below.
* Gravity compensation: MPPI optimises *residual* torques on top of
  gravity-compensation torques read from the PhysX articulation view.
  The dynamics callback adds gravity comp before every apply_robot_cmd call.
* Contact sensor on wrist_3_link: the objective penalises large contact
  forces so MPPI learns compliant approach trajectories.
* Cost = EE-to-goal distance + contact-force penalty + joint-velocity penalty.

Run with:
    cd /home/paolo/Documents/isaaclabmpc
    CONDA_PREFIX=/home/paolo/miniconda3/envs/env_isaaclab \\
        /home/paolo/miniconda3/envs/env_isaaclab/bin/python \\
        examples/ur16e_force_reach/planner.py
"""

# ===========================================================================
# 1. Simulator bootstrap — must happen first
# ===========================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR16e force-control reach MPPI planner")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# 2. All other imports
# ===========================================================================
import os
import sys

import torch
import yaml
import zerorpc
from dataclasses import dataclass, field
from typing import Dict, List

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg

from mppi_torch.mppi import MPPIConfig
from isaaclab_mpc.planner.mppi_isaaclab import MPPIIsaacLabPlanner
from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabConfig
from robots.ur16e import UR16E_CFG


# ===========================================================================
# 3. Robot config — effort (torque) control mode
# ===========================================================================

# Switch the actuator from velocity-tracking (damping=200) to effort mode:
# stiffness=0 means no position spring; damping=10 provides a passive velocity
# damper that prevents oscillation without fighting the MPPI torques.
# activate_contact_sensors enables PhysX contact reporting on all rigid bodies.
UR16E_EFFORT_CFG = UR16E_CFG.replace(
    spawn=UR16E_CFG.spawn.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        activate_contact_sensors=True,
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit_sim=330.0,
            velocity_limit_sim=3.14,
            stiffness=0.0,   # no position spring — pure effort control
            damping=10.0,    # passive damping to prevent oscillation [Nm/(rad/s)]
        ),
    },
)

# Contact sensor placed at the wrist flange — mirrors the built-in 6-axis F/T
# sensor on the UR16e e-Series.
WRIST_CONTACT_SENSOR = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
    update_period=0.0,
    history_length=0,
    debug_vis=False,
)


# ===========================================================================
# 4. Config
# ===========================================================================

@dataclass
class IsaacLabCfg:
    dt: float = 1.0 / 60.0
    visualize_rollouts: bool = True


@dataclass
class WeightsCfg:
    ee_to_goal:    float = 1.0
    contact_force: float = 0.1
    joint_vel:     float = 0.05


@dataclass
class PlannerConfig:
    n_steps: int = 10000
    nx: int = 12
    goal: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.6])
    ee_link_name: str = "wrist_3_link"
    weights: WeightsCfg = field(default_factory=WeightsCfg)
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    isaaclab: IsaacLabCfg = field(default_factory=IsaacLabCfg)


def _load_config(yaml_path: str) -> PlannerConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cfg = PlannerConfig()
    cfg.n_steps      = raw.get("n_steps",      cfg.n_steps)
    cfg.nx           = raw.get("nx",           cfg.nx)
    cfg.goal         = raw.get("goal",         cfg.goal)
    cfg.ee_link_name = raw.get("ee_link_name", cfg.ee_link_name)

    if "weights" in raw:
        w = raw["weights"]
        cfg.weights = WeightsCfg(
            ee_to_goal=w.get("ee_to_goal",     cfg.weights.ee_to_goal),
            contact_force=w.get("contact_force", cfg.weights.contact_force),
            joint_vel=w.get("joint_vel",       cfg.weights.joint_vel),
        )

    if "mppi" in raw:
        cfg.mppi = MPPIConfig(**{k: v for k, v in raw["mppi"].items()})

    if "isaaclab" in raw:
        il = raw["isaaclab"]
        cfg.isaaclab = IsaacLabCfg(
            dt=il.get("dt", 1.0 / 60.0),
            visualize_rollouts=il.get("visualize_rollouts", True),
        )

    return cfg


# ===========================================================================
# 5. Objective
# ===========================================================================

class Objective:
    """Compliant reach cost.

    Components
    ----------
    ee_to_goal:    L2 distance from EE to goal position.  Primary driver.
    contact_force: Magnitude of net contact force at wrist_3_link.
                   Penalising this makes MPPI prefer trajectories that do
                   not press hard against obstacles — compliance.
    joint_vel:     L2 norm of joint velocities.  Keeps motion smooth and
                   prevents high-speed swings that could damage hardware.
    """

    def __init__(self, weights: WeightsCfg):
        self.weights = {
            "ee_to_goal":    weights.ee_to_goal,
            "contact_force": weights.contact_force,
            "joint_vel":     weights.joint_vel,
        }

    def reset(self):
        pass

    def compute_cost(self, sim) -> torch.Tensor:
        """
        Args:
            sim: IsaacLabWrapper

        Returns:
            (num_envs,) cost tensor — lower is better.
        """
        ee_pos = sim.get_ee_pos()   # (num_envs, 3)
        goal   = sim.get_goal()     # (3,)

        dist = torch.linalg.norm(ee_pos - goal.unsqueeze(0), dim=1)  # (num_envs,)

        # Contact force magnitude at the wrist sensor (sensor index 0)
        forces    = sim.get_contact_forces(0)           # (num_envs, 1, 3)
        force_mag = torch.linalg.norm(forces[:, 0, :], dim=1)  # (num_envs,)

        # Joint velocity norm — penalise jerky / high-speed motion
        dq      = sim.get_joint_vel()                   # (num_envs, DOF)
        vel_pen = torch.linalg.norm(dq, dim=1)          # (num_envs,)

        # Replace NaNs (can appear during heavy contact) with a large penalty
        for t in (dist, force_mag, vel_pen):
            t[torch.isnan(t)] = 100.0

        return (
            self.weights["ee_to_goal"]    * dist
            + self.weights["contact_force"] * force_mag
            + self.weights["joint_vel"]     * vel_pen
        )


# ===========================================================================
# 6. Planner — adds gravity compensation to the MPPI dynamics
# ===========================================================================

class ForceReachPlanner(MPPIIsaacLabPlanner):
    """MPPI planner with effort-space control and gravity compensation.

    Subclasses MPPIIsaacLabPlanner and overrides:

    * _dynamics: adds gravity-compensation torques before every apply_robot_cmd.
    * compute_action_tensor: after planning, returns the optimal EE pose
      [x, y, z, qw, qx, qy, qz] (7 floats) instead of joint torques.
      This is what the real-robot bridge expects when control_mode='force':
      the compliance controller tracks a Cartesian pose target, while MPPI
      determines *which* pose to reach next by optimising in torque space.
    """

    def _dynamics(self, _, u, t=None):
        """One physics step with gravity compensation.

        u: (num_envs, DOF) residual torques from MPPI [Nm].
        The effective torque applied is u + gravity_compensation.
        """
        grav = self.sim.get_gravity_torques()   # (num_envs, DOF)
        self.sim.apply_robot_cmd(u + grav)
        self.sim.step()
        return (self._state_ph, u)

    def compute_action_tensor(
        self, dof_state_bytes: bytes, root_state_bytes: bytes
    ) -> bytes:
        """RPC entry point — runs MPPI and returns the optimal EE pose.

        Internally MPPI still samples and evaluates joint-torque trajectories.
        After optimisation we read the EE pose from env-0 of the sim (which
        was reset to the current robot state) and return it as a 7-element
        tensor [x, y, z, qw, qx, qy, qz] in the local (base-link) frame.

        The real-robot bridge publishes this as a PoseStamped to the
        ur_cartesian_compliance_controller, which then tracks it with the
        configured Cartesian stiffness/damping — giving compliant behaviour.
        """
        from isaaclab_mpc.utils.transport import torch_to_bytes, bytes_to_torch

        self._latest_dof_state = dof_state_bytes
        self.objective.reset()

        dof_state = bytes_to_torch(dof_state_bytes)
        DOF = self.sim.num_dof
        q  = dof_state[:DOF].to(self.device, dtype=torch.float32).view(-1)
        dq = dof_state[DOF:DOF * 2].to(self.device, dtype=torch.float32).view(-1)

        object_states = []
        offset = DOF * 2
        while offset + 7 <= dof_state.numel():
            pos  = dof_state[offset:     offset + 3].to(self.device, dtype=torch.float32)
            quat = dof_state[offset + 3: offset + 7].to(self.device, dtype=torch.float32)
            object_states.append((pos, quat))
            offset += 7

        self.sim.reset_to_state(q, dq, object_states=object_states if object_states else None)

        # Run MPPI optimisation (torque space)
        self.mppi.command(self._state_ph)

        # Read the EE pose from env 0 after the planning horizon
        # (the sim is now in the predicted optimal end-state for env 0)
        ee_pos  = self.sim.get_ee_pos()[0]    # (3,)  local frame
        ee_quat = self.sim.get_ee_quat()[0]   # (4,)  w, x, y, z

        # Pack as [x, y, z, qw, qx, qy, qz]
        ee_pose = torch.cat([ee_pos, ee_quat]).cpu()
        return torch_to_bytes(ee_pose)


# ===========================================================================
# 7. Main
# ===========================================================================

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = _load_config(cfg_path)

    objective = Objective(cfg.weights)
    planner = ForceReachPlanner(
        cfg,
        objective,
        robot_cfg=UR16E_EFFORT_CFG,
        prior=None,
        contact_sensor_cfgs=[WRIST_CONTACT_SENSOR],
    )

    server = zerorpc.Server(planner)
    server.bind("tcp://0.0.0.0:4242")
    print("[planner] Force-reach MPPI server listening on tcp://0.0.0.0:4242")
    server.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
