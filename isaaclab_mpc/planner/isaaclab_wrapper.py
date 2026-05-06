"""Isaac Lab parallel-environment wrapper for MPPI.

Analogous to genesis_mpc/planner/genesis_wrapper.py.

Manages a vectorised IsaacLab scene (num_envs copies of the robot) and
exposes the minimal interface required by MPPIIsaacLabPlanner:

    wrapper.apply_robot_cmd(u)          # set joint-velocity targets, shape (num_envs, DOF)
    wrapper.step()                      # advance physics one step for all envs
    wrapper.reset_to_state(q, dq)       # broadcast (DOF,) state to all envs
    wrapper.get_ee_pos()                # (num_envs, 3) end-effector world positions
    wrapper.get_ee_quat()               # (num_envs, 4) end-effector world quaternions (w,x,y,z)
    wrapper.get_object_pos(idx)         # (num_envs, 3) object position by index
    wrapper.get_goal()                  # (3,) goal position
    wrapper.set_goal(goal)              # update goal tensor

NOTE: AppLauncher must be created and the simulation app started
BEFORE importing or instantiating this class (IsaacLab requirement).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional

import torch

# IsaacLab imports — only safe after AppLauncher is running
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class IsaacLabConfig:
    """Physics simulation parameters — mirrors GenesisConfig."""
    dt: float = 1.0 / 60.0          # Physics timestep [s]
    device: str = "cuda:0"
    visualize_rollouts: bool = True
    render: bool = False             # Set True for the world sim (viewer)


# ---------------------------------------------------------------------------
# Scene configuration (built at wrapper init from the robot cfg argument)
# ---------------------------------------------------------------------------

def _make_scene_cfg(
    robot_cfg: ArticulationCfg,
    num_envs: int,
    env_spacing: float,
    object_cfgs: Optional[List[RigidObjectCfg]] = None,
    contact_sensor_cfgs: Optional[List[ContactSensorCfg]] = None,
    static_cfgs: Optional[List[AssetBaseCfg]] = None,
):
    """Dynamically create an InteractiveSceneCfg subclass with robot, any
    number of rigid objects, contact sensors, and static scene obstacles.

    Objects get scene keys ``object_{i}`` / prim paths ``{ENV_REGEX_NS}/Object{i}``.
    Contact sensors are added as-is under keys ``contact_sensor_{i}`` (caller
    is responsible for setting the correct prim_path with ``{ENV_REGEX_NS}``).
    Static objects (non-tracked collision geometry) get keys ``static_{i}``
    and prim paths ``{ENV_REGEX_NS}/Static{i}``.
    """
    ns: dict = {
        "__annotations__": {
            "ground": AssetBaseCfg,
            "light":  AssetBaseCfg,
            "robot":  ArticulationCfg,
        },
        "ground": AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        ),
        "light": AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
        ),
        "robot": robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    }

    for i, ocfg in enumerate(object_cfgs or []):
        key = f"object_{i}"
        ns["__annotations__"][key] = RigidObjectCfg
        ns[key] = ocfg.replace(prim_path=f"{{ENV_REGEX_NS}}/Object{i}")

    for i, scfg in enumerate(contact_sensor_cfgs or []):
        key = f"contact_sensor_{i}"
        ns["__annotations__"][key] = ContactSensorCfg
        ns[key] = scfg  # prim_path already set by caller

    for i, stcfg in enumerate(static_cfgs or []):
        key = f"static_{i}"
        ns["__annotations__"][key] = AssetBaseCfg
        ns[key] = stcfg.replace(prim_path=f"{{ENV_REGEX_NS}}/Static{i}")

    _SceneCfg = configclass(type("_SceneCfg", (InteractiveSceneCfg,), ns))
    return _SceneCfg(num_envs=num_envs, env_spacing=env_spacing)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class IsaacLabWrapper:
    """
    Wraps Isaac Lab SimulationContext + InteractiveScene to provide a
    batch-parallel environment interface for MPPI.

    Parameters
    ----------
    cfg : IsaacLabConfig
        Physics / sim settings.
    robot_cfg : ArticulationCfg
        Robot articulation config (e.g. UR16E_CFG from robots/ur16e.py).
    num_envs : int
        Number of parallel environments (= MPPI num_samples).
    ee_link_name : str
        Name of the end-effector body inside the articulation.
    goal : list[float]
        Initial goal position [x, y, z] in world frame.
    env_spacing : float
        Spacing between parallel environment origins [m].
    object_cfgs : list[RigidObjectCfg], optional
        Rigid objects to include in every environment.  Accessible via
        ``self.objects[i]`` and ``get_object_pos(i)``.
    """

    def __init__(
        self,
        cfg: IsaacLabConfig,
        robot_cfg: ArticulationCfg,
        num_envs: int,
        ee_link_name: str = "wrist_3_link",
        goal: List[float] = None,
        env_spacing: float = 1.5,
        object_cfgs: Optional[List[RigidObjectCfg]] = None,
        contact_sensor_cfgs: Optional[List[ContactSensorCfg]] = None,
        static_cfgs: Optional[List[AssetBaseCfg]] = None,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = cfg.device
        self.ee_link_name = ee_link_name
        self.visualize_link_buffer: list = []

        # ------------------------------------------------------------------
        # Simulation context
        # ------------------------------------------------------------------
        sim_cfg = sim_utils.SimulationCfg(dt=cfg.dt, device=cfg.device)
        self.sim_context = SimulationContext(sim_cfg)
        self.sim_dt = cfg.dt

        # ------------------------------------------------------------------
        # Scene (num_envs parallel copies)
        # ------------------------------------------------------------------
        scene_cfg = _make_scene_cfg(
            robot_cfg,
            num_envs=num_envs,
            env_spacing=env_spacing,
            object_cfgs=object_cfgs,
            contact_sensor_cfgs=contact_sensor_cfgs,
            static_cfgs=static_cfgs,
        )
        self.scene = InteractiveScene(scene_cfg)

        # Kick off the simulation
        self.sim_context.reset()

        # ------------------------------------------------------------------
        # Robot handle and book-keeping
        # ------------------------------------------------------------------
        self.robot = self.scene["robot"]
        self.num_dof = self.robot.num_joints

        # Resolve EE link index once
        body_names = list(self.robot.body_names)
        if ee_link_name in body_names:
            self._ee_idx = body_names.index(ee_link_name)
        else:
            self._ee_idx = len(body_names) - 1
            print(
                f"[IsaacLabWrapper] WARNING: EE link '{ee_link_name}' not found. "
                f"Using '{body_names[self._ee_idx]}' instead. "
                f"Available links: {body_names}"
            )

        # ------------------------------------------------------------------
        # Object handles — one entry per configured object
        # ------------------------------------------------------------------
        self.objects: list = [
            self.scene[f"object_{i}"] for i in range(len(object_cfgs or []))
        ]

        # ------------------------------------------------------------------
        # Contact sensor handles — one entry per configured sensor
        # ------------------------------------------------------------------
        self.contact_sensors: list = [
            self.scene[f"contact_sensor_{i}"]
            for i in range(len(contact_sensor_cfgs or []))
        ]

        # ------------------------------------------------------------------
        # Apply init_state and warm up one physics step so that joint_pos
        # buffers reflect InitialStateCfg rather than zeros.
        # sim_context.reset() initialises PhysX but does not write the
        # default joint positions; we must do that explicitly.
        # ------------------------------------------------------------------
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos,
            self.robot.data.default_joint_vel,
        )
        self.scene.write_data_to_sim()
        self.sim_context.step(render=False)
        self.scene.update(cfg.dt)

        # ------------------------------------------------------------------
        # Goal (thread-safe)
        # ------------------------------------------------------------------
        goal_val = goal if goal is not None else [0.5, 0.0, 0.5]
        self._goal = torch.tensor(goal_val, device=self.device, dtype=torch.float32)
        self._goal_lock = threading.Lock()

    # ------------------------------------------------------------------
    # MPPI interface
    # ------------------------------------------------------------------

    def apply_robot_cmd(self, u: torch.Tensor):
        """Apply joint velocity commands to all parallel environments.

        Args:
            u: (num_envs, DOF) tensor of joint velocity targets [rad/s].
        """
        u = u.to(self.device, non_blocking=True)
        self.robot.set_joint_velocity_target(u)

    def apply_effort_cmd(self, u: torch.Tensor):
        """Apply joint torque commands directly (effort/torque control mode).

        Use this for torque-controlled robots (stiffness=0) instead of
        apply_robot_cmd, which sets velocity targets and would scale the
        torque by the actuator damping gain.

        Args:
            u: (num_envs, DOF) tensor of joint torques [Nm].
        """
        u = u.to(self.device, non_blocking=True)
        self.robot.set_joint_effort_target(u)

    def step(self):
        """Advance physics one step for all environments."""
        if self.cfg.visualize_rollouts:
            ee_pos = self.get_ee_pos().detach().clone()  # (num_envs, 3)
            self.visualize_link_buffer.append(ee_pos)

        self.scene.write_data_to_sim()
        self.sim_context.step(render=self.cfg.render)
        self.scene.update(self.sim_dt)

    def _reset_object(self, obj, pos: torch.Tensor, quat: torch.Tensor):
        """Reset one rigid object to the given local-frame pose across all envs."""
        if obj is None:
            return
        pos_w  = pos.to(self.device, dtype=torch.float32).view(1, 3) + self.scene.env_origins
        quat_w = quat.to(self.device, dtype=torch.float32).view(1, 4).expand(self.num_envs, -1).contiguous()
        pose = torch.cat([pos_w, quat_w], dim=-1)
        vel  = torch.zeros(self.num_envs, 6, device=self.device)
        obj.write_root_link_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(vel)

    def reset_to_state(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        object_states: Optional[List[tuple]] = None,
    ):
        """Broadcast a joint state (and optional object states) to all envs.

        Call this at the start of each MPPI planning step to synchronise
        all envs with the current robot state before rollout.

        Args:
            q:             (DOF,) or (1, DOF) joint positions [rad].
            dq:            (DOF,) or (1, DOF) joint velocities [rad/s].
            object_states: list of (pos, quat) tuples, one per object in
                           ``self.objects``.  If shorter than the object
                           list the remaining objects are not reset.
        """
        q  = q.to(self.device,  dtype=torch.float32).view(1, -1).expand(self.num_envs, -1).contiguous()
        dq = dq.to(self.device, dtype=torch.float32).view(1, -1).expand(self.num_envs, -1).contiguous()

        self.robot.write_joint_state_to_sim(q, dq)

        for i, (pos, quat) in enumerate(object_states or []):
            if i < len(self.objects):
                self._reset_object(self.objects[i], pos, quat)

        self.scene.reset()
        self.visualize_link_buffer = []

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_ee_pos(self) -> torch.Tensor:
        """End-effector position in local env frame, shape (num_envs, 3)."""
        pos_w = self.robot.data.body_link_pose_w[:, self._ee_idx, :3]
        return pos_w - self.scene.env_origins

    def get_ee_quat(self) -> torch.Tensor:
        """End-effector orientation (w, x, y, z), shape (num_envs, 4)."""
        return self.robot.data.body_link_pose_w[:, self._ee_idx, 3:7]

    def get_joint_pos(self) -> torch.Tensor:
        """Joint positions, shape (num_envs, DOF)."""
        return self.robot.data.joint_pos

    def get_joint_vel(self) -> torch.Tensor:
        """Joint velocities, shape (num_envs, DOF)."""
        return self.robot.data.joint_vel

    def get_gravity_torques(self) -> torch.Tensor:
        """Generalized gravity compensation torques, shape (num_envs, DOF).

        Uses the PhysX articulation view to read per-joint gravity torques at
        the current configuration.  These can be added to MPPI effort commands
        so the planner optimises residual torques on top of gravity compensation.

        Returns zeros if the PhysX view is unavailable (e.g. CPU pipeline).
        """
        try:
            return self.robot.root_physx_view.get_gravity_compensation_forces().to(self.device)
        except Exception:
            try:
                return self.robot.root_physx_view.get_generalized_gravity_forces().to(self.device)
            except Exception:
                return torch.zeros(self.num_envs, self.num_dof, device=self.device)

    def get_contact_forces(self, sensor_idx: int = 0) -> torch.Tensor:
        """Net contact forces for a sensor in world frame, shape (num_envs, num_bodies, 3).

        Returns zeros if the sensor was not configured.
        """
        if sensor_idx >= len(self.contact_sensors):
            return torch.zeros(self.num_envs, 1, 3, device=self.device)
        return self.contact_sensors[sensor_idx].data.net_forces_w

    def get_object_quat(self, idx: int = 0) -> torch.Tensor:
        """Object orientation (w, x, y, z) in local env frame, shape (num_envs, 4).

        Returns identity quaternion if the object was not configured.
        """
        if idx >= len(self.objects):
            q = torch.zeros(self.num_envs, 4, device=self.device)
            q[:, 0] = 1.0  # w=1 identity
            return q
        return self.objects[idx].data.root_link_quat_w

    def get_object_pos(self, idx: int = 0) -> torch.Tensor:
        """Object position in local env frame, shape (num_envs, 3).

        Args:
            idx: index into ``self.objects`` (default 0).
        Returns zeros if the object was not configured.
        """
        if idx >= len(self.objects):
            return torch.zeros(self.num_envs, 3, device=self.device)
        return self.objects[idx].data.root_link_pos_w - self.scene.env_origins

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def get_goal(self) -> torch.Tensor:
        """Return goal position, shape (3,)."""
        with self._goal_lock:
            return self._goal.clone()

    def set_goal(self, goal: torch.Tensor):
        """Set goal position, shape (3,) or (1, 3)."""
        with self._goal_lock:
            self._goal = goal.to(self.device, dtype=torch.float32).view(3)
