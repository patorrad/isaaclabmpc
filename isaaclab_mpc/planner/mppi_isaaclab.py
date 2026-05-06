"""MPPI planner backed by Isaac Lab parallel physics.

Analogous to genesis_mpc/planner/mppi_genesis.py.

Architecture
------------
- num_envs == num_samples: each parallel Isaac Lab environment IS one MPPI
  trajectory sample.
- dynamics()      : apply one velocity command to all envs, step physics.
- running_cost()  : delegate to the user-supplied Objective.compute_cost().
- compute_action(): reset all envs to current robot state, call mppi.command(),
                    return the optimal first action.
- zerorpc interface mirrors genesismpc so the same world.py/bridge pattern works.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional

import torch

from mppi_torch.mppi import MPPIPlanner

from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabWrapper, IsaacLabConfig
from isaaclab_mpc.utils.transport import torch_to_bytes, bytes_to_torch


def _euler_xyz_to_quat_wxyz(euler_xyz) -> torch.Tensor:
    """Convert intrinsic XYZ Euler angles (radians) to wxyz quaternion."""
    ex, ey, ez = float(euler_xyz[0]), float(euler_xyz[1]), float(euler_xyz[2])
    cx, sx = math.cos(ex / 2), math.sin(ex / 2)
    cy, sy = math.cos(ey / 2), math.sin(ey / 2)
    cz, sz = math.cos(ez / 2), math.sin(ez / 2)
    return torch.tensor([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ], dtype=torch.float32)


class MPPIIsaacLabPlanner:
    """
    MPPI planner that uses Isaac Lab as the dynamics model.

    Parameters
    ----------
    cfg :
        Config object with attributes:
          cfg.mppi        — MPPIConfig (from mppi_torch)
          cfg.isaaclab    — IsaacLabConfig
          cfg.nx          — state dimension (DOF * 2)
          cfg.goal        — initial goal [x, y, z]
          cfg.ee_link_name— end-effector body name in the articulation
    objective :
        Object with:
          .reset()                        — called before each planning step
          .compute_cost(wrapper) -> (num_envs,) tensor
    prior :
        Optional prior policy (currently unused, pass None).
    robot_cfg :
        ArticulationCfg for the robot (e.g. UR16E_CFG).
    """

    def __init__(
        self,
        cfg,
        objective: Callable,
        robot_cfg,
        prior: Optional[Callable] = None,
        object_cfgs: Optional[list] = None,
        contact_sensor_cfgs: Optional[list] = None,
        static_cfgs: Optional[list] = None,
    ):
        self.cfg = cfg
        self.objective = objective
        self._latest_dof_state: Optional[bytes] = None
        self._latest_object_states: Optional[List] = None
        self.num_envs = cfg.mppi.num_samples
        self.device = cfg.mppi.device

        # ------------------------------------------------------------------
        # Isaac Lab parallel simulation
        # ------------------------------------------------------------------
        self.sim = IsaacLabWrapper(
            cfg=IsaacLabConfig(
                dt=cfg.isaaclab.dt,
                device=cfg.mppi.device,
                visualize_rollouts=cfg.isaaclab.visualize_rollouts,
            ),
            robot_cfg=robot_cfg,
            num_envs=cfg.mppi.num_samples,
            ee_link_name=cfg.ee_link_name,
            goal=cfg.goal,
            env_spacing=getattr(cfg.isaaclab, "env_spacing", 1.5),
            object_cfgs=object_cfgs,
            contact_sensor_cfgs=contact_sensor_cfgs,
            static_cfgs=static_cfgs,
        )

        # ------------------------------------------------------------------
        # Prior (optional learned policy)
        # ------------------------------------------------------------------
        if prior is not None:
            self.prior = lambda state, t: prior.compute_command(self.sim)
        else:
            self.prior = None

        # ------------------------------------------------------------------
        # MPPI planner (from mppi_torch)
        # ------------------------------------------------------------------
        self.mppi = MPPIPlanner(
            cfg.mppi,
            cfg.nx,
            dynamics=self._dynamics,
            running_cost=self._running_cost,
            prior=self.prior,
        )

        # Placeholder state tensor (actual state lives in the simulator)
        self._state_ph = torch.zeros(
            (self.num_envs, cfg.nx), device=self.device
        )

        # Warm up: compile CUDA kernels and fill the GPU pipeline so the
        # first real call isn't slower than subsequent ones.
        # print("[MPPIIsaacLabPlanner] warming up CUDA kernels …", flush=True)
        # for _ in range(5):
        #     self.mppi.command(self._state_ph)
        # torch.cuda.synchronize()
        # print("[MPPIIsaacLabPlanner] warm-up done.", flush=True)

    # ------------------------------------------------------------------
    # MPPI callbacks
    # ------------------------------------------------------------------

    def _dynamics(self, _, u, t=None):
        """One physics step for all parallel envs.

        The actual state is in the Isaac Lab simulator, so the state
        argument is ignored (same pattern as genesismpc).
        """
        self.sim.apply_robot_cmd(u)
        self.sim.step()
        return (self._state_ph, u)

    def _running_cost(self, _):
        """Per-env cost at the current simulation state."""
        return self.objective.compute_cost(self.sim)

    # ------------------------------------------------------------------
    # Core planning entry point
    # ------------------------------------------------------------------

    def compute_action(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Run MPPI and return the optimal first action.

        Args:
            q:  (DOF,) or (1, DOF) current joint positions [rad].
            dq: (DOF,) or (1, DOF) current joint velocities [rad/s].

        Returns:
            (1, DOF) action tensor on CPU.
        """
        self.objective.reset()

        q = torch.as_tensor(q, device=self.device, dtype=torch.float32).view(-1)
        dq = torch.as_tensor(dq, device=self.device, dtype=torch.float32).view(-1)

        # Broadcast current state to all MPPI environments
        self.sim.reset_to_state(q, dq)

        # Optimise and return best first action
        actions = self.mppi.command(self._state_ph)
        return actions.detach().cpu()

    # ------------------------------------------------------------------
    # zerorpc-compatible interface (bytes in / bytes out)
    # Mirrors the genesismpc MPPIGenesisPlanner API so existing bridges
    # and world.py visualisers can connect without modification.
    # ------------------------------------------------------------------

    def compute_action_tensor(
        self, dof_state_bytes: bytes, root_state_bytes: bytes
    ) -> bytes:
        """RPC entry point called by the real-robot bridge.

        dof_state_bytes: [q(DOF), dq(DOF), b0_pos(3), b0_quat(4), b1_pos(3),
                          b1_quat(4), ...] packed with torch_to_bytes.
                         Any number of (pos, quat) box-state pairs may follow
                         the joint state.  When absent (len == DOF*2) no object
                         state is reset.
        root_state_bytes: unused, kept for API compatibility.
        """
        self._latest_dof_state = dof_state_bytes
        self.objective.reset()

        dof_state = bytes_to_torch(dof_state_bytes)
        DOF = self.sim.num_dof
        q = dof_state[:DOF].to(self.device, dtype=torch.float32).view(-1)
        dq = dof_state[DOF: DOF * 2].to(self.device, dtype=torch.float32).view(-1)

        # Parse any number of (pos[3], quat[4]) object-state blocks
        object_states = []
        offset = DOF * 2
        while offset + 7 <= dof_state.numel():
            pos  = dof_state[offset:     offset + 3].to(self.device, dtype=torch.float32)
            quat = dof_state[offset + 3: offset + 7].to(self.device, dtype=torch.float32)
            object_states.append((pos, quat))
            offset += 7

        if not object_states and self._latest_object_states:
            object_states = self._latest_object_states
        self.sim.reset_to_state(q, dq, object_states=object_states if object_states else None)
        return self._command()

    def _command(self) -> bytes:
        action = self.mppi.command(self._state_ph)
        # torch.cuda.synchronize()
        return torch_to_bytes(action)

    def get_robot_state(self) -> bytes:
        """Return the latest dof state received from the bridge."""
        if self._latest_dof_state is None:
            return torch_to_bytes(torch.zeros(self.sim.num_dof * 2))
        return self._latest_dof_state

    def set_object_states(self, obj_state_bytes: bytes):
        """Receive object states from the bridge (Genesis format).

        Format: [q_obj0(6), q_obj1(6), ..., dq_obj0(6), dq_obj1(6), ...]
        where each q_obj = [x, y, z, euler_x, euler_y, euler_z].
        """
        data = bytes_to_torch(obj_state_bytes)
        n_objs = data.numel() // 12  # 6 pose + 6 vel per object
        states = []
        for i in range(n_objs):
            q_obj = data[i * 6: i * 6 + 6]
            pos  = q_obj[:3].to(self.device, dtype=torch.float32)
            quat = _euler_xyz_to_quat_wxyz(q_obj[3:6]).to(self.device)
            states.append((pos, quat))
        self._latest_object_states = states if states else None

    def get_object_states(self) -> bytes:
        """Return latest object states as flat [pos(3), quat(4), ...] tensor."""
        if not self._latest_object_states:
            return torch_to_bytes(torch.zeros(0))
        parts = []
        for pos, quat in self._latest_object_states:
            parts.append(pos.cpu())
            parts.append(quat.cpu())
        return torch_to_bytes(torch.cat(parts))

    def get_goal(self) -> bytes:
        return torch_to_bytes(self.sim.get_goal())

    def set_goal(self, goal_bytes: bytes):
        self.sim.set_goal(bytes_to_torch(goal_bytes))

    def get_rollouts(self) -> bytes:
        """Return accumulated EE positions from the last planning step."""
        if not self.sim.cfg.visualize_rollouts or not self.sim.visualize_link_buffer:
            return torch_to_bytes(torch.zeros((1, 1, 3)))
        return torch_to_bytes(torch.stack(self.sim.visualize_link_buffer))

    def get_mppi_horizon(self) -> bytes:
        return torch_to_bytes(torch.tensor(self.cfg.mppi.horizon))

    def get_mppi_num_samples(self) -> bytes:
        return torch_to_bytes(torch.tensor(self.cfg.mppi.num_samples))

    def update_weights(self, weights: dict):
        self.objective.weights = weights

    def get_current_step(self) -> bytes:
        """Return the current sequential push step index (if objective supports it)."""
        return torch_to_bytes(torch.tensor(float(getattr(self.objective, "current_step", 0))))

    def get_total_steps(self) -> bytes:
        """Return the total number of sequential push steps (if objective supports it)."""
        return torch_to_bytes(torch.tensor(float(len(getattr(self.objective, "steps", [])))))

    def get_current_goal_pos(self) -> bytes:
        """Return the goal position for the current push step (block target, local frame)."""
        steps = getattr(self.objective, "steps", [])
        current = getattr(self.objective, "current_step", 0)
        if steps and current < len(steps):
            goal = torch.tensor(steps[current]["end_pos"], dtype=torch.float32)
        else:
            goal = torch.zeros(3)
        return torch_to_bytes(goal)

    def test(self, msg: str):
        """Ping/echo for connection testing."""
        print(f"[MPPIIsaacLabPlanner] test: {msg}")
