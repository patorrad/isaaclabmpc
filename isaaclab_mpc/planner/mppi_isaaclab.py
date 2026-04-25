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

from typing import Callable, Optional

import torch

from mppi_torch.mppi import MPPIPlanner

from isaaclab_mpc.planner.isaaclab_wrapper import IsaacLabWrapper, IsaacLabConfig
from isaaclab_mpc.utils.transport import torch_to_bytes, bytes_to_torch


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

        self.sim.reset_to_state(q, dq, object_states=object_states if object_states else None)
        return self._command()

    def _command(self) -> bytes:
        return torch_to_bytes(self.mppi.command(self._state_ph))

    def get_robot_state(self) -> bytes:
        """Return the latest dof state received from the bridge."""
        if self._latest_dof_state is None:
            return torch_to_bytes(torch.zeros(self.sim.num_dof * 2))
        return self._latest_dof_state

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

    def test(self, msg: str):
        """Ping/echo for connection testing."""
        print(f"[MPPIIsaacLabPlanner] test: {msg}")
