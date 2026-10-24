"""Glue between the hidden-object search and MPPI (after mppi_playground's server.plan_request).

SearchGuide keeps the floor belief, updates it from each ToF frame, re-plans the tree when
the current plan is done, stuck or old (not on a fixed cadence: MPPI needs time to carry out
a push), and turns the best path into
  - a waypoint for MPPI's goal (point `lookahead` ahead along the path), and/or
  - a prior: PathTrackingPrior drives `n_samples` rollouts along the path, unperturbed,
    so the path competes with MPPI's own samples on cost (mppi_torch sample_other_priors).

The belief and tree are numpy; PathTrackingPrior runs on the planner's sim (torch).
"""

import math

import numpy as np

from .belief import Box, FloorBelief, yaw_from_quat
from .search_mcts import search


class SearchGuide:
    """p: the `search:` config dict (see examples/ur16e_reach_stand_tote/config.yaml)."""

    def __init__(self, p: dict, interior, floor_z: float, walls: list, object_halves: list, seed: int = 0):
        self.p = p
        self.interior = interior
        self.walls = walls
        self.halves = [np.asarray(h, float) for h in object_halves]
        self.belief = FloorBelief(interior[0], interior[1], floor_z, p.get("res", 0.01))
        zones = p.get("zones", 8)
        self.zone_half_angle = math.radians(p.get("fov_deg", 45.0)) / zones / 2
        self.boxes: list = []
        self.tcp = None
        self.path = None          # (P, 3) best path from the last search
        self.result = None
        self._calls = 0
        self._seed = seed
        self._age = 0
        self._best_progress = 0.0
        self._last_progress_call = 0

    # ── inputs ──────────────────────────────────────────────────────────────
    def set_objects(self, poses):
        """poses: [(pos (3,), quat wxyz (4,)) or None] of the known objects, in config order;
        None = not detected (yet), left out of the search's model."""
        self.boxes = [Box(np.asarray(pq[0], float).copy(), h, yaw_from_quat(pq[1]))
                      for pq, h in zip(poses, self.halves) if pq is not None]

    def observe(self, origins, dirs, dists, tcp) -> int:
        """Latest ToF frame (zone-centre rays, world/env-local frame) and the current TCP."""
        self.tcp = np.asarray(tcp, float)
        return self.belief.observe(origins, dirs, dists, self.zone_half_angle)

    # ── planning ────────────────────────────────────────────────────────────
    def plan(self, force: bool = False):
        """Re-plan only when needed, so MPPI can finish executing a plan (pushes take seconds):
        no plan yet, the TCP reached the path's end, no progress along it for
        `replan_stuck_steps` calls, or the plan is `replan_max_age` calls old.
        Returns the new result when it ran, else None."""
        self._calls += 1
        if self.tcp is None:
            return None
        reason = None
        if force or self.path is None:
            reason = "new"
        else:
            self._age += 1
            prog = self.progress(self.tcp)
            if prog > self._best_progress + 0.01:
                self._best_progress, self._last_progress_call = prog, self._calls
            if np.linalg.norm(self.tcp - self.path[-1]) < self.p.get("replan_reach", 0.02):
                reason = "reached"
            elif self._calls - self._last_progress_call >= int(self.p.get("replan_stuck_steps", 30)):
                reason = "stuck"
            elif self._age >= int(self.p.get("replan_max_age", 150)):
                reason = "age"
        if reason is None:
            return None
        self.result = search(self.tcp, self.belief, self.boxes, self.walls, self.interior, self.p,
                             seed=self._seed + self._calls)
        self.result["reason"] = reason
        self.path = self.result["best_path"] if len(self.result["best_path"]) > 1 else None
        self._age, self._best_progress, self._last_progress_call = 0, 0.0, self._calls
        return self.result

    def progress(self, tcp) -> float:
        """Arc length along the path of the TCP's projection onto it."""
        path = self.path
        seg0, d = path[:-1], np.diff(path, axis=0)
        L = np.linalg.norm(d, axis=1)
        t = np.clip(np.einsum("ij,ij->i", tcp - seg0, d) / np.maximum(L ** 2, 1e-12), 0.0, 1.0)
        i = int(np.argmin(np.linalg.norm(seg0 + t[:, None] * d - tcp, axis=1)))
        return float(np.concatenate([[0.0], np.cumsum(L)])[i] + t[i] * L[i])

    def path_point(self, tcp, lookahead: float):
        """Point `lookahead` metres along the path beyond the TCP's projection onto it."""
        path = self.path
        if path is None:
            return None
        seg0, seg1 = path[:-1], path[1:]
        d = seg1 - seg0
        L = np.linalg.norm(d, axis=1)
        t = np.clip(np.einsum("ij,ij->i", tcp - seg0, d) / np.maximum(L ** 2, 1e-12), 0.0, 1.0)
        proj = seg0 + t[:, None] * d
        i = int(np.argmin(np.linalg.norm(proj - tcp, axis=1)))
        arc = np.concatenate([[0.0], np.cumsum(L)])
        s = arc[i] + t[i] * L[i] + lookahead
        # Corner gate: don't look past a sharp corner the TCP hasn't reached, or MPPI cuts it
        # (e.g. sweeping diagonally onto a box instead of descending beside it first).
        wp = self.p.get("waypoint", {})
        cos_max = math.cos(math.radians(wp.get("corner_deg", 45.0)))
        tol = wp.get("corner_tol", 0.015)
        for v in range(i + 1, len(path) - 1):
            if arc[v] > s:
                break
            a, b = d[v - 1], d[v]
            sharp = L[v - 1] > 1e-9 and L[v] > 1e-9 and (a @ b) / (L[v - 1] * L[v]) < cos_max
            if sharp and np.linalg.norm(tcp - path[v]) > tol:
                s = arc[v]
                break
        j = min(int(np.searchsorted(arc, s)), len(path) - 1)
        if j == 0:
            return path[0].copy()
        frac = np.clip((s - arc[j - 1]) / max(L[j - 1], 1e-12), 0.0, 1.0)
        return path[j - 1] + frac * (path[j] - path[j - 1])

    def mppi_goal(self, tcp=None):
        """Waypoint along the best path (waypoint toggle) or the path's end.

        Without a path (the tree found nothing safe from here, e.g. the tool ended up in contact
        with an object, or outside the tree's workspace) the goal is straight up from the TCP to
        `fallback_z`, so MPPI frees the tool and the next search can start from a clean state.
        """
        if self.path is None:
            if self.tcp is None:
                return None
            z_hi = self.p.get("workspace_hi", [0, 0, 1.3])[2]
            z = min(max(self.tcp[2] + 0.05, self.p.get("fallback_z", 0.95)), z_hi - 0.02)
            return np.array([self.tcp[0], self.tcp[1], z])
        wp = self.p.get("waypoint", {})
        if wp.get("enabled", True):
            return self.path_point(self.tcp if tcp is None else np.asarray(tcp, float), wp.get("lookahead", 0.08))
        return self.path[-1].copy()


class PathTrackingPrior:
    """mppi_torch prior: joint velocities that make envs 0..n-1 follow the guide's best path.

    Pure pursuit in Cartesian space with the tool kept pointing down, mapped to joint
    velocities with damped least squares on the TCP Jacobian. Returns (0, DOF) when there
    is no path or the prior is disabled, which leaves MPPI's samples untouched.
    """

    def __init__(self, guide: SearchGuide, tool_length: float, u_min: float, u_max: float):
        self.guide = guide
        self.tool_length = tool_length
        self.u_min, self.u_max = u_min, u_max
        pr = guide.p.get("prior", {})
        self.enabled = bool(pr.get("enabled", False))
        self.n = int(pr.get("n_samples", 1))
        self.gain = float(pr.get("gain", 4.0))
        self.lookahead = float(pr.get("lookahead", 0.05))
        self.speed = float(guide.p.get("speed", 0.1))
        self.ori_gain = float(pr.get("ori_gain", 2.0))
        self.damping = float(pr.get("damping", 0.05))

    def compute_command(self, sim, t=None):
        import torch
        from isaaclab_mpc.cost.utils import quat_apply

        n = self.n if (self.enabled and self.guide.path is not None) else 0
        if n == 0:
            return torch.zeros(0, sim.num_dof, device=sim.device)
        ee_pos, ee_quat = sim.get_ee_pos()[:n], sim.get_ee_quat()[:n]
        z = torch.zeros(n, 3, device=sim.device); z[:, 2] = 1.0
        axis = quat_apply(ee_quat, z)                            # tool axis (EE +Z) in world
        r = axis * self.tool_length                              # EE origin → TCP
        tcp = ee_pos + r
        targets = torch.tensor(np.stack([self.guide.path_point(p, self.lookahead)
                                         for p in tcp.detach().cpu().numpy()]),
                               dtype=torch.float32, device=sim.device)
        v = self.gain * (targets - tcp)
        speed = v.norm(dim=1, keepdim=True).clamp(min=1e-9)
        v = v * torch.clamp(self.speed / speed, max=1.0)
        down = torch.zeros_like(axis); down[:, 2] = -1.0
        w = self.ori_gain * torch.cross(axis, down, dim=1)       # rotate the tool axis toward -Z
        J = sim.get_ee_jacobian()[:n]                            # (n, 6, DOF), world frame
        Jl, Ja = J[:, :3], J[:, 3:]
        rx = torch.zeros(n, 3, 3, device=sim.device)             # [r]x
        rx[:, 0, 1], rx[:, 0, 2] = -r[:, 2], r[:, 1]
        rx[:, 1, 0], rx[:, 1, 2] = r[:, 2], -r[:, 0]
        rx[:, 2, 0], rx[:, 2, 1] = -r[:, 1], r[:, 0]
        Jt = torch.cat([Jl - rx @ Ja, Ja], dim=1)                # TCP twist Jacobian
        twist = torch.cat([v, w], dim=1).unsqueeze(-1)
        A = Jt @ Jt.transpose(1, 2) + (self.damping ** 2) * torch.eye(6, device=sim.device)
        dq = (Jt.transpose(1, 2) @ torch.linalg.solve(A, twist)).squeeze(-1)
        return dq.clamp(self.u_min, self.u_max)
