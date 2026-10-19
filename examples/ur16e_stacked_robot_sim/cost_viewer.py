"""Interactive 3-D viewer of the MPPI cost function over the ur16e_stacked_robot_sim scene.

Evaluates the *real* ``Objective.compute_cost`` from planner.py on a grid of TCP
positions (tool pointing straight down) and draws the result as a heightmap
and/or a flat coloured plane over the table, bin and blocks.  Individual cost
terms can be toggled and re-weighted live.

No simulator is launched: Isaac Lab is replaced by import stubs
(isaaclab_mpc/utils/isaac_stubs.py) and a duck-typed ``GridSim`` stands in for
IsaacLabWrapper, so planner.py / scene.py are used unmodified.

Run with (viser lives in genesistest2; env_isaaclab pins websockets==12):
    cd /home/paolo/Documents/isaaclabmpc
    conda activate genesistest2
    python examples/ur16e_stacked_robot_sim/cost_viewer.py [--scenario X.yaml] [--solution_path Y.json]
    # then open http://localhost:8080

    python examples/ur16e_stacked_robot_sim/cost_viewer.py --selftest   # headless consistency check
"""

import argparse
import copy
import json
import os
import sys
import threading
import time

import numpy as np
import torch
import yaml

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab_mpc.utils.isaac_stubs import install_isaac_stubs

install_isaac_stubs()

parser = argparse.ArgumentParser(description="MPPI cost-function surface viewer")
parser.add_argument("--scenario", type=str, default=None,
                    help="Puzzle scenario YAML (same as planner.py). Sets block positions and bin walls.")
parser.add_argument("--solution_path", type=str, default=None,
                    help="Solution JSON (same as planner.py). Overrides cfg.solution_path.")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--res", type=float, default=0.005, help="Initial grid resolution (m).")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--selftest", action="store_true",
                    help="Check the per-term decomposition against compute_cost and exit.")
args, _ = parser.parse_known_args()

# planner.py parses sys.argv itself (parse_known_args) — --scenario/--solution_path mean the same there.
from examples.ur16e_stacked_robot_sim import planner as P
from examples.ur16e_stacked_robot_sim.scene import (
    make_static_cfgs, make_block_cfgs, _bin_to_mppi_local, _BLOCK_SPECS, _BLOCK_SIZE,
)
from assets.robots.ur16e import get_tool_length, UR16E_URDF_PATH
from isaaclab_mpc.cost.utils import quat_apply

_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
TERMS = list(P.CostWeights.__dataclass_fields__)
# Terms that depend only on robot state (orientation, joint vel, contact forces, Jacobian),
# which GridSim holds fixed — they are constant over the grid.
CONST_TERMS = {"robot_ori", "collision", "joint_vel", "singularity", "bin_collision"}
# Viewer-only geometric stand-in for bin_collision (not part of the planner's cost).
BIN_PROXY = "bin_clearance*"
ALL_TERMS = TERMS + [BIN_PROXY]
# Why a term can be flat over an XY slice at fixed TCP z.
FLAT_REASON = {"obj_to_goal": "depends on the block only → Sweep: Active block / TCP + block",
               "intruder_penalty": "depends on the other blocks only",
               "height_match": "depends on TCP z vs block z → use an XZ / YZ slice",
               "tcp_floor": "depends on TCP z only → use an XZ / YZ slice"}
_TOOL_DOWN_WXYZ = (0.0, 1.0, 0.0, 0.0)   # 180° about X: EE +Z (tool axis) → world −Z


# ===========================================================================
# Fake simulator: one "env" per grid point
# ===========================================================================

class GridSim:
    """Duck-types the IsaacLabWrapper accessors used by Objective.compute_cost."""

    def __init__(self, tcp: torch.Tensor, block_pos: torch.Tensor, tool_length: float):
        self.device = tcp.device
        self.num_envs = n = tcp.shape[0]
        self._quat = torch.tensor(_TOOL_DOWN_WXYZ, device=self.device).expand(n, 4)
        tool = torch.tensor([0.0, 0.0, tool_length], device=self.device).expand(n, 3)
        # Inverse of the planner's TCP offset so compute_cost recovers `tcp` exactly.
        self._ee_pos = tcp - quat_apply(self._quat, tool)
        # (n_blocks, 3) shared by all envs, or (N, n_blocks, 3) per env (block sweep).
        self._blocks = block_pos.to(self.device)
        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(n, 4)
        self.goal = None

    def get_ee_pos(self):
        return self._ee_pos

    def get_ee_quat(self):
        return self._quat

    def get_object_pos(self, idx: int = 0):
        if self._blocks.dim() == 3:
            return self._blocks[:, idx]
        return self._blocks[idx].expand(self.num_envs, 3)

    def get_object_quat(self, idx: int = 0):
        return self._identity_quat

    def get_joint_vel(self):
        return torch.zeros(self.num_envs, 6, device=self.device)

    def get_contact_forces(self, sensor_idx: int = 0):
        return torch.zeros(self.num_envs, 1, 3, device=self.device)

    def get_contact_pair_forces(self, sensor_idx: int = 0):
        return torch.zeros(self.num_envs, 1, 1, 3, device=self.device)

    def get_ee_jacobian(self):
        return torch.eye(6, device=self.device).expand(self.num_envs, 6, 6)

    def set_goal(self, goal: torch.Tensor):
        self.goal = goal.detach().cpu().numpy()


# ===========================================================================
# Geometric bin-collision stand-in
# ===========================================================================

def tool_capsules(urdf_path: str, tool_length: float):
    """tool0 collision primitives as vertical capsules above the TCP (tool pointing down).

    Returns [(dz_lo, dz_hi, radius)] with dz measured upward from the TCP tip.
    Cylinders are approximated by capsules; spheres are zero-length capsules.
    """
    import xml.etree.ElementTree as ET
    caps = []
    for link in ET.parse(urdf_path).getroot().iter("link"):
        if link.get("name") != "tool0":
            continue
        for col in link.findall("collision"):
            origin = col.find("origin")
            zc = float(origin.get("xyz", "0 0 0").split()[2]) if origin is not None else 0.0
            d = tool_length - zc                     # distance of the primitive centre above the tip
            cyl, sph = col.find("geometry/cylinder"), col.find("geometry/sphere")
            if cyl is not None:
                h = float(cyl.get("length")) / 2
                caps.append((d - h, d + h, float(cyl.get("radius"))))
            elif sph is not None:
                caps.append((d, d, float(sph.get("radius"))))
    return caps


def box_sdf(p: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Signed distance from points p (..., 3) to an axis-aligned box (negative inside)."""
    q = np.abs(p - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside


def bin_penetration(tcp: np.ndarray, walls, capsules, margin: float = 0.0,
                    sample_step: float = 0.005) -> np.ndarray:
    """Sum over walls of the deepest tool penetration (m) into that wall.  tcp: (N, 3) → (N,)."""
    out = np.zeros(len(tcp))
    for center, half in walls:
        deepest = np.zeros(len(tcp))
        for lo, hi, r in capsules:
            dz = np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / sample_step)) + 1))
            pts = tcp[:, None, :] + np.stack([0 * dz, 0 * dz, dz], axis=1)[None]   # (N, S, 3)
            pen = (r + margin) - box_sdf(pts, center, half)                       # (N, S)
            deepest = np.maximum(deepest, np.clip(pen.max(axis=1), 0.0, None))
        out += deepest
    return out


# ===========================================================================
# Per-term evaluation through the real Objective
# ===========================================================================

class CostField:
    """Evaluates every cost term separately by calling compute_cost with one-hot weights."""

    def __init__(self, cfg, table_surface_z: float, n_blocks: int, device: str):
        self.device = device
        self.tool_length = get_tool_length()
        cfg_all = copy.deepcopy(cfg)
        for k in TERMS:                       # all terms active → all get their projections
            setattr(cfg_all.costs.weights, k, 1.0)
        self.obj = P.Objective(cfg_all, table_surface_z=table_surface_z, n_blocks=n_blocks)
        # above_target is only computed when obj_avoidance is inactive (elif branch).
        cfg_at = copy.deepcopy(cfg_all)
        cfg_at.costs.weights.obj_avoidance = 0.0
        self.obj_at = P.Objective(cfg_at, table_surface_z=table_surface_z, n_blocks=n_blocks)
        self.steps = self.obj.steps

    def resolve_target_frame(self, target_pos):
        for o in (self.obj, self.obj_at):
            o.resolve_target_frame(torch.as_tensor(target_pos, dtype=torch.float32))

    @torch.no_grad()
    def evaluate(self, tcp: np.ndarray, block_pos: np.ndarray, step: int, push_mode: bool):
        """Returns ({term: (N,) unweighted-but-projected cost}, goal_pos).

        block_pos: (n_blocks, 3), or (N, n_blocks, 3) for per-point block positions.
        """
        sim = GridSim(torch.as_tensor(tcp, dtype=torch.float32, device=self.device),
                      torch.as_tensor(block_pos, dtype=torch.float32), self.tool_length)
        for o in (self.obj, self.obj_at):
            o.current_step = step
            o._first_call = False             # don't derive the mode from env 0 …
            o._is_push_mode = push_mode       # … use the one chosen in the GUI
        out = {}
        for k in TERMS:
            o = self.obj_at if k == "above_target" else self.obj
            o.weights = {t: (1.0 if t == k else 0.0) for t in TERMS}
            out[k] = o.compute_cost(sim).float().cpu().numpy()
        return out, sim.goal


# ===========================================================================
# Scene setup helpers (mirror planner.main())
# ===========================================================================

def load_initial_blocks(solution_path: str, scenario_path: str | None):
    """Block positions (MPPI frame) and bin size.  Priority: scenario → solution → _BLOCK_SPECS."""
    if scenario_path is not None:
        with open(scenario_path) as f:
            sc = yaml.safe_load(f)
        is_ = sc["initial_state"]
        bin_positions = [is_["target_pos"]] + [o["pos"] for o in is_["obstacles"]]
        return [_bin_to_mppi_local(p) for p in bin_positions], sc.get("bin_size"), "scenario"
    with open(solution_path) as f:
        sol = json.load(f)
    is_ = sol.get("initial_state")
    if is_ and sol.get("coordinate_frame", "robot") == "bin":
        bin_positions = [is_["target_pos"]] + list(is_["obstacle_pos"])
        return [_bin_to_mppi_local(p) for p in bin_positions], None, "solution initial_state"
    return [pos for pos, _ in _BLOCK_SPECS], None, "scene._BLOCK_SPECS"


def settle_blocks(pos: np.ndarray, table_z: float, half: float) -> np.ndarray:
    """Drop blocks straight down onto the table / blocks below (as the sim does after spawn)."""
    out = pos.copy()
    order = np.argsort(pos[:, 2])
    settled = []
    for i in order:
        support = table_z
        for j in settled:
            if abs(out[i, 0] - out[j, 0]) < 2 * half and abs(out[i, 1] - out[j, 1]) < 2 * half:
                support = max(support, out[j, 2] + half)
        out[i, 2] = support + half
        settled.append(i)
    return out


def grid_mesh(nx: int, ny: int):
    """Triangle indices for a (ny, nx) vertex grid, row-major."""
    r, c = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    a = (r * nx + c).ravel()
    b, d, e = a + 1, a + nx, a + nx + 1
    return np.concatenate([np.stack([a, b, e], 1), np.stack([a, e, d], 1)])


# ===========================================================================
# Self-test
# ===========================================================================

def selftest(cfg, field: CostField, table_z, n_blocks, blocks0):
    ref = P.Objective(cfg, table_surface_z=table_z, n_blocks=n_blocks)
    if getattr(ref, "_target_frame_unresolved", False):
        ref.resolve_target_frame(torch.as_tensor(blocks0[0], dtype=torch.float32))
    w_cfg = {k: getattr(cfg.costs.weights, k) for k in TERMS}
    rng = np.random.default_rng(0)
    tcp = np.column_stack([rng.uniform(0.3, 0.8, 500), rng.uniform(-0.1, 0.5, 500),
                           rng.uniform(table_z, table_z + 0.2, 500)])
    blocks = settle_blocks(np.asarray(blocks0, dtype=float), table_z, _BLOCK_SIZE[0] / 2)
    ok = True
    for step in range(len(field.steps)):
        for push in (True, False):
            terms, _ = field.evaluate(tcp, blocks, step, push)
            mine = sum(w_cfg[k] * terms[k] for k in TERMS)
            sim = GridSim(torch.as_tensor(tcp, dtype=torch.float32, device=field.device),
                          torch.as_tensor(blocks, dtype=torch.float32), field.tool_length)
            ref.current_step, ref._first_call, ref._is_push_mode = step, False, push
            with torch.no_grad():
                theirs = ref.compute_cost(sim).float().cpu().numpy()
            err = np.max(np.abs(mine - theirs) / np.maximum(1.0, np.abs(theirs)))
            ok &= err < 1e-5
            print(f"  step {step} {'push    ' if push else 'approach'}  max rel err {err:.2e}  "
                  f"cost range [{theirs.min():.3f}, {theirs.max():.3f}]")
    sim = GridSim(torch.as_tensor(tcp, dtype=torch.float32), torch.as_tensor(blocks), field.tool_length)
    tcp_back = sim.get_ee_pos() + quat_apply(sim.get_ee_quat(), torch.tensor([0, 0, field.tool_length]))
    tcp_err = (tcp_back - torch.as_tensor(tcp, dtype=torch.float32)).abs().max().item()
    ok &= tcp_err < 1e-5
    print(f"  TCP round-trip max err {tcp_err:.2e}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ===========================================================================
# Viewer
# ===========================================================================

class CostViewer:
    def __init__(self, cfg, field: CostField, static_cfgs, blocks0, block_colors, table_z, port):
        import viser
        self.cfg, self.field, self.table_z = cfg, field, table_z
        self.half = _BLOCK_SIZE[0] / 2
        self.blocks0 = np.asarray(blocks0, dtype=float)
        self.block_overrides: dict[int, np.ndarray] = {}
        self.terms: dict[str, np.ndarray] = {}
        self.goal = None
        self._lock = threading.Lock()
        self._need_eval = True
        self._need_draw = False
        self._need_vol_draw = False
        self._vol_key = None
        self.vol_terms: dict[str, np.ndarray] = {}

        self.server = s = viser.ViserServer(port=port)
        s.scene.set_up_direction("+z")
        self._init_camera()
        self.static_cfgs = static_cfgs
        # Bin walls (static_cfgs[2:], only with a scenario bin_size) for the geometric stand-in.
        self.walls = [(np.array(c.init_state.pos, dtype=float), np.array(c.spawn.size, dtype=float) / 2)
                      for c in static_cfgs[2:]]
        self.capsules = tool_capsules(UR16E_URDF_PATH, field.tool_length)
        self._add_static_scene(static_cfgs)
        self._build_gui()
        self.block_handles = [
            s.scene.add_box(f"/blocks/{i}", color=block_colors[i], dimensions=_BLOCK_SIZE,
                            position=tuple(self.blocks0[i]))
            for i in range(len(self.blocks0))
        ]
        self.block_ctrls = {}
        self._ctrl_step = None
        self.probe = s.scene.add_transform_controls(
            "/probe", scale=0.08, disable_rotations=True, depth_test=False,
            position=tuple(self.blocks0[0] + np.array([0.1, 0.0, 0.0])))
        s.scene.add_icosphere("/probe/ball", radius=0.006, color=(255, 255, 255))
        @self.probe.on_update
        def _(_):
            if self.sweep.value != "TCP":
                self._need_eval = True
            else:
                self._update_probe()
        self._snap_z()
        self._toggle_block_ctrls()

    # ── scene ───────────────────────────────────────────────────────────────
    def _init_camera(self):
        v = {}
        try:
            with open(_CFG_PATH) as f:
                v = yaml.safe_load(f).get("viewer", {})
        except Exception:
            pass
        lookat = np.array(v.get("lookat", [0.45, 0.2, 0.04])) + [0, 0, self.table_z]
        eye = np.array(v.get("eye", [1.5, 0.0, 0.6])) + [0, 0, self.table_z]
        self.server.initial_camera.position = tuple(eye)
        self.server.initial_camera.look_at = tuple(lookat)

    def _add_static_scene(self, static_cfgs):
        for i, c in enumerate(static_cfgs):
            vm = c.spawn.visual_material
            opacity = vars(vm).get("opacity")        # stubs return a stub for unset attrs
            if i == 1:                               # the 5 m tall stand wall
                opacity = 0.15
            self.server.scene.add_box(f"/static/{i}", color=vm.diffuse_color,
                                      dimensions=tuple(c.spawn.size),
                                      position=tuple(c.init_state.pos), opacity=opacity)
        self.server.scene.add_frame("/robot_base", axes_length=0.1, axes_radius=0.004,
                                    position=tuple(self.cfg.robot_init_pos))

    # ── GUI ─────────────────────────────────────────────────────────────────
    def _build_gui(self):
        g = self.server.gui
        w_cfg = {k: getattr(self.cfg.costs.weights, k) for k in TERMS}
        w_cfg[BIN_PROXY] = 1000.0            # metres → mm of penetration
        self.readout = g.add_markdown("…")
        self.colorbar = g.add_image(np.zeros((12, 256, 3), np.uint8), label="colormap")

        with g.add_folder("Plan context"):
            labels = [f"{i}: {s['obj_name']} (idx {s['obj_idx']})" for i, s in enumerate(self.field.steps)]
            self.step_dd = g.add_dropdown("Step", labels or ["(no steps)"])
            self.mode_dd = g.add_dropdown("Mode", ["push", "approach"])
            self.apply_prev = g.add_checkbox("Apply earlier steps", True,
                                             hint="Move blocks to end_pos of earlier steps / start_pos of this one")
            self.settle = g.add_checkbox("Settle blocks", True,
                                         hint="Drop blocks onto the table as the sim does after spawn")
            self.drag_dd = g.add_dropdown("Drag blocks", ["active block", "all", "off"],
                                          hint="Gizmos are drawn on top of the surface")
            reset_blocks = g.add_button("Reset dragged blocks")

        with g.add_folder("Cost terms"):
            self.enabled = {}
            for k in TERMS:
                on = w_cfg[k] != 0.0 and k not in CONST_TERMS
                self.enabled[k] = g.add_checkbox(k + (" (const)" if k in CONST_TERMS else ""), on)
            self.enabled[BIN_PROXY] = g.add_checkbox(
                BIN_PROXY + " (viewer)", bool(self.walls),
                hint="Geometric stand-in for bin_collision, NOT the planner's cost: tool0 collision "
                     "shapes (from the URDF) as capsules; value = deepest penetration per wall (m), "
                     "summed over walls. Needs --scenario with bin_size.")
            self.proxy_margin = g.add_slider("bin* margin (m)", 0.0, 0.05, 0.001, 0.0,
                                             hint="Inflate the tool radius to show a safety margin")
            all_on, all_off = g.add_button("All on"), g.add_button("All off")
            self.flat_md = g.add_markdown("")

        with g.add_folder("Weights", expand_by_default=False):
            self.weights = {k: g.add_number(k, w_cfg[k], min=0.0, step=0.5) for k in ALL_TERMS}
            reset_w = g.add_button("Reset to config")

        tx0, tx1 = 0.65 - 0.7, 0.65 + 0.7
        xr, yr = self._default_ranges()
        z0, z1 = round(self.table_z - 0.02, 3), round(self.table_z + 0.3, 3)
        with g.add_folder("Grid"):
            self.sweep = g.add_dropdown(
                "Sweep", ["TCP", "Active block", "TCP + block (pushing)"],
                hint="TCP: grid = TCP positions, blocks fixed.  Active block: grid = positions of the "
                     "block being pushed, TCP fixed at the probe (reveals obj_to_goal).  "
                     "TCP + block: grid = block positions, TCP carried along at the probe's current "
                     "offset from the block (the cost of pushing the block there).")
            self.slice = g.add_dropdown("Slice", ["XY (horizontal)", "XZ (vertical)", "YZ (vertical)"],
                                        hint="Vertical slices reveal height-dependent terms (height_match, tcp_floor)")
            self.x_rng = g.add_multi_slider("X range", tx0, tx1, 0.01, xr, min_range=0.02)
            self.y_rng = g.add_multi_slider("Y range", -1.0, 1.0, 0.01, yr, min_range=0.02)
            self.z_rng = g.add_multi_slider("Z range", z0, z1, 0.005,
                                            (round(self.table_z, 3), round(self.table_z + 0.15, 3)),
                                            min_range=0.01,
                                            hint="Used by vertical slices and the 3-D cloud")
            self.z = g.add_slider("Slice z", z0, z1, 0.001, round(self.table_z + self.half, 3))
            self.slice_y = g.add_slider("Slice y", -1.0, 1.0, 0.001, 0.0, visible=False)
            self.slice_x = g.add_slider("Slice x", tx0, tx1, 0.001, 0.5, visible=False)
            self.res = g.add_slider("Resolution (m)", 0.002, 0.02, 0.001, args.res)
            snap = g.add_button("Snap slice to active block")
            behind = g.add_button(
                "Probe → behind active block",
                hint="Put the TCP in contact on the far side of the block from its goal "
                     "(the pushing pose; sets the TCP + block offset)")

        with g.add_folder("Display"):
            self.display = g.add_dropdown("Surface", ["Heightmap", "Plane", "Both", "None"],
                                          initial_value="Both")
            self.hscale = g.add_slider("Height scale (m)", 0.0, 0.5, 0.005, 0.15)
            self.scale_dd = g.add_dropdown(
                "Scale", ["relative", "absolute"],
                hint="relative: min→clip percentile (shows shape; constant terms vanish).  "
                     "absolute: 0→max (constant terms lift the whole surface).")
            self.clip = g.add_slider("Color clip (pctl)", 50.0, 100.0, 0.5, 98.0)
            self.abs_max = g.add_number("Absolute max", 100.0, min=1e-6, step=1.0, visible=False)
            fit = g.add_button("Fit absolute max", visible=False)
            self._abs_widgets = (self.abs_max, fit)
            self.cmap = g.add_dropdown("Colormap", ["viridis", "turbo", "plasma", "magma", "coolwarm"])

        with g.add_folder("Volume (3D cloud)"):
            self.vol_on = g.add_checkbox(
                "Show cloud", False,
                hint="Evaluate the cost on a 3-D grid over the Grid X/Y/Z ranges (same sweep, terms, "
                     "weights, scale and colormap as the surface)")
            self.vol_res = g.add_slider("Volume res (m)", 0.005, 0.03, 0.001, 0.01)
            self.cut_x = g.add_multi_slider("Cut X", tx0, tx1, 0.005, (tx0, tx1), min_range=0.005)
            self.cut_y = g.add_multi_slider("Cut Y", -1.0, 1.0, 0.005, (-1.0, 1.0), min_range=0.005)
            self.cut_z = g.add_multi_slider("Cut Z", z0, z1, 0.005, (z0, z1), min_range=0.005)
            self.vol_pct = g.add_multi_slider(
                "Show cost pctl", 0.0, 100.0, 0.5, (0.0, 100.0), min_range=0.5,
                hint="Only draw points whose total cost lies in this percentile window "
                     "(e.g. 0–10 = the cheapest 10%)")
            self.pt_size = g.add_slider("Point size (m)", 0.001, 0.03, 0.001, 0.006)
            self.vol_md = g.add_markdown("")

        with g.add_folder("Probe"):
            self.probe_md = g.add_markdown("Drag the white probe gizmo.")

        def evaluate(_):
            self._need_eval = True

        def redraw(_):
            self._need_draw = True

        def redraw_cloud(_):
            self._need_vol_draw = True

        for h in (self.vol_on, self.cut_x, self.cut_y, self.cut_z, self.vol_pct, self.pt_size):
            h.on_update(redraw_cloud)

        for h in (self.step_dd, self.mode_dd, self.apply_prev, self.settle, self.sweep,
                  self.x_rng, self.y_rng, self.z_rng, self.res, self.z, self.slice_x, self.slice_y,
                  self.proxy_margin):
            h.on_update(evaluate)
        for h in (*self.enabled.values(), *self.weights.values(),
                  self.display, self.hscale, self.clip, self.cmap, self.abs_max):
            h.on_update(redraw)

        @self.slice.on_update
        def _(_):
            kind = self.slice.value[:2]
            self.z.visible, self.slice_y.visible, self.slice_x.visible = (
                kind == "XY", kind == "XZ", kind == "YZ")
            self._need_eval = True

        @self.scale_dd.on_update
        def _(_):
            for h in self._abs_widgets:
                h.visible = self.scale_dd.value == "absolute"
            self._need_draw = True

        @fit.on_click
        def _(_):
            if self.terms:
                self.abs_max.value = float(max(np.percentile(self._total(self.terms), self.clip.value), 1e-6))
        self.drag_dd.on_update(lambda _: self._toggle_block_ctrls())

        @reset_blocks.on_click
        def _(_):
            self.block_overrides.clear()
            self._need_eval = True

        @all_on.on_click
        def _(_):
            for h in self.enabled.values():
                h.value = True

        @all_off.on_click
        def _(_):
            for h in self.enabled.values():
                h.value = False

        @reset_w.on_click
        def _(_):
            for k, h in self.weights.items():
                h.value = w_cfg[k]

        snap.on_click(lambda _: self._snap_z())
        behind.on_click(lambda _: self._probe_behind_block())

    def _probe_behind_block(self):
        """Move the probe to the pushing pose: behind the active block w.r.t. its push direction."""
        step = self._step()
        if step is None:
            return
        blk = self._block_positions()[self.field.steps[step]["obj_idx"]]
        if step == len(self.field.steps) - 1:       # exit step pushes toward −X
            d = np.array([-1.0, 0.0])
        else:
            d = np.asarray(self.field.steps[step]["end_pos"][:2], dtype=float) - blk[:2]
            d = d / max(np.linalg.norm(d), 1e-9)
        lo_min = min((c[0] for c in self.capsules), default=0.0)
        tip_r = max((r for lo, hi, r in self.capsules if lo <= lo_min + 1e-6), default=0.005)
        standoff = self.half + tip_r
        self.probe.position = (blk[0] - d[0] * standoff, blk[1] - d[1] * standoff, blk[2])
        self._need_eval = True

    def _snap_z(self):
        step = self._step()
        if step is not None:
            x, y, z = self._block_positions()[self.field.steps[step]["obj_idx"]]
            self.z.value, self.slice_y.value, self.slice_x.value = float(z), float(y), float(x)

    def _default_ranges(self):
        pts = [self.blocks0[:, :2]]
        pts += [np.array(s["end_pos"][:2])[None] for s in self.field.steps]
        for c in self.static_cfgs[2:]:                  # bin walls (scenario only) — include their footprint
            ctr, half = np.array(c.init_state.pos[:2]), np.array(c.spawn.size[:2]) / 2
            pts += [(ctr - half)[None], (ctr + half)[None]]
        pts = np.concatenate(pts)
        lo, hi = pts.min(0) - 0.12, pts.max(0) + 0.12
        lo[0], hi[0] = max(lo[0], -0.05), min(hi[0], 1.35)
        lo[1], hi[1] = max(lo[1], -1.0), min(hi[1], 1.0)
        return (round(lo[0], 2), round(hi[0], 2)), (round(lo[1], 2), round(hi[1], 2))

    def _toggle_block_ctrls(self):
        for c in self.block_ctrls.values():
            c.remove()
        self.block_ctrls = {}
        self._ctrl_step = self._step()
        mode = self.drag_dd.value
        if mode == "off" or self._ctrl_step is None:
            return
        pos_all = self._block_positions()
        active = self.field.steps[self._ctrl_step]["obj_idx"]
        for i in (range(len(pos_all)) if mode == "all" else [active]):
            ctrl = self.server.scene.add_transform_controls(
                f"/block_ctrl/{i}", scale=0.07, disable_rotations=True, depth_test=False,
                position=tuple(pos_all[i]))

            def on_move(_, i=i, ctrl=ctrl):
                self.block_overrides[i] = np.array(ctrl.position)
                self._need_eval = True

            ctrl.on_update(on_move)
            self.block_ctrls[i] = ctrl

    def _sync_block_ctrls(self, blocks):
        """Keep gizmos on their blocks after step changes / resets (not while dragged)."""
        if self.drag_dd.value == "active block" and self._ctrl_step != self._step():
            self._toggle_block_ctrls()            # the active block changed
            return
        for i, ctrl in self.block_ctrls.items():
            if i not in self.block_overrides:
                ctrl.position = tuple(blocks[i])

    # ── state ───────────────────────────────────────────────────────────────
    def _step(self):
        if not self.field.steps:
            return None
        return self.step_dd.options.index(self.step_dd.value)

    def _block_positions(self):
        pos = self.blocks0.copy()
        step = self._step()
        if step is not None and self.apply_prev.value:
            for s in self.field.steps[:step]:
                pos[s["obj_idx"]] = s["end_pos"]
            cur = self.field.steps[step]
            if "start_pos" in cur:
                pos[cur["obj_idx"]] = cur["start_pos"]
        if self.settle.value:
            pos = settle_blocks(pos, self.table_z, self.half)
        for i, p in self.block_overrides.items():
            pos[i] = p
        return pos

    def _grid(self):
        """Grid points (N, 3) on the selected axis-aligned slice, its (nu, nv) shape and normal axis."""
        r = self.res.value
        rng = {0: self.x_rng.value, 1: self.y_rng.value, 2: self.z_rng.value}
        kind = self.slice.value[:2]
        u_ax, v_ax, n_ax, w = {"XY": (0, 1, 2, self.z.value),
                               "XZ": (0, 2, 1, self.slice_y.value),
                               "YZ": (1, 2, 0, self.slice_x.value)}[kind]
        us = np.arange(rng[u_ax][0], rng[u_ax][1] + 1e-9, r)
        vs = np.arange(rng[v_ax][0], rng[v_ax][1] + 1e-9, r)
        U, V = np.meshgrid(us, vs)                      # (nv, nu), row-major like grid_mesh
        pts = np.empty((U.size, 3))
        pts[:, u_ax], pts[:, v_ax], pts[:, n_ax] = U.ravel(), V.ravel(), w
        return pts, len(us), len(vs), n_ax

    def _total(self, terms):
        n = len(next(iter(terms.values())))
        total = np.zeros(n)
        for k in ALL_TERMS:
            if self.enabled[k].value:
                total += self.weights[k].value * terms[k]
        return total

    def _eval_points(self, pts, blocks, step, chunk: int = 100_000):
        """Evaluate all terms at grid points `pts` for the current sweep (TCP or active block)."""
        parts, goal = [], None
        for a in range(0, len(pts), chunk):
            p = pts[a:a + chunk]
            if self.sweep.value == "TCP":
                terms, goal = self._eval_terms(p, blocks, step)
            else:   # grid = active block positions
                idx = self.field.steps[step]["obj_idx"]
                per_env = np.repeat(blocks[None], len(p), axis=0)
                per_env[:, idx] = p
                probe = np.array(self.probe.position, dtype=float)
                if self.sweep.value == "Active block":      # TCP fixed at the probe
                    tcp = np.repeat(probe[None], len(p), axis=0)
                else:                                       # TCP carried with the block
                    tcp = p + (probe - blocks[idx])
                terms, goal = self._eval_terms(tcp, per_env, step)
            parts.append(terms)
        return {k: np.concatenate([t[k] for t in parts]) for k in parts[0]}, goal

    def _normalize(self, total):
        """Map total cost to [0, 1] with the current Scale settings → (norm, lo, hi)."""
        if self.scale_dd.value == "absolute":
            lo, hi = 0.0, self.abs_max.value
        else:
            lo, hi = total.min(), np.percentile(total, self.clip.value)
        span = hi - lo if hi - lo > 1e-9 else 1.0
        return np.clip((total - lo) / span, 0.0, 1.0), lo, hi

    def _eval_terms(self, tcp, blocks, step):
        terms, goal = self.field.evaluate(tcp, blocks, step, self.mode_dd.value == "push")
        terms[BIN_PROXY] = bin_penetration(tcp, self.walls, self.capsules, self.proxy_margin.value)
        return terms, goal

    # ── compute + draw ──────────────────────────────────────────────────────
    def _evaluate(self):
        step = self._step()
        if step is None:
            return
        blocks = self._block_positions()
        pts, nx, ny, n_ax = self._grid()
        t0 = time.time()
        terms, goal = self._eval_points(pts, blocks, step)
        self.eval_ms = (time.time() - t0) * 1e3
        self.terms, self.goal, self.pts, self.nx, self.ny, self.n_ax = terms, goal, pts, nx, ny, n_ax
        self.blocks = blocks
        for i, h in enumerate(self.block_handles):
            h.position = tuple(blocks[i])
        self._sync_block_ctrls(blocks)
        self._draw_context(step, blocks, goal)
        self._update_probe()

    def _draw_context(self, step, blocks, goal):
        s = self.server.scene
        st = self.field.steps[step]
        obj = blocks[st["obj_idx"]]
        s.add_icosphere("/goal", radius=0.012, color=(40, 220, 90), position=tuple(goal))
        s.add_label("/goal/label", f"goal step {step}", position=(0, 0, 0.03))
        s.add_box("/active_marker", color=(255, 215, 0), dimensions=(0.015, 0.015, 0.015),
                  position=tuple(obj + [0, 0, 0.08]))
        is_last = step == len(self.field.steps) - 1
        if is_last:   # exit step: planner pushes toward −X past the exit line
            end = np.array([self.cfg.costs.intruder_exit_x - 0.08, obj[1], obj[2]])
        else:
            end = np.asarray(goal, dtype=float)
        s.add_line_segments("/push_dir", np.array([[obj, end]]), (255, 140, 0), line_width=3)

    def _draw(self):
        import matplotlib
        import trimesh
        if not self.terms:
            return
        total = self._total(self.terms)
        nx, ny, ax = self.nx, self.ny, self.n_ax
        norm, lo, hi = self._normalize(total)
        cmap = matplotlib.colormaps[self.cmap.value]
        rgba = (cmap(norm) * 255).astype(np.uint8)

        faces = grid_mesh(nx, ny)
        faces2 = np.concatenate([faces, faces[:, ::-1] + len(total)])   # double-sided

        def displaced(offset):              # push vertices along the slice normal
            v = self.pts.copy()
            v[:, ax] += offset
            return v

        def mesh(v):
            return trimesh.Trimesh(np.concatenate([v, v]), faces2,
                                   vertex_colors=np.concatenate([rgba, rgba]), process=False)

        mode = self.display.value
        s = self.server.scene
        height_v = displaced(self.hscale.value * norm)
        s.add_mesh_trimesh("/surface/height", mesh(height_v), visible=mode in ("Heightmap", "Both"))
        s.add_mesh_trimesh("/surface/plane", mesh(displaced(5e-4)), visible=mode in ("Plane", "Both"))
        i = int(np.argmin(total))
        s.add_icosphere("/argmin", radius=0.007, color=(255, 30, 30),
                        position=tuple(height_v[i] if mode != "Plane" else self.pts[i]),
                        visible=mode != "None")

        bar = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        self.colorbar.image = np.repeat(bar[None], 12, axis=0)
        active = [k for k in ALL_TERMS if self.enabled[k].value]
        flat = []
        for k in ALL_TERMS:
            v = self.terms[k]
            if k not in CONST_TERMS and np.ptp(v) <= 1e-7 * max(1.0, abs(v.mean())):
                push = self.mode_dd.value == "push"
                if not v.any() and (k in ("obj_avoidance", "above_target") and push
                                    or k == "obj_to_goal" and not push):
                    why = f"zeroed by the planner in {self.mode_dd.value} mode"
                else:
                    why = FLAT_REASON.get(k, "constant over this slice")
                on = "" if self.enabled[k].value else " (off)"
                flat.append(f"- {k}{on}: {self.weights[k].value * v[0]:.3f} — {why}")
        self.flat_md.content = ("**Flat over this slice** (shift the surface, no shape):\n" + "\n".join(flat)
                                if flat else "")
        self.readout.content = (
            f"**Total cost** over {nx}×{ny} grid ({self.eval_ms:.0f} ms)  \n"
            f"min **{total.min():.3f}** · color range [{lo:.3f}, {hi:.3f}] · max {total.max():.3f}  \n"
            f"argmin {'TCP' if self.sweep.value == 'TCP' else 'block'} xyz "
            f"`({self.pts[i, 0]:.3f}, {self.pts[i, 1]:.3f}, {self.pts[i, 2]:.3f})`  \n"
            f"terms: {', '.join(active) or '—'}"
        )

    # ── 3-D cloud ───────────────────────────────────────────────────────────
    _VOL_MAX_POINTS = 1_000_000

    def _vol_grid(self):
        r = self.vol_res.value
        axes = [np.arange(lo, hi + 1e-9, r) for lo, hi in
                (self.x_rng.value, self.y_rng.value, self.z_rng.value)]
        return axes, int(np.prod([len(a) for a in axes]))

    def _vol_signature(self):
        """Everything the volume evaluation depends on; re-evaluate when it changes."""
        probe = tuple(np.round(self.probe.position, 5)) if self.sweep.value != "TCP" else None
        return (self._step(), self.mode_dd.value, self.sweep.value, probe,
                self._block_positions().round(5).tobytes(), tuple(self.x_rng.value),
                tuple(self.y_rng.value), tuple(self.z_rng.value), self.vol_res.value,
                self.proxy_margin.value)

    def _eval_volume(self):
        step = self._step()
        axes, n = self._vol_grid()
        if step is None or n > self._VOL_MAX_POINTS:
            self.vol_terms = {}
            self.vol_md.content = (f"**{n:,} points** — over the {self._VOL_MAX_POINTS:,} limit; "
                                   "coarsen *Volume res* or shrink the Grid ranges.")
            return
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        self.vol_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        t0 = time.time()
        self.vol_terms, _ = self._eval_points(self.vol_pts, self._block_positions(), step)
        self.vol_ms = (time.time() - t0) * 1e3

    def _draw_cloud(self):
        import matplotlib
        s = self.server.scene
        if not self.vol_on.value or not self.vol_terms:
            s.add_point_cloud("/volume", np.zeros((0, 3), np.float32), (0, 0, 0), visible=False)
            if not self.vol_on.value:
                self.vol_md.content = ""
            return
        total = self._total(self.vol_terms)
        norm, lo, hi = self._normalize(total)
        pts = self.vol_pts
        plo, phi = np.percentile(total, self.vol_pct.value)
        mask = (total >= plo) & (total <= phi)
        for ax, cut in enumerate((self.cut_x, self.cut_y, self.cut_z)):
            mask &= (pts[:, ax] >= cut.value[0]) & (pts[:, ax] <= cut.value[1])
        rgb = (matplotlib.colormaps[self.cmap.value](norm[mask])[:, :3] * 255).astype(np.uint8)
        s.add_point_cloud("/volume", pts[mask].astype(np.float32), rgb,
                          point_size=self.pt_size.value, point_shape="circle", precision="float32")
        i = int(np.argmin(total))
        self.vol_md.content = (
            f"{len(pts):,} points evaluated ({self.vol_ms:.0f} ms) · **{int(mask.sum()):,} shown**  \n"
            f"cost min {total.min():.3f} · max {total.max():.3f} · shown window [{plo:.3f}, {phi:.3f}]  \n"
            f"argmin {'TCP' if self.sweep.value == 'TCP' else 'block'} xyz "
            f"`({pts[i, 0]:.3f}, {pts[i, 1]:.3f}, {pts[i, 2]:.3f})`")

    def _update_probe(self):
        step = self._step()
        if step is None or not hasattr(self, "blocks"):
            return
        p = np.array(self.probe.position)[None]
        terms, _ = self._eval_terms(p, self.blocks, step)
        rows = ["| term | raw | w | w·raw |", "|---|---:|---:|---:|"]
        total = 0.0
        for k in ALL_TERMS:
            on = self.enabled[k].value
            w = self.weights[k].value
            v = float(terms[k][0])
            total += w * v if on else 0.0
            name = k if on else f"~~{k}~~"
            rows.append(f"| {name} | {v:.4f} | {w:g} | {w * v:.4f} |")
        self.probe_md.content = (f"probe `({p[0, 0]:.3f}, {p[0, 1]:.3f}, {p[0, 2]:.3f})`  \n"
                                 f"**total {total:.4f}**\n\n" + "\n".join(rows))

    def run(self):
        print(f"[cost_viewer] open http://localhost:{self.server.get_port()}", flush=True)
        while True:
            if self._need_eval:
                self._need_eval = False
                self._need_draw = True
                self._evaluate()
            if self._need_draw:
                self._need_draw = False
                self._need_vol_draw = True
                self._draw()
                self._update_probe()
            if self.vol_on.value:
                key = self._vol_signature()
                if key != self._vol_key:
                    self._vol_key = key
                    self._need_vol_draw = True
                    self._eval_volume()
            if self._need_vol_draw:
                self._need_vol_draw = False
                self._draw_cloud()
            time.sleep(0.03)


# ===========================================================================
# Main
# ===========================================================================

def main():
    cfg = P._load_config(_CFG_PATH)
    if args.solution_path:
        cfg.solution_path = args.solution_path

    blocks0, bin_size, src = load_initial_blocks(cfg.solution_path, args.scenario)
    print(f"[cost_viewer] {len(blocks0)} blocks from {src}")
    block_cfgs = make_block_cfgs(positions=blocks0)
    static_cfgs = make_static_cfgs(stand_urdf=cfg.stand_urdf, bin_size=bin_size)
    table_cfg = static_cfgs[0]
    table_surface_z = table_cfg.init_state.pos[2] + table_cfg.spawn.size[2] / 2   # as planner.main()

    field = CostField(cfg, table_surface_z, n_blocks=len(block_cfgs), device=args.device)
    if getattr(field.obj, "_target_frame_unresolved", False):
        field.resolve_target_frame(blocks0[0])

    if args.selftest:
        sys.exit(0 if selftest(cfg, field, table_surface_z, len(block_cfgs), blocks0) else 1)

    colors = [c.spawn.visual_material.diffuse_color for c in block_cfgs]
    CostViewer(cfg, field, static_cfgs, blocks0, colors, table_surface_z, args.port).run()


if __name__ == "__main__":
    main()
