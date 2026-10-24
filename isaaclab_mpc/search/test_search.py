"""Tests for the hidden-object search (numpy only; no simulator).

    python -m pytest isaaclab_mpc/search/test_search.py -q
"""
import math

import numpy as np
import pytest

from isaaclab_mpc.search.belief import Box, FloorBelief, segments_hit_box
from isaaclab_mpc.search.search_mcts import pick_waypoint, search, segments_hit_boxes

# Tote-like geometry (matches examples/ur16e_reach_stand_tote): interior, floor and rim heights.
INTERIOR = ((0.16, 0.64), (-0.17, 0.17))
FLOOR_Z, RIM_Z, TABLE_Z = 0.734, 1.007, 0.72
WALL_T = 0.03


def walls():
    (x0, x1), (y0, y1) = INTERIOR
    zc, hz = (TABLE_Z + RIM_Z) / 2, (RIM_Z - TABLE_Z) / 2
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2 + WALL_T, (y1 - y0) / 2 + WALL_T
    return [
        Box(np.array([x0 - WALL_T / 2, ym, zc]), np.array([WALL_T / 2, hy, hz])),
        Box(np.array([x1 + WALL_T / 2, ym, zc]), np.array([WALL_T / 2, hy, hz])),
        Box(np.array([xm, y0 - WALL_T / 2, zc]), np.array([hx, WALL_T / 2, hz])),
        Box(np.array([xm, y1 + WALL_T / 2, zc]), np.array([hx, WALL_T / 2, hz])),
    ]


def cracker_box(x, y):
    return Box(np.array([x, y, FLOOR_Z + 0.036]), np.array([0.107, 0.0815, 0.036]))


PARAMS = dict(
    n_dirs=6, n_sub=3, speed=0.1, action_time=0.5, iterations=800, c=0.7, depth=10, rollout_depth=4,
    workspace_lo=[0.16, -0.17, FLOOR_Z], workspace_hi=[0.64, 0.17, 1.10],
    rod_radius=0.006, coupling_radius=0.04, coupling_bottom=0.101, coupling_top=0.135,
    sensor_height=0.118, fov=math.radians(45.0),
    w_info=1.0, w_push=0.02, w_collision=1.0, w_obstacle=0.0, margin=0.02, w_dturn=0.01, w_len=0.05,
    w_heur=0.3, heur_sigma=0.1, push_macros=True, push_dist=0.06, n_views=3, view_grid=0.05,
)


def belief_only(mask_fn):
    b = FloorBelief(*INTERIOR, floor_z=FLOOR_Z, res=0.01)
    keep = mask_fn(b.cells)
    b.p = np.where(keep, 1.0, 0.0)
    b.p /= b.p.sum()
    return b


def path_is_safe(path):
    """Every path point inside the tote interior (with the rod's radius) and above the floor."""
    (x0, x1), (y0, y1) = INTERIOR
    r = PARAMS["rod_radius"]
    return bool(np.all((path[:, 0] >= x0 + r) & (path[:, 0] <= x1 - r) & (path[:, 1] >= y0 + r)
                       & (path[:, 1] <= y1 - r) & (path[:, 2] >= FLOOR_Z + r)))


# ── belief ────────────────────────────────────────────────────────────────────

def test_floor_hit_clears_its_patch_and_renormalises():
    b = FloorBelief(*INTERIOR, floor_z=FLOOR_Z, res=0.01)
    o = np.array([[0.40, 0.0, 1.00]]); d = np.array([[0.0, 0.0, -1.0]])
    cleared = b.observe(o, d, np.array([1.00 - FLOOR_Z]), zone_half_angle=math.radians(45 / 16))
    near = np.sum((b.cells - [0.40, 0.0]) ** 2, axis=-1) < 0.005 ** 2
    assert cleared > 0 and np.all(b.p[near] == 0)
    assert b.p[0, 0] > 0 and math.isclose(b.p.sum(), 1.0)


def test_hit_on_an_object_leaves_the_cells_beneath():
    b = FloorBelief(*INTERIOR, floor_z=FLOOR_Z, res=0.01)
    p0 = b.p.copy()
    b.observe(np.array([[0.40, 0.0, 1.00]]), np.array([[0.0, 0.0, -1.0]]), np.array([0.20]),
              zone_half_angle=math.radians(45 / 16))       # hit at z 0.80: an object's top
    assert np.array_equal(b.p, p0)


def test_no_return_changes_nothing():
    b = FloorBelief(*INTERIOR, floor_z=FLOOR_Z, res=0.01)
    p0 = b.p.copy()
    b.observe(np.array([[0.4, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]]), np.array([np.inf]), 0.05)
    assert np.array_equal(b.p, p0)


# ── visibility geometry ───────────────────────────────────────────────────────

def test_cells_under_and_behind_a_box_are_occluded():
    box = cracker_box(0.33, 0.0)
    sensor = np.array([0.33, 0.0, 1.0])
    under = np.array([0.33, 0.0, FLOOR_Z + 0.002])
    open_cell = np.array([0.55, 0.12, FLOOR_Z + 0.002])
    assert segments_hit_box(sensor[None], under[None], box)[0]
    assert not segments_hit_box(sensor[None], open_cell[None], box)[0]
    # vectorised version agrees
    hits = segments_hit_boxes(np.stack([sensor, sensor]), np.stack([under, open_cell]),
                              box.center[None], box.half[None], np.array([box.yaw]))
    assert hits[:, 0].tolist() == [True, False]


def test_rotated_box_occlusion():
    box = Box(np.array([0.4, 0.0, FLOOR_Z + 0.036]), np.array([0.15, 0.01, 0.036]), yaw=math.pi / 2)
    s = np.array([[0.4, -0.1, 1.0]])
    assert segments_hit_box(s, np.array([[0.4, 0.1, FLOOR_Z]]), box)[0]     # the long side now runs along y
    assert not segments_hit_box(s, np.array([[0.55, 0.0, FLOOR_Z]]), box)[0]


def test_pick_waypoint():
    path = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]])
    assert np.allclose(pick_waypoint(path, 0.15), [0.2, 0, 0])
    assert np.allclose(pick_waypoint(path, 5.0), [0.3, 0, 0])
    assert pick_waypoint(path[:1], 0.1) is None


# ── search ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_open_mass_is_found_by_moving_not_pushing(seed):
    """Remaining mass in an open corner: the tree should go and look, without pushing."""
    b = belief_only(lambda c: (c[..., 0] > 0.52) & (c[..., 1] > 0.06))
    res = search([0.30, -0.08, 0.90], b, [cracker_box(0.33, 0.0)], walls(), INTERIOR, PARAMS, seed=seed)
    end = res["best_path"][-1]
    assert not res["push_planned"]
    assert res["best_gain"] > 0.5, res["best_gain"]
    assert np.hypot(end[0] - 0.58, end[1] - 0.115) < np.hypot(0.30 - 0.58, -0.08 - 0.115)  # heads for it
    assert path_is_safe(res["best_path"])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_mass_only_under_a_box_needs_a_push(seed):
    """All remaining mass beneath a box that no viewpoint can see under: the tree must push it.

    The tool's 4 cm coupling sits below the rim when the tip is low, so it can only go down
    away from the walls; here there is room on the +x side to push the box toward -x."""
    box = cracker_box(0.33, 0.0)
    b = belief_only(lambda c: box.contains_xy(c))
    res = search([0.50, 0.0, 0.90], b, [box], walls(), INTERIOR, PARAMS, seed=seed)
    assert res["push_planned"], res
    assert res["best_gain"] > 0.05, res["best_gain"]     # a push uncovers a strip of the footprint
    assert path_is_safe(res["best_path"])


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_path_never_enters_walls(seed):
    b = FloorBelief(*INTERIOR, floor_z=FLOOR_Z, res=0.01)
    start = [0.18, 0.15, 0.80]                      # near a corner, below the rim
    res = search(start, b, [cracker_box(0.40, -0.05)], walls(), INTERIOR, PARAMS, seed=seed)
    assert path_is_safe(res["best_path"])


# ── guide ─────────────────────────────────────────────────────────────────────

def test_waypoint_stops_at_sharp_corners_until_reached():
    from isaaclab_mpc.search.guide import SearchGuide
    g = SearchGuide(dict(PARAMS, waypoint={"enabled": True, "lookahead": 0.10, "corner_deg": 45, "corner_tol": 0.015}),
                    INTERIOR, FLOOR_Z, walls(), [])
    # descend 5 cm, then sweep 10 cm sideways (a push macro's last corner)
    g.path = np.array([[0.40, 0.0, 0.85], [0.40, 0.0, 0.80], [0.40, 0.10, 0.80]])
    above = g.path_point(np.array([0.40, 0.0, 0.84]), 0.10)
    assert np.allclose(above, [0.40, 0.0, 0.80])                  # gated at the corner, not cut diagonally
    at_corner = g.path_point(np.array([0.40, 0.0, 0.801]), 0.10)
    assert at_corner[1] > 0.05                                    # past the gate: looks along the sweep
    g.p["waypoint"]["corner_deg"] = 120                           # a 90° turn no longer counts as sharp
    assert g.path_point(np.array([0.40, 0.0, 0.84]), 0.10)[1] > 0.0


def test_tool_inside_a_box_footprint_can_still_plan():
    """Tip just below a box's top, inside its footprint (e.g. after a push went wrong): lifting
    out must be possible, so the tree still returns a plan."""
    box = cracker_box(0.33, 0.0)
    b = FloorBelief(*INTERIOR, floor_z=FLOOR_Z, res=0.01)
    res = search([0.33, 0.02, box.top - 0.002], b, [box], walls(), INTERIOR, PARAMS, seed=0)
    assert len(res["best_path"]) > 1


def test_fallback_goal_lifts_the_tool_when_there_is_no_path():
    from isaaclab_mpc.search.guide import SearchGuide
    g = SearchGuide(dict(PARAMS, fallback_z=0.95), INTERIOR, FLOOR_Z, walls(), [])
    g.tcp, g.path = np.array([0.4, 0.0, 0.80]), None
    assert np.allclose(g.mppi_goal(), [0.4, 0.0, 0.95])
    g.tcp = np.array([0.4, 0.0, 1.00])                 # already above fallback_z: lifts 5 cm more
    assert np.isclose(g.mppi_goal()[2], 1.05)
    g.tcp = np.array([0.4, 0.0, 1.08])                 # near the 1.10 ceiling: capped 2 cm below it
    assert np.isclose(g.mppi_goal()[2], PARAMS["workspace_hi"][2] - 0.02)
