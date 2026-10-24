"""Perception tests on synthetic ToF frames (no simulator).

A numpy ray caster plays VL53L5CX sensors (closest return per zone) against box stand-ins for the
objects and the tote; the pipeline must recover the objects' poses from the frames alone.

    python -m pytest isaaclab_mpc/perception/test_perception.py -q
"""
import math

import numpy as np
import pytest

from isaaclab_mpc.perception.scene_perception import ScenePerception, models_from_catalogue
from isaaclab_mpc.search.belief import Box

INTERIOR = ((0.162, 0.638), (-0.17, 0.17))
TABLE_Z, FLOOR_Z, RIM_Z = 0.72, 0.7343, 1.007
LO, HI = (0.15, -0.18, 0.725), (0.65, 0.18, 1.0)
NAMES = ["cracker_box", "soup_can", "sugar_box"]
SIZES = {"cracker_box": (0.214, 0.163, 0.072), "soup_can": (0.068, 0.067, 0.102),
         "sugar_box": (0.176, 0.092, 0.045), "tuna_can": (0.085, 0.085, 0.033)}
FOV, ZONES = math.radians(45.0), 8


def tote_boxes():
    (x0, x1), (y0, y1) = INTERIOR
    t, zc, hz = 0.03, (TABLE_Z + RIM_Z) / 2, (RIM_Z - TABLE_Z) / 2
    xm, ym, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 + t, (y1 - y0) / 2 + t
    return [Box(np.array([x0 - t / 2, ym, zc]), np.array([t / 2, hy, hz])),
            Box(np.array([x1 + t / 2, ym, zc]), np.array([t / 2, hy, hz])),
            Box(np.array([xm, y0 - t / 2, zc]), np.array([hx, t / 2, hz])),
            Box(np.array([xm, y1 + t / 2, zc]), np.array([hx, t / 2, hz])),
            Box(np.array([xm, ym, (TABLE_Z + FLOOR_Z) / 2]), np.array([hx, hy, (FLOOR_Z - TABLE_Z) / 2]))]


def obj_box(name, x, y, yaw=0.0, bottom=FLOOR_Z):
    sx, sy, sz = SIZES[name]
    return Box(np.array([x, y, bottom + sz / 2]), np.array([sx / 2, sy / 2, sz / 2]), yaw)


def raycast(o, d, boxes):
    """Closest hit distance along rays (R, 3) against upright boxes → (R,), inf on a miss."""
    best = np.full(len(o), np.inf)
    for b in boxes:
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
        a, v = (o - b.center) @ R.T, d @ R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-b.half - a) / v, (b.half - a) / v
        tmin = np.where(v == 0, np.where(np.abs(a) <= b.half, -np.inf, np.inf), np.minimum(t1, t2)).max(1)
        tmax = np.where(v == 0, np.where(np.abs(a) <= b.half, np.inf, -np.inf), np.maximum(t1, t2)).min(1)
        hit = (tmin <= tmax) & (tmax > 0)
        t = np.where(tmin > 0, tmin, tmax)
        best = np.where(hit & (t < best), t, best)
    return best


def zone_dirs(k=1):
    """Downward-looking zone directions (ZONES*ZONES, k*k, 3); boresight -Z."""
    sub = (np.arange(k) + 0.5) / k
    g = ((np.arange(ZONES)[:, None] + sub[None]) / ZONES * FOV - FOV / 2)        # (ZONES, k)
    ax = g[None, :, None, :].repeat(ZONES, 0).repeat(k, 2)
    ay = g[:, None, :, None].repeat(ZONES, 1).repeat(k, 3)
    d = np.stack([np.tan(ax), np.tan(ay), -np.ones_like(ax)], -1).reshape(ZONES * ZONES, k * k, 3)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def tof_frame(sensor, boxes, k=4):
    """VL53L5CX-like frame from `sensor` looking down: zone-centre dirs + closest return per zone."""
    sub = zone_dirs(k)                                                       # (Z, k*k, 3)
    centre = zone_dirs(1)[:, 0]
    o = np.repeat(np.asarray(sensor, float)[None], sub.shape[0] * sub.shape[1], 0)
    t = raycast(o, sub.reshape(-1, 3), boxes).reshape(sub.shape[:2])
    dist = t.min(1)
    dist[dist > 4.0] = np.inf
    return np.repeat(np.asarray(sensor, float)[None], len(centre), 0), centre, dist


VIEWS = [(x, y, 1.02) for x in (0.24, 0.40, 0.56) for y in (-0.10, 0.0, 0.10)]


def perceive(scene, views=VIEWS, perc=None, names=NAMES, passes=2):
    perc = perc or ScenePerception(LO, HI, tote_boxes(), models_from_catalogue(names), FLOOR_Z, res=0.01)
    boxes = tote_boxes() + scene
    for _ in range(passes):
        for v in views:
            o, d, r = tof_frame(v, boxes)
            perc.add_tof(o, d, r, zone_angle=FOV / ZONES)
    perc.update()
    return perc


def yaw_err(a, b, period):
    if period == 0:
        return 0.0
    e = (a - b) % period
    return min(e, period - e)


def check(perc, name, x, y, z=None, tol_xy=0.02, tol_yaw=math.radians(12)):
    assert name in perc.tracks, f"{name} not found; tracks {list(perc.tracks)}"
    f = perc.tracks[name]
    assert math.hypot(f.pos[0] - x, f.pos[1] - y) < tol_xy, (name, f.pos, (x, y))
    if z is not None:
        assert abs(f.pos[2] - z) < 0.012, (name, f.pos[2], z)
    return f


def test_single_box_pose():
    perc = perceive([obj_box("cracker_box", 0.40, 0.02, yaw=0.3)])
    f = check(perc, "cracker_box", 0.40, 0.02, z=FLOOR_Z + 0.036)
    assert yaw_err(f.yaw, 0.3, perc.models["cracker_box"].yaw_period) < math.radians(12)


def test_three_objects():
    scene = [obj_box("cracker_box", 0.40, 0.02, 0.3), obj_box("soup_can", 0.56, -0.11),
             obj_box("sugar_box", 0.24, 0.10, 1.2)]
    perc = perceive(scene)
    check(perc, "cracker_box", 0.40, 0.02)
    check(perc, "soup_can", 0.56, -0.11)
    f = check(perc, "sugar_box", 0.24, 0.10)
    assert yaw_err(f.yaw, 1.2, perc.models["sugar_box"].yaw_period) < math.radians(15)


def test_box_on_a_hidden_can_is_placed_at_its_true_height():
    can = obj_box("tuna_can", 0.33, 0.0)
    box = obj_box("cracker_box", 0.33, 0.0, bottom=FLOOR_Z + 0.033)
    perc = perceive([can, box])                    # the tuna can is not a known object here
    check(perc, "cracker_box", 0.33, 0.0, z=FLOOR_Z + 0.033 + 0.036)


def test_moved_object_is_re_tracked():
    perc = perceive([obj_box("cracker_box", 0.42, 0.04, 0.0)])
    check(perc, "cracker_box", 0.42, 0.04)
    perceive([obj_box("cracker_box", 0.28, -0.07, 0.0)], perc=perc, passes=3)   # pushed elsewhere
    check(perc, "cracker_box", 0.28, -0.07)
