"""Belief over where a hidden target lies on the tote floor, plus the box geometry the search uses.

The target rests on the floor and the ToF sensors look down from the tool, so the
belief is a 2.5D grid over the floor: p[i, j] = probability the target is in cell
(i, j). A zone whose closest hit is the floor clears its footprint patch (nothing
stands there). A hit on a known object's surface says nothing about what lies
beneath it, so the cells under known objects keep their probability.

Known objects and the tote walls are upright oriented boxes, used for occlusion
(can the sensor see a floor cell?) and for the push model.

numpy only; no simulator dependency.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    """Upright box: centre (x, y, z), half extents (hx, hy, hz), rotation `yaw` about +Z."""
    center: np.ndarray
    half: np.ndarray
    yaw: float = 0.0

    def moved(self, dxy) -> "Box":
        c = self.center.copy()
        c[:2] += dxy
        return Box(c, self.half, self.yaw)

    @property
    def top(self) -> float:
        return float(self.center[2] + self.half[2])

    def corners_xy(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        local = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * self.half[:2]
        return local @ np.array([[c, s], [-s, c]]) + self.center[:2]

    def to_local_xy(self, xy: np.ndarray) -> np.ndarray:
        """World XY points (..., 2) → box frame."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        d = xy - self.center[:2]
        return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)

    def contains_xy(self, xy: np.ndarray, pad: float = 0.0) -> np.ndarray:
        q = np.abs(self.to_local_xy(xy))
        return (q[..., 0] <= self.half[0] + pad) & (q[..., 1] <= self.half[1] + pad)


def yaw_from_quat(q) -> float:
    """Yaw (about +Z) of a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def segments_hit_box(p0: np.ndarray, p1: np.ndarray, box: Box, pad: float = 0.0) -> np.ndarray:
    """Slab test: do segments p0[k] → p1[k] (K, 3) pass through the (padded) box? → (K,) bool."""
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])        # world → box frame
    a = (p0 - box.center) @ R.T
    d = (p1 - p0) @ R.T
    h = box.half + pad
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1, t2 = (-h - a) * inv, (h - a) * inv
    tmin = np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2))
    tmax = np.where(np.isnan(t2), np.inf, np.maximum(t1, t2))
    # Parallel to a slab and outside it: never inside.
    outside = (d == 0) & (np.abs(a) > h)
    tmin = np.where(outside, np.inf, tmin)
    t_enter = np.max(tmin, axis=1)
    t_exit = np.min(tmax, axis=1)
    return (t_enter <= t_exit) & (t_exit >= 0.0) & (t_enter <= 1.0)


def rects_overlap(a: Box, b: Box, margin: float = 0.0) -> bool:
    """Separating-axis test on the two boxes' XY footprints."""
    ca, cb = a.corners_xy(), b.corners_xy()
    for poly in (ca, cb):
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        pa, pb = ca @ normals.T, cb @ normals.T
        if np.any((pa.max(0) + margin < pb.min(0)) | (pb.max(0) + margin < pa.min(0))):
            return False
    return True


class FloorBelief:
    """p over floor cells of an axis-aligned region (the tote interior)."""

    def __init__(self, x_range, y_range, floor_z: float, res: float = 0.01):
        self.res = res
        self.floor_z = floor_z
        self.xs = np.arange(x_range[0] + res / 2, x_range[1], res)
        self.ys = np.arange(y_range[0] + res / 2, y_range[1], res)
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.cells = np.stack([X, Y], axis=-1)                       # (nx, ny, 2) cell centres
        self.p = np.full(X.shape, 1.0 / X.size)

    @property
    def shape(self):
        return self.p.shape

    def cells_floor(self) -> np.ndarray:
        """(nx, ny, 3) cell centres on the floor."""
        z = np.full(self.p.shape + (1,), self.floor_z)
        return np.concatenate([self.cells, z], axis=-1)

    def observe(self, origins, dirs, dists, zone_half_angle: float, floor_tol: float = 0.012) -> int:
        """Clear the footprint patch of every zone whose closest hit is the floor.

        origins, dirs : (M, 3) zone-centre ray origins and unit directions (world frame)
        dists         : (M,) zone distances (m); inf = no target
        Returns the number of cells cleared.
        """
        origins, dirs, dists = (np.asarray(v, dtype=float) for v in (origins, dirs, dists))
        ok = np.isfinite(dists)
        before = self.p > 0
        for o, d, r in zip(origins[ok], dirs[ok], dists[ok]):
            hit = o + r * d
            if hit[2] > self.floor_z + floor_tol:
                continue                    # an object (known or not): no info beneath it
            radius = max(r * np.tan(zone_half_angle), self.res / 2)
            near = np.sum((self.cells - hit[:2]) ** 2, axis=-1) <= radius ** 2
            self.p[near] = 0.0
        cleared = int((before & (self.p == 0)).sum())
        total = self.p.sum()
        if cleared and total > 0:
            self.p /= total
        return cleared

    def mass(self) -> float:
        """Unnormalised remaining support (fraction of cells still possible)."""
        return float((self.p > 0).mean())
