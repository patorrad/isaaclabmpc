"""Known-object models, segmentation of the voxel map, and pose fitting.

Pipeline per update:
  1. occupied voxels minus known static structure (tote, table) → connected components (clusters)
  2. each cluster is matched against the known object models not yet placed: over yaw and a small
     x/y search around the cluster, with the object's top at the cluster's top (from above the
     sensors see top surfaces; this also places a box resting on something hidden at the right
     height). Score:
        explained : fraction of the cluster's voxels inside the object's (padded) box
        support   : fraction of the object's observed surface points that land on occupied voxels
        conflict  : fraction of the object's interior points in confidently free voxels
        score = explained + support - w_conflict * conflict
  3. greedy assignment, best score first, one pose per object; a cluster that is only partly
     explained is re-fitted on its leftover voxels (two touching objects).

numpy + scipy (clusters) + torch (map lookups).
"""

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import ConvexHull

from .voxel_map import VoxelMap


@dataclass
class ObjectModel:
    name: str
    half: np.ndarray                  # bounding-box half extents in the rest frame
    surface: np.ndarray               # (S, 3) surface samples, rest frame centred on the bbox
    interior: np.ndarray              # (I, 3) interior samples
    yaw_period: float = 2 * math.pi   # pi for box-like objects, 2*pi in general; ~0 for cylinders

    @classmethod
    def from_hull(cls, name: str, hull_pts, n_surface: int = 500, seed: int = 0) -> "ObjectModel":
        pts = np.asarray(hull_pts, float)
        lo, hi = pts.min(0), pts.max(0)
        pts = pts - (lo + hi) / 2
        half = (hi - lo) / 2
        h = ConvexHull(pts)
        tris = pts[h.simplices]                                              # (F, 3, 3)
        areas = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
        rng = np.random.default_rng(seed)
        f = rng.choice(len(tris), size=n_surface, p=areas / areas.sum())
        r1, r2 = rng.random(n_surface), rng.random(n_surface)
        s = np.sqrt(r1)
        surf = (1 - s)[:, None] * tris[f, 0] + (s * (1 - r2))[:, None] * tris[f, 1] + (s * r2)[:, None] * tris[f, 2]
        interior = surf * 0.6                                                # shrunk toward the centre
        # rotational symmetry about z: a footprint close to a circle → yaw irrelevant
        r = np.linalg.norm(pts[:, :2], axis=1)
        if half[0] > 0 and abs(half[0] - half[1]) / max(half[0], half[1]) < 0.08 and r.std() / r.mean() < 0.08:
            period = 0.0
        elif abs(half[0] - half[1]) > 0.005:
            period = math.pi
        else:
            period = 2 * math.pi
        return cls(name, half, surf, interior, period)

    def transform(self, pts: np.ndarray, pos, yaw: float) -> np.ndarray:
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return pts @ R.T + np.asarray(pos, float)


@dataclass
class Fit:
    name: str
    pos: np.ndarray
    yaw: float
    score: float
    explained: float
    support: float
    conflict: float
    cluster_size: int


@dataclass
class Cluster:
    voxels: np.ndarray                    # (M, 3) voxel centres (world)
    idx: np.ndarray = field(repr=False)   # (M, 3) voxel indices


def static_mask(vmap: VoxelMap, boxes, pad: float) -> torch.Tensor:
    """Voxels inside any known static box (center, half, yaw), padded."""
    idx = torch.stack(torch.meshgrid(*[torch.arange(n, device=vmap.device) for n in vmap.shape], indexing="ij"), -1)
    c = vmap.centers(idx.view(-1, 3)).cpu().numpy()
    m = np.zeros(len(c), bool)
    for b in boxes:
        cy, sy = math.cos(b.yaw), math.sin(b.yaw)
        d = c - b.center
        lx = np.abs(cy * d[:, 0] + sy * d[:, 1]); ly = np.abs(-sy * d[:, 0] + cy * d[:, 1])
        m |= (lx <= b.half[0] + pad) & (ly <= b.half[1] + pad) & (np.abs(d[:, 2]) <= b.half[2] + pad)
    return torch.as_tensor(m.reshape(vmap.shape), device=vmap.device)


def boxes_raycast_fn(boxes):
    """Closest-hit distance along rays to upright boxes (center, half, yaw) → fn(o, d) → (R,), inf on miss."""
    C = torch.tensor(np.array([b.center for b in boxes]), dtype=torch.float32)
    H = torch.tensor(np.array([b.half for b in boxes]), dtype=torch.float32)
    Y = torch.tensor([b.yaw for b in boxes], dtype=torch.float32)

    def fn(o, d):
        o, d = torch.as_tensor(o, dtype=torch.float32), torch.as_tensor(d, dtype=torch.float32)
        c, s = torch.cos(Y).to(o.device), torch.sin(Y).to(o.device)
        a = o[:, None] - C.to(o.device)[None]                                 # (R, B, 3)
        v = d[:, None].expand_as(a)
        A = torch.stack([c * a[..., 0] + s * a[..., 1], -s * a[..., 0] + c * a[..., 1], a[..., 2]], -1)
        V = torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)
        h = H.to(o.device)[None]
        t1, t2 = (-h - A) / V, (h - A) / V
        par = V == 0
        ins = A.abs() <= h
        tmin = torch.where(par, torch.where(ins, -torch.inf, torch.inf), torch.minimum(t1, t2)).amax(-1)
        tmax = torch.where(par, torch.where(ins, torch.inf, -torch.inf), torch.maximum(t1, t2)).amin(-1)
        t = torch.where(tmin > 0, tmin, tmax)
        t = torch.where((tmin <= tmax) & (tmax > 0), t, torch.full_like(t, torch.inf))
        return t.amin(-1)
    return fn


def clusters(vmap: VoxelMap, static: torch.Tensor, occ_thresh: float = 0.5, min_voxels: int = 6,
             bridge: int = 1):
    """Connected components of occupied, non-static voxels. Components are found on the occupancy
    dilated by `bridge` voxels (ToF zones sample surfaces sparsely, leaving gaps inside one object),
    but each cluster keeps only its truly occupied voxels."""
    occ = (vmap.occupied(occ_thresh) & ~static).cpu().numpy()
    grown = ndimage.binary_dilation(occ, iterations=bridge) if bridge > 0 else occ
    lab, n = ndimage.label(grown, structure=np.ones((3, 3, 3)))
    lab = np.where(occ, lab, 0)
    out = []
    for k in range(1, n + 1):
        idx = np.argwhere(lab == k)
        if len(idx) >= min_voxels:
            out.append(Cluster(vmap.centers(torch.as_tensor(idx, device=vmap.device)).cpu().numpy(), idx))
    return out


class PoseFitter:
    """Places known objects into clusters. A cluster may hold several objects (ToF zones smear
    footprints, so neighbours often merge): the best (object, pose) over the whole cluster is
    placed, its voxels removed, and the rest re-fitted.

    Pose search: coarse grid over the cluster's extent (xy_coarse, n_yaw_coarse), then a local
    refinement (xy_step, n_yaw). The candidate's top is the top of the cluster voxels under its
    footprint (from above the sensors see tops; a box resting on something hidden lands at the
    right height); it never sinks below the floor.

    Score = support + coverage + w_explained * explained - w_conflict * conflict
      support   : fraction of the object's observed surface points on occupied voxels
      coverage  : cluster voxels under its footprint / footprint cells (capped at 1)
      explained : fraction of the cluster's voxels it accounts for, weighted strongly (one object
                  explaining a whole blob beats a smaller one that fits inside a partly-seen blob)
      conflict  : fraction of its interior points in confidently free voxels
    """

    def __init__(self, models: dict, floor_z: float, res: float, n_yaw: int = 24, xy_search: float = 0.03,
                 xy_step: float = 0.01, w_conflict: float = 2.0, min_score: float = 1.5,
                 xy_coarse: float = 0.02, n_yaw_coarse: int = 12, w_explained: float = 2.0,
                 min_voxels: int = 6):
        self.models = models
        self.floor_z, self.res = floor_z, res
        self.n_yaw, self.xy_search, self.xy_step = n_yaw, xy_search, xy_step
        self.w_conflict, self.min_score = w_conflict, min_score
        self.xy_coarse, self.n_yaw_coarse, self.w_explained = xy_coarse, n_yaw_coarse, w_explained
        self.min_voxels = min_voxels

    def _yaws(self, m: ObjectModel, n: int) -> np.ndarray:
        return np.array([0.0]) if m.yaw_period == 0.0 else np.arange(n) * m.yaw_period / n

    def _score(self, vmap, m: ObjectModel, xy_yaw: np.ndarray, vox: torch.Tensor):
        """xy_yaw (P, 3) candidates [x, y, yaw] against cluster voxels vox (M, 3) →
        (score, z, support, coverage, explained, conflict, inside (P, M))."""
        dev = vmap.device
        P = len(xy_yaw)
        if P == 0 or len(vox) == 0:
            e = torch.empty(0, device=dev)
            return e, e, e, e, e, e, torch.zeros(0, len(vox), dtype=torch.bool, device=dev)
        c = torch.as_tensor(np.cos(xy_yaw[:, 2]), dtype=torch.float32, device=dev)
        s = torch.as_tensor(np.sin(xy_yaw[:, 2]), dtype=torch.float32, device=dev)
        txy = torch.as_tensor(xy_yaw[:, :2], dtype=torch.float32, device=dev)
        h = torch.as_tensor(m.half, dtype=torch.float32, device=dev)
        d = vox[None, :, :2] - txy[:, None]                                     # (P, M, 2)
        lx = (c[:, None] * d[..., 0] + s[:, None] * d[..., 1]).abs()
        ly = (-s[:, None] * d[..., 0] + c[:, None] * d[..., 1]).abs()
        under = (lx <= h[0] + self.res) & (ly <= h[1] + self.res)                # footprint, any height
        # top = highest voxel layer under the footprint holding a real share of its voxels (>= 5%,
        # min 2). ToF hits smear slightly upward; a few stray voxels one layer up would otherwise
        # lift the object ~1 cm and put its top surface in free space.
        layers, layer_of = torch.unique(vox[:, 2], return_inverse=True)        # (Lz,), (M,)
        onehot = torch.nn.functional.one_hot(layer_of, len(layers)).float()     # (M, Lz)
        per_layer = under.float() @ onehot                                       # (P, Lz)
        n_under = under.sum(1, keepdim=True).float()
        valid = per_layer >= torch.maximum(0.05 * n_under, torch.full_like(n_under, 2.0))
        top = torch.where(valid, layers[None].expand_as(per_layer), torch.full_like(per_layer, -torch.inf)).amax(1)
        z = torch.maximum(top - h[2], torch.full_like(top, self.floor_z + float(m.half[2])))
        inside = under & ((vox[None, :, 2] - z[:, None]).abs() <= h[2] + self.res)
        n_in = inside.sum(1).float()
        coverage = (n_in / ((2 * h[0] * 2 * h[1]) / self.res ** 2)).clamp(max=1.0)
        explained = n_in / max(len(vox), 1)
        R = torch.zeros(P, 3, 3, device=dev)
        R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
        t = torch.stack([txy[:, 0], txy[:, 1], z], 1)
        surf = torch.as_tensor(m.surface, dtype=torch.float32, device=dev)
        inner = torch.as_tensor(m.interior, dtype=torch.float32, device=dev)
        Ls = vmap.lookup((torch.einsum("pij,sj->psi", R, surf) + t[:, None]).reshape(-1, 3)).view(P, -1)
        Li = vmap.lookup((torch.einsum("pij,sj->psi", R, inner) + t[:, None]).reshape(-1, 3)).view(P, -1)
        known = Ls.abs() > 0.3
        support = ((Ls > 0.5) & known).sum(1).float() / known.sum(1).clamp(min=1).float()
        conflict = (Li < -1.0).float().mean(1)
        score = support + coverage + self.w_explained * explained - self.w_conflict * conflict
        score = torch.where(n_in > 0, score, torch.full_like(score, -torch.inf))
        return score, z, support, coverage, explained, conflict, inside

    def best_pose(self, vmap, m: ObjectModel, vox: torch.Tensor):
        """Coarse-to-fine search over the cluster → (Fit-like tuple) or None."""
        if len(vox) == 0:
            return None
        v = vox.cpu().numpy()
        lo, hi = v[:, :2].min(0), v[:, :2].max(0)
        xs = np.arange(lo[0], hi[0] + 1e-9, self.xy_coarse)
        ys = np.arange(lo[1], hi[1] + 1e-9, self.xy_coarse)
        Y, X, YY = np.meshgrid(self._yaws(m, self.n_yaw_coarse), xs, ys, indexing="ij")
        cand = np.column_stack([X.ravel(), YY.ravel(), Y.ravel()])
        sc = self._score(vmap, m, cand, vox)[0].cpu().numpy()
        if not len(sc) or not np.isfinite(sc).any():
            return None
        x0 = cand[int(np.argmax(sc))]
        offs = np.arange(-self.xy_search, self.xy_search + 1e-9, self.xy_step)
        dyaw = m.yaw_period / self.n_yaw_coarse if m.yaw_period else 0.0
        yaws = x0[2] + np.linspace(-dyaw, dyaw, 9) if dyaw else np.array([0.0])     # ~4° steps
        Y, DX, DY = np.meshgrid(yaws, offs, offs, indexing="ij")
        fine = np.column_stack([x0[0] + DX.ravel(), x0[1] + DY.ravel(), Y.ravel()])
        score, z, sup, cov, ex, co, inside = self._score(vmap, m, fine, vox)
        if not len(score):
            return None
        i = int(torch.argmax(score))
        if not torch.isfinite(score[i]):
            return None
        pos = np.array([fine[i, 0], fine[i, 1], float(z[i])])
        return Fit(m.name, pos, float(fine[i, 2]), float(score[i]), float(ex[i]), float(sup[i]),
                   float(co[i]), len(vox)), inside[i]

    def fit(self, vmap, cls: list, names: list, max_objects_per_cluster: int = 4) -> tuple:
        """→ (fits {name: Fit}, unexplained clusters [voxels (M, 3)])."""
        from isaaclab_mpc.search.belief import Box, rects_overlap
        remaining = list(names)
        fits, placed, unexplained = {}, [], []
        for cl in sorted(cls, key=lambda c: -len(c.voxels)):
            vox = torch.as_tensor(cl.voxels, dtype=torch.float32, device=vmap.device)
            for _ in range(max_objects_per_cluster):
                if len(vox) < self.min_voxels or not remaining:
                    break
                options = []
                for name in remaining:
                    r = self.best_pose(vmap, self.models[name], vox)
                    if r is not None:
                        options.append(r)
                options.sort(key=lambda t: -t[0].score)
                chosen = None
                for f, inside in options:
                    if f.score < self.min_score:
                        break
                    box = Box(f.pos, self.models[f.name].half, f.yaw)
                    # Would interpenetrate an object already placed? Fits are off by 1-2 cm, so objects
                    # that nearly touch may appear to overlap by that much; only deeper overlap rejects.
                    if any(rects_overlap(box, b, margin=-2 * self.res) and abs(box.center[2] - b.center[2]) < box.half[2] + b.half[2]
                           for b in placed):
                        continue
                    chosen = (f, inside, box)
                    break
                if chosen is None:
                    break
                f, inside, box = chosen
                fits[f.name] = f
                placed.append(box)
                remaining.remove(f.name)
                vox = vox[~inside]
            if len(vox) >= self.min_voxels:
                unexplained.append(vox.cpu().numpy())
        return fits, unexplained
