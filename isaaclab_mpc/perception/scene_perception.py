"""Scene perception without privileged poses: map → clusters → known-object poses → tracks.

    perc = ScenePerception(lo, hi, static_boxes, models, floor_z)
    perc.add_tof(origins, dirs, dists, zone_angle)        # every ToF frame
    perc.add_depth(origins, dirs, dists)                  # every (subsampled) depth frame
    perc.update()                                         # every few steps: segment + fit + track
    perc.poses()  → {name: (pos (3,), yaw)} of the objects currently believed placed

Tracks: a fresh fit replaces an object's pose. An object not re-fitted keeps its last pose
unless its fitted volume has since been observed free (it was pushed away), which drops it.
"""

import numpy as np
import torch

from .objects import ObjectModel, PoseFitter, boxes_raycast_fn, clusters, static_mask
from .voxel_map import VoxelMap


class ScenePerception:
    def __init__(self, lo, hi, static_boxes, models: dict, floor_z: float, res: float = 0.01,
                 device="cpu", p: dict | None = None):
        p = p or {}
        self.p = p
        self.map = VoxelMap(lo, hi, res, device)
        self.static = static_mask(self.map, static_boxes, pad=p.get("static_pad", 1.5 * res))
        self.map.set_static(boxes_raycast_fn(static_boxes), tol=p.get("static_tol", 0.015))
        self.models = models
        self.fitter = PoseFitter(models, floor_z, res, n_yaw=p.get("n_yaw", 24),
                                 xy_search=p.get("xy_search", 0.03), xy_step=p.get("xy_step", 0.01),
                                 w_conflict=p.get("w_conflict", 2.0), min_score=p.get("min_score", 1.5),
                                 min_voxels=p.get("min_voxels", 6))
        self.tracks: dict = {}          # name → Fit
        self.unexplained: list = []     # voxel arrays of clusters no known object explains
        self.max_range = p.get("max_range", 1.0)

    def add_tof(self, origins, dirs, dists, zone_angle: float, k: int | None = None):
        self.map.integrate_tof_zones(origins, dirs, dists, zone_angle, k=k or self.p.get("tof_subrays", 3),
                                     max_range=self.max_range)

    def add_depth(self, origins, dirs, dists, occ_weight=None):
        """Depth-camera rays; occ_weight 0 marks a range limit (no return within range): carve only."""
        self.map.integrate_rays(origins, dirs, dists, occ_weight=occ_weight, max_range=self.max_range)

    def update(self):
        cls = clusters(self.map, self.static, occ_thresh=self.p.get("occ_thresh", 0.5),
                       min_voxels=self.p.get("min_voxels", 6))
        fits, self.unexplained = self.fitter.fit(self.map, cls, list(self.models))
        drop_frac = self.p.get("drop_conflict", 0.3)
        for name, f in fits.items():
            self.tracks[name] = f
        for name in [n for n in self.tracks if n not in fits]:
            f, m = self.tracks[name], self.models[name]
            inner = torch.as_tensor(m.transform(m.interior, f.pos, f.yaw), dtype=torch.float32,
                                    device=self.map.device)
            if float((self.map.lookup(inner) < -1.0).float().mean()) > drop_frac:
                del self.tracks[name]                 # its volume is now seen free: it moved
        return fits

    def poses(self) -> dict:
        return {n: (f.pos.copy(), f.yaw) for n, f in self.tracks.items()}


def models_from_catalogue(names) -> dict:
    """ObjectModel per YCB short name, from the hull in assets/ycb/objects.json."""
    from assets.ycb import get_spec
    return {n: ObjectModel.from_hull(n, get_spec(n).hull) for n in names}
