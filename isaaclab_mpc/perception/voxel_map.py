"""Log-odds occupancy voxel map fed by ToF zones and depth-camera rays (torch, GPU-friendly).

Sensor models
-------------
Depth camera : one exact ray per (subsampled) pixel. Voxels along the ray before the hit are
               free; the voxel at the hit is occupied.
ToF zone     : the VL53L5CX reports the *closest* return inside each zone's cone. So everything
               in the cone up to that distance is free (strong evidence), but where in the zone
               the hit was is unknown. A zone is expanded into k x k sub-rays that all carve free
               space up to its distance; each sub-ray endpoint gets 1/k of a full hit (spreading it
               as 1/k^2 leaves small objects too sparse to segment).
No return    : carve free space up to `max_range` (clipped to the map).

Known static structure (the tote, optional): a zone that reads about the distance predicted from
the structure is explained by it, and adds no occupied evidence. Otherwise the closest-return
semantics would put phantom hits above an obliquely-seen floor (all sub-rays at the zone's nearest
floor distance). A zone reading clearly shorter than predicted holds something unknown; its hit
goes only on the sub-rays where the structure is farther than the reading.

Carving is what lets the map forget an object that has been pushed away. The converse is
occlusion aging: the space just behind a hit is now hidden, so its log-odds decay toward unknown
(`occlusion_decay` over `occlusion_depth`). Without it, space carved free before an object was
pushed there stays "free" under the object forever and contradicts its pose.
"""

import math

import torch


class VoxelMap:
    def __init__(self, lo, hi, res: float = 0.01, device="cpu",
                 l_free: float = -0.4, l_occ: float = 0.85, l_min: float = -2.0, l_max: float = 2.5,
                 occlusion_depth: float = 0.10, occlusion_decay: float = 0.5):
        # Tight clamps keep the map responsive: objects get pushed, and space carved free must be
        # able to turn occupied again within a few frames (and vice versa).
        self.res = float(res)
        self.device = device
        self.lo = torch.as_tensor(lo, dtype=torch.float32, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.float32, device=device)
        self.shape = tuple(int(math.ceil(float(v))) for v in ((self.hi - self.lo) / self.res))
        self.L = torch.zeros(self.shape, dtype=torch.float32, device=device)
        self.l_free, self.l_occ, self.l_min, self.l_max = l_free, l_occ, l_min, l_max
        self.occlusion_depth, self.occlusion_decay = occlusion_depth, occlusion_decay

    # ── geometry ────────────────────────────────────────────────────────────
    def to_index(self, p: torch.Tensor) -> torch.Tensor:
        return torch.floor((p - self.lo) / self.res).long()

    def centers(self, idx: torch.Tensor) -> torch.Tensor:
        return self.lo + (idx.float() + 0.5) * self.res

    def _inside(self, idx: torch.Tensor) -> torch.Tensor:
        s = torch.tensor(self.shape, device=idx.device)
        return ((idx >= 0) & (idx < s)).all(dim=-1)

    def _add(self, idx: torch.Tensor, val: torch.Tensor):
        ok = self._inside(idx)
        idx, val = idx[ok], val[ok]
        if not len(idx):
            return
        flat = (idx[:, 0] * self.shape[1] + idx[:, 1]) * self.shape[2] + idx[:, 2]
        L = self.L.view(-1)
        L.index_add_(0, flat, val)
        L.clamp_(self.l_min, self.l_max)

    # ── integration ─────────────────────────────────────────────────────────
    def set_static(self, raycast_fn, tol: float = 0.015):
        """raycast_fn(origins (R, 3), dirs (R, 3)) → (R,) distance to the known static structure."""
        self.static_raycast, self.static_tol = raycast_fn, tol

    def integrate_rays(self, origins, dirs, dists, occ_weight=None, max_range: float = 1.0):
        """origins, dirs: (R, 3); dists: (R,) with inf = no return; occ_weight: (R,) scale on l_occ."""
        origins = torch.as_tensor(origins, dtype=torch.float32, device=self.device)
        dirs = torch.as_tensor(dirs, dtype=torch.float32, device=self.device)
        dists = torch.as_tensor(dists, dtype=torch.float32, device=self.device)
        if not len(dists):
            return
        hit = torch.isfinite(dists)
        free_len = torch.where(hit, dists - self.res, torch.full_like(dists, max_range))
        free_len = free_len.clamp(min=0.0, max=max_range)
        # Free space: samples every res/2 along each ray up to free_len. One update per voxel per ray.
        n = int(math.ceil(float(free_len.max()) / (self.res / 2))) + 1
        t = torch.arange(n, device=self.device, dtype=torch.float32) * (self.res / 2)       # (S,)
        pts = origins[:, None] + dirs[:, None] * t[None, :, None]                            # (R, S, 3)
        valid = t[None] <= free_len[:, None]
        idx = self.to_index(pts)                                                             # (R, S, 3)
        idx = torch.where(valid[..., None], idx, torch.full_like(idx, -1))
        # dedupe consecutive repeats of the same voxel along a ray
        same = torch.zeros_like(valid)
        same[:, 1:] = (idx[:, 1:] == idx[:, :-1]).all(-1)
        keep = valid & ~same
        free_idx = idx[keep]
        self._add(free_idx, torch.full((len(free_idx),), self.l_free, device=self.device))
        # Occupied at hits
        if hit.any():
            w = torch.ones_like(dists) if occ_weight is None else torch.as_tensor(
                occ_weight, dtype=torch.float32, device=self.device)
            end = origins[hit] + dirs[hit] * dists[hit, None]
            self._add(self.to_index(end), self.l_occ * w[hit])
            real = hit & (w > 0)                     # zero-weight "hits" are range limits, not surfaces
            self._age_behind(origins[real], dirs[real], dists[real])

    def _age_behind(self, origins, dirs, dists):
        """Decay the log-odds of voxels within occlusion_depth behind each hit toward 0 (unknown)."""
        if self.occlusion_depth <= 0 or self.occlusion_decay >= 1.0:
            return
        n = int(math.ceil(self.occlusion_depth / self.res))
        t = (torch.arange(1, n + 1, device=self.device, dtype=torch.float32) + 0.5) * self.res
        pts = origins[:, None] + dirs[:, None] * (dists[:, None, None] + t[None, :, None])
        idx = self.to_index(pts.reshape(-1, 3))
        idx = idx[self._inside(idx)]
        if len(idx):
            idx = torch.unique(idx, dim=0)
            self.L[idx[:, 0], idx[:, 1], idx[:, 2]] *= self.occlusion_decay

    def integrate_tof_zones(self, origins, dirs_center, dists, zone_angle: float, k: int = 3,
                            max_range: float = 1.0):
        """ToF zones: origins (Z, 3), zone-centre dirs (Z, 3), dists (Z,), zone angular size (rad).

        Each zone becomes k x k sub-rays spread across its square cone; all carve free space up to
        the zone distance, and each sub-ray endpoint gets 1/k of a full hit.
        """
        origins = torch.as_tensor(origins, dtype=torch.float32, device=self.device)
        dc = torch.as_tensor(dirs_center, dtype=torch.float32, device=self.device)
        dists = torch.as_tensor(dists, dtype=torch.float32, device=self.device)
        # Orthonormal frame per zone around its centre direction.
        ref = torch.where(dc[:, 2:3].abs() < 0.9, torch.tensor([0.0, 0.0, 1.0], device=self.device),
                          torch.tensor([1.0, 0.0, 0.0], device=self.device))
        u = torch.linalg.cross(dc, ref); u = u / u.norm(dim=1, keepdim=True)
        v = torch.linalg.cross(dc, u)
        off = (torch.arange(k, device=self.device, dtype=torch.float32) + 0.5) / k - 0.5    # (k,)
        a, b = torch.meshgrid(off, off, indexing="ij")
        a, b = (a.reshape(-1) * zone_angle), (b.reshape(-1) * zone_angle)                    # (k*k,)
        d = dc[:, None] + torch.tan(a)[None, :, None] * u[:, None] + torch.tan(b)[None, :, None] * v[:, None]
        d = d / d.norm(dim=-1, keepdim=True)                                                  # (Z, k*k, 3)
        kk = k * k
        o = origins[:, None].expand(-1, kk, 3).reshape(-1, 3)
        d = d.reshape(-1, 3)
        r = dists[:, None].expand(-1, kk).reshape(-1)
        w = torch.full((len(r),), 1.0 / k, device=self.device)
        if getattr(self, "static_raycast", None) is not None:
            t_static = torch.as_tensor(self.static_raycast(o, d), dtype=torch.float32, device=self.device).view(-1, kk)
            explained = dists >= t_static.min(dim=1).values - self.static_tol                 # (Z,)
            consistent = t_static > dists[:, None] + self.static_tol                          # (Z, kk)
            w = torch.where(explained[:, None] | ~consistent, torch.zeros_like(t_static), w.view(-1, kk)).reshape(-1)
        self.integrate_rays(o, d, r, occ_weight=w, max_range=max_range)

    # ── queries ─────────────────────────────────────────────────────────────
    def occupied(self, thresh: float = 0.5) -> torch.Tensor:
        return self.L > thresh

    def free(self, thresh: float = -1.0) -> torch.Tensor:
        return self.L < thresh

    def lookup(self, p: torch.Tensor) -> torch.Tensor:
        """Log-odds at world points (N, 3); 0 (unknown) outside the map."""
        idx = self.to_index(p)
        ok = self._inside(idx)
        out = torch.zeros(len(p), device=self.device)
        i = idx[ok]
        out[ok] = self.L[i[:, 0], i[:, 1], i[:, 2]]
        return out
