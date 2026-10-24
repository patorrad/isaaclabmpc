"""Multizone time-of-flight sensors (ST VL53L5CX) built on Isaac Lab's MultiMeshRayCaster.

The VL53L5CX reports one distance per zone over an 8x8 grid spanning a 45 x 45 deg
square field of view (63 deg diagonal), up to 4 m. Each zone is modelled with
`rays_per_zone` x `rays_per_zone` rays spread evenly inside the zone; the zone
reads the closest hit, like the sensor's default "closest target first" output.
Distances are along the ray (time of flight), not depth along the boresight.

Sensor frame: the boresight is +Z, zone rows run along +Y and columns along +X,
so frames come out as (rows, cols) = (8, 8) in row-major order.

MultiMeshRayCaster can track moving meshes, so the rays see the tracked rigid
objects as well as the static scene. The robot itself is not a ray target.
"""

import math
from collections.abc import Callable
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils.math import quat_apply
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass

# VL53L5CX datasheet values.
VL53L5CX_ZONES = (8, 8)
VL53L5CX_FOV_DEG = (45.0, 45.0)
VL53L5CX_MIN_RANGE = 0.02
VL53L5CX_MAX_RANGE = 4.0


def tof_zone_pattern(cfg: "ToFPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Rays for a zoned ToF sensor, ordered zone by zone (row-major), sub-rays within a zone.

    Returns (ray_starts, ray_directions), each (rows * cols * k * k, 3), in the sensor frame.
    """
    rows, cols = cfg.zones
    k = cfg.rays_per_zone
    fov_x, fov_y = (math.radians(a) for a in cfg.fov_deg)
    # Angular centres of the k x k sub-rays inside each zone, across the whole FoV.
    sub = (torch.arange(k, device=device) + 0.5) / k                       # (k,)
    ax = (torch.arange(cols, device=device)[:, None] + sub[None]) / cols * fov_x - fov_x / 2   # (cols, k)
    ay = (torch.arange(rows, device=device)[:, None] + sub[None]) / rows * fov_y - fov_y / 2   # (rows, k)
    # Index order: row, col, sub-row, sub-col → zone-major, so zone z owns rays [z*k*k, (z+1)*k*k).
    AY = ay[:, None, :, None].expand(rows, cols, k, k)
    AX = ax[None, :, None, :].expand(rows, cols, k, k)
    dirs = torch.stack([torch.tan(AX), torch.tan(AY), torch.ones_like(AX)], dim=-1).reshape(-1, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    return torch.zeros_like(dirs), dirs


@configclass
class ToFPatternCfg(PatternBaseCfg):
    """Zoned ToF ray pattern (defaults: VL53L5CX in 8x8 mode)."""

    func: Callable = tof_zone_pattern
    zones: tuple[int, int] = VL53L5CX_ZONES
    """(rows, cols) of zones."""
    fov_deg: tuple[float, float] = VL53L5CX_FOV_DEG
    """Field of view (x, y) in degrees."""
    rays_per_zone: int = 2
    """Rays per zone side; each zone casts rays_per_zone ** 2 rays."""


def make_tof_cfg(
    prim_path: str,
    static_targets: list[str],
    moving_targets: list[str] = (),
    pos=(0.0, 0.0, 0.0),
    rot=(1.0, 0.0, 0.0, 0.0),
    rays_per_zone: int = 2,
    max_range: float = VL53L5CX_MAX_RANGE,
    debug_vis: bool = False,
    shared_meshes: bool = True,
    hit_radius: float = 0.02,
) -> MultiMeshRayCasterCfg:
    """MultiMeshRayCasterCfg for one VL53L5CX.

    prim_path      : link the sensor is fixed to, e.g. "{ENV_REGEX_NS}/Robot/wrist_3_link".
    static_targets : prims the rays can hit that never move (regex, {ENV_REGEX_NS} allowed).
    moving_targets : prims that move (e.g. rigid objects); their transforms are tracked every update.
    pos, rot       : sensor pose in the link frame; rot is (w, x, y, z) and the boresight is +Z.
    shared_meshes  : every env has the same geometry under each target path (true for cloned scenes),
                     so env_0's meshes are parsed once and reused. Without it, startup parses every
                     env's meshes and compares them pairwise (~2 min for 750 envs). Poses of moving
                     targets are still tracked per env.
    """
    return MultiMeshRayCasterCfg(
        prim_path=prim_path,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=tuple(pos), rot=tuple(rot)),
        ray_alignment="base",
        pattern_cfg=ToFPatternCfg(rays_per_zone=rays_per_zone),
        max_distance=max_range,
        mesh_prim_paths=(
            [MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=e, track_mesh_transforms=False,
                                                    is_shared=shared_meshes)
             for e in static_targets]
            + [MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=e, track_mesh_transforms=True,
                                                      is_shared=shared_meshes)
               for e in moving_targets]
        ),
        update_period=0.0,
        debug_vis=debug_vis,
        # debug_vis draws a sphere at every ray hit; Isaac Lab's default radius is 2 cm.
        visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(
            prim_path="/Visuals/RayCaster",
            markers={"hit": sim_utils.SphereCfg(
                radius=hit_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))},
        ),
    )


def zone_distances(sensor, zones: tuple[int, int] = VL53L5CX_ZONES, rays_per_zone: int = 2,
                   min_range: float = VL53L5CX_MIN_RANGE,
                   max_range: float = VL53L5CX_MAX_RANGE) -> torch.Tensor:
    """(num_envs, rows, cols) distance per zone in metres: the closest hit among the zone's rays.

    Zones with no target in [min_range, max_range] read +inf (the real sensor flags them invalid).

    Isaac Lab bakes cfg.offset into the ray pattern, so data.pos_w / quat_w are the parent
    link's pose; the rays start at pos_w + quat_w * offset.pos, which is what we measure from.
    """
    data = sensor.data
    offset = torch.tensor(sensor.cfg.offset.pos, device=data.pos_w.device).expand_as(data.pos_w)
    origin = data.pos_w + quat_apply(data.quat_w, offset)                      # (N, 3)
    d = torch.linalg.norm(data.ray_hits_w - origin[:, None, :], dim=-1)       # (N, B)
    d = torch.where(torch.isfinite(d) & (d >= min_range) & (d <= max_range), d,
                    torch.full_like(d, math.inf))
    rows, cols = zones
    return d.view(d.shape[0], rows, cols, rays_per_zone ** 2).min(dim=-1).values


def zone_centre_dirs(zones: tuple[int, int] = VL53L5CX_ZONES,
                     fov_deg: tuple[float, float] = VL53L5CX_FOV_DEG, device="cpu") -> torch.Tensor:
    """(rows * cols, 3) unit zone-centre directions in the sensor frame (boresight +Z), row-major."""
    _, d = tof_zone_pattern(ToFPatternCfg(zones=zones, fov_deg=fov_deg, rays_per_zone=1), device)
    return d


def zone_rays(origin_w: torch.Tensor, quat_w: torch.Tensor, zones_dist: torch.Tensor,
              fov_deg: tuple[float, float] = VL53L5CX_FOV_DEG):
    """World-frame zone-centre rays for one sensor: (origins (Z, 3), dirs (Z, 3), dists (Z,)).

    origin_w, quat_w : sensor pose in the world (w, x, y, z); for the real VL53L5CX this comes from FK
    zones_dist       : (rows, cols) zone distances (m), inf = no target
    """
    rows, cols = zones_dist.shape
    d_local = zone_centre_dirs((rows, cols), fov_deg, zones_dist.device)
    dirs = quat_apply(quat_w.expand(len(d_local), 4), d_local)
    return origin_w.expand(len(d_local), 3).clone(), dirs, zones_dist.reshape(-1)


def zone_rays_from_sensor(sensor, env: int = 0, rays_per_zone: int = 2):
    """zone_rays for env `env` of a simulated ToF (MultiMeshRayCaster from make_tof_cfg).

    data.pos_w / quat_w are the parent link's pose; cfg.offset is baked into the pattern, so the
    sensor pose is the link pose composed with the offset.
    """
    from isaaclab.utils.math import quat_mul
    data = sensor.data
    off_p = torch.tensor(sensor.cfg.offset.pos, device=data.pos_w.device)
    off_q = torch.tensor(sensor.cfg.offset.rot, device=data.pos_w.device)
    origin = data.pos_w[env] + quat_apply(data.quat_w[env], off_p)
    quat = quat_mul(data.quat_w[env], off_q)
    dist = zone_distances(sensor, rays_per_zone=rays_per_zone)[env]
    return zone_rays(origin, quat, dist, sensor.cfg.pattern_cfg.fov_deg)
