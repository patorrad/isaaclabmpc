"""Wrist depth camera (RealSense D405-like) for Isaac Lab, and conversion of its images to rays.

Defaults follow the D405: 87 x 58 deg field of view, 7–50 cm usable range. Depth noise is a
simple model, sigma(z) = sigma0 + k * z^2 (m), applied per pixel.

depth_rays() turns one image into world-frame rays (origins, dirs, ranges) for the voxel map:
pixels with no valid depth become "no return within range" (carve to `far`, no hit), and pixels
whose hit lies on the robot's own tool (a capsule given by two points and a radius) are dropped,
so the tool doesn't enter the map as an object. The real camera needs the same filtering, with
the tool pose from FK.
"""

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

D405 = dict(hfov_deg=87.0, vfov_deg=58.0, near=0.07, far=0.50, sigma0=0.0005, sigma_k=0.004)


def make_depth_camera_cfg(name: str = "WristCam", width: int = 160, height: int = 96,
                          hfov_deg: float = D405["hfov_deg"]) -> CameraCfg:
    """Depth camera spawned at the env level; move it each step with set_camera_pose().

    A camera parented to a robot link doesn't follow the link's simulated motion here (its pose
    stays at the spawn pose), so the pose is set explicitly from the link's physics pose and the
    mount offset — the same way the real camera's pose comes from FK."""
    focal = 1.0
    h_ap = 2 * focal * math.tan(math.radians(hfov_deg) / 2)
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=h_ap,
                                         vertical_aperture=h_ap * height / width, clipping_range=(0.01, 2.0)),
        width=width, height=height, update_period=0.0,
        update_latest_camera_pose=True,     # the camera moves with the wrist; pose needed for its rays
    )


def set_camera_pose(camera, link_pos_w: torch.Tensor, link_quat_w: torch.Tensor, offset_pos, offset_rot):
    """Place the camera at link pose ⊕ mount offset (offset in the link frame, ROS convention:
    +Z forward, so offset_rot = identity looks along the link's +Z). Shapes: (N, 3), (N, 4)."""
    from isaaclab.utils.math import quat_apply, quat_mul
    op = torch.as_tensor(offset_pos, dtype=torch.float32, device=link_pos_w.device).expand_as(link_pos_w)
    oq = torch.as_tensor(offset_rot, dtype=torch.float32, device=link_pos_w.device).expand_as(link_quat_w)
    camera.set_world_poses(link_pos_w + quat_apply(link_quat_w, op), quat_mul(link_quat_w, oq), convention="ros")


def _point_segment_dist(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = b - a
    t = ((p - a) @ ab / ab.dot(ab).clamp(min=1e-12)).clamp(0.0, 1.0)
    return (p - (a + t[:, None] * ab)).norm(dim=1)


def depth_rays(camera, env: int = 0, stride: int = 2, near: float = D405["near"], far: float = D405["far"],
               sigma0: float = D405["sigma0"], sigma_k: float = D405["sigma_k"], tool=None, noise: bool = True,
               generator: torch.Generator | None = None):
    """(origins (R, 3), dirs (R, 3), ranges (R,), occ_weight (R,)) in the world frame.

    tool: optional (a (3,), b (3,), radius) capsule in the world frame; hits on it are dropped.
    Pixels without valid depth carve to `far` with zero occupancy weight.
    """
    data = camera.data
    depth = data.output["distance_to_image_plane"][env, ::stride, ::stride, 0]          # (h, w) along +Z
    K = data.intrinsic_matrices[env]
    H, W = data.output["distance_to_image_plane"].shape[1:3]
    v, u = torch.meshgrid(torch.arange(0, H, stride, device=depth.device, dtype=torch.float32),
                          torch.arange(0, W, stride, device=depth.device, dtype=torch.float32), indexing="ij")
    x = (u + 0.5 - K[0, 2]) / K[0, 0]
    y = (v + 0.5 - K[1, 2]) / K[1, 1]
    d_cam = torch.stack([x, y, torch.ones_like(x)], -1).reshape(-1, 3)
    z = depth.reshape(-1).clone()
    valid = torch.isfinite(z) & (z >= near) & (z <= far)
    if noise:
        z = z + torch.randn(z.shape, device=z.device, generator=generator) * (sigma0 + sigma_k * z.clamp(max=far) ** 2)
    rng = z * d_cam.norm(dim=1)                                                        # along the ray
    dirs_c = d_cam / d_cam.norm(dim=1, keepdim=True)
    from isaaclab.utils.math import quat_apply
    q = data.quat_w_ros[env].expand(len(dirs_c), 4)
    dirs = quat_apply(q, dirs_c)
    origins = data.pos_w[env].expand(len(dirs), 3).clone()
    far_rng = far * d_cam.norm(dim=1)
    ranges = torch.where(valid, rng, far_rng)
    occ = valid.float()
    keep = torch.ones_like(valid)
    if tool is not None:
        a, b, r = tool
        hits = origins + dirs * ranges[:, None]
        on_tool = valid & (_point_segment_dist(hits, a, b) < r)
        keep &= ~on_tool
    return origins[keep], dirs[keep], ranges[keep], occ[keep]
