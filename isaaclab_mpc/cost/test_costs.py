"""Equivalence tests: new cost modules vs. original inline formulas in planner.py."""
import torch
import pytest

from isaaclab_mpc.cost import (
    DistCost,
    OrientationCost,
    HeightMatchCost,
    PushAlignCost,
    ContactForceCost,
    JointVelCost,
    SingularityCost,
    GaussianProjection,
)
from isaaclab_mpc.cost.utils import quat_apply, quat_to_yaw_pitch

B = 16
torch.manual_seed(0)


def test_dist_cost():
    disp = torch.randn(B, 3)
    expected = torch.linalg.norm(disp, dim=1)
    assert torch.allclose(DistCost()(disp), expected)


def test_quat_apply():
    def _quat_apply_ref(q, v):
        w, x, y, z = q.unbind(-1)
        vx, vy, vz = v.unbind(-1)
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        return torch.stack([
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ], dim=-1)

    q = torch.randn(B, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    v = torch.randn(B, 3)
    assert torch.allclose(quat_apply(q, v), _quat_apply_ref(q, v))


def test_orientation_cost():
    # Identity quaternion → tool points +Z world → angle from -Z = π
    # atol=1e-3: acos near ±1 loses ~4e-4 precision due to the stability clamp
    q_id = torch.zeros(B, 4)
    q_id[:, 0] = 1.0  # w=1
    assert torch.allclose(OrientationCost()(q_id), torch.full((B,), torch.pi), atol=1e-3)

    # 180° around X: [w=0, x=1, y=0, z=0] → tool maps +Z → -Z → angle = 0
    q_down = torch.zeros(B, 4)
    q_down[:, 1] = 1.0  # x=1
    assert torch.allclose(OrientationCost()(q_down), torch.zeros(B), atol=1e-3)

    # Verify formula for random unit quaternions
    q = torch.randn(B, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    tool_local = torch.zeros(B, 3)
    tool_local[:, 2] = 1.0
    tool_world = quat_apply(q, tool_local)
    expected = torch.acos((-tool_world[:, 2]).clamp(-1 + 1e-7, 1 - 1e-7))
    assert torch.allclose(OrientationCost()(q), expected)


def test_height_match_cost():
    tcp_z = torch.randn(B)
    obj_z = torch.randn(B)
    expected = torch.abs(tcp_z - obj_z)
    assert torch.allclose(HeightMatchCost()(tcp_z, obj_z), expected)


def test_push_align_cost():
    align_gate_dist = 0.036  # obj_size/2 + 0.01 for obj_size=0.05
    gate_width = 0.03

    robot_to_obj = torch.randn(B, 3)
    obj_to_goal  = torch.randn(B, 3)
    dist = torch.linalg.norm(robot_to_obj, dim=1)

    r2b_2d = robot_to_obj[:, :2]
    b2g_2d = obj_to_goal[:, :2]
    ref = (
        torch.sum(r2b_2d * b2g_2d, dim=1)
        / (torch.linalg.norm(r2b_2d, dim=1).clamp(min=1e-6)
           * torch.linalg.norm(b2g_2d, dim=1).clamp(min=1e-6))
        + 1.0
    )
    gate = torch.sigmoid((dist - align_gate_dist) / gate_width)
    ref = ref * gate

    cost = PushAlignCost(align_gate_dist=align_gate_dist, gate_width=gate_width)
    assert torch.allclose(cost(robot_to_obj, obj_to_goal, dist), ref)


def test_contact_force_cost():
    forces = torch.randn(B, 1, 3)
    expected = torch.abs(forces[:, 0, 2])
    assert torch.allclose(ContactForceCost()(forces), expected)


def test_joint_vel_cost():
    joint_vel = torch.randn(B, 6)
    expected = torch.linalg.norm(joint_vel, dim=1)
    assert torch.allclose(JointVelCost()(joint_vel), expected)


def test_singularity_cost():
    J = torch.randn(B, 6, 6)
    expected = 1.0 / (torch.abs(torch.linalg.det(J)) + 1e-6)
    assert torch.allclose(SingularityCost()(J), expected)


# ---------------------------------------------------------------------------
# GaussianProjection
# ---------------------------------------------------------------------------

def test_gaussian_projection_passthrough():
    """c=0 must return the input unchanged (identity)."""
    x = torch.randn(B)
    proj = GaussianProjection(n=1, c=0.0, s=0.0, r=0.0)
    assert torch.allclose(proj(x), x)


def test_gaussian_projection_formula():
    """Non-zero c must match the reference formula exactly."""
    n, c, s, r = 1, 0.5, 0.1, 1e-5
    x = torch.randn(B).abs()  # positive distances
    xp = x - s
    expected = 1.0 - ((-1.0) ** n) * torch.exp(-xp ** 2 / (2.0 * c ** 2)) + r * xp ** 4
    assert torch.allclose(GaussianProjection(n=n, c=c, s=s, r=r)(x), expected)


def test_gaussian_projection_zero_at_centre():
    """With n=2, s=0, r=0: f(0) = 1 - exp(0) = 0 (distance cost: zero error → zero cost)."""
    proj = GaussianProjection(n=2, c=0.5, s=0.0, r=0.0)
    x = torch.zeros(B)
    assert torch.allclose(proj(x), torch.zeros(B))


def test_gaussian_projection_saturates_at_infinity():
    """With n=2, r=0: f(x) → 1 for large x (distance cost saturates to 1)."""
    proj = GaussianProjection(n=2, c=0.5, s=0.0, r=0.0)
    x = torch.full((B,), 100.0)
    assert torch.allclose(proj(x), torch.ones(B), atol=1e-4)
