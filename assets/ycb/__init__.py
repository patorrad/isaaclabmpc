"""YCB objects for Isaac Lab scenes (same catalogue and USD wrapper as the puzzle repo).

objects.json is copied from puzzle/assets/ycb/objects.json, which
puzzle/scripts/prepare_ycb.py builds. Per object it holds the asset URL, the
published mass, the resting orientation (most probable stable pose, footprint
axis-aligned) and the convex hull in that rest frame. The rest frame is centred
on the bounding box with z up, so an object rests on its bottom face at
z = -height/2.

At runtime Kit streams the textured meshes from NVIDIA's asset server (needs
internet the first time; Kit caches them). Each object collides as its convex
hull, so concave objects such as the mug or bowl can't nest.

Catalogue lookups are pure Python; make_ycb_object_cfg needs a running
SimulationApp (it writes USD with pxr and imports isaaclab).
"""

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache

CATALOGUE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objects.json")
USD_DIR = os.path.join(os.path.expanduser("~"), ".cache", "isaaclabmpc", "ycb")

# Extra short names on top of the id without its number ('003_cracker_box' -> 'cracker_box').
_ALIASES = {
    "soup_can": "tomato_soup_can", "tuna_can": "tuna_fish_can", "meat_can": "potted_meat_can",
    "chef_can": "master_chef_can", "mustard": "mustard_bottle", "bleach": "bleach_cleanser",
    "pitcher": "pitcher_base", "drill": "power_drill", "marker": "large_marker",
    "clamp": "large_clamp", "xl_clamp": "extra_large_clamp",
}


@dataclass(frozen=True)
class YcbSpec:
    ycb_id: str
    url: str
    mass: float
    size: tuple          # rest-pose bounding box [x, y, z] (m)
    rest_quat: tuple     # (w, x, y, z) applied to the YCB mesh, then offset
    offset: tuple
    hull: tuple          # convex hull vertices in the rest frame

    @property
    def name(self) -> str:
        return short_name(self.ycb_id)

    @property
    def height(self) -> float:
        return float(self.size[2])


def short_name(ycb_id: str) -> str:
    """'003_cracker_box' -> 'cracker_box'."""
    head, _, rest = ycb_id.partition("_")
    return rest if head.isdigit() else ycb_id


@lru_cache(maxsize=1)
def load_catalogue(path: str = CATALOGUE_PATH) -> dict:
    """{short name: YcbSpec} for every object in the catalogue."""
    with open(path) as f:
        entries = json.load(f)["objects"]
    return {
        short_name(e["id"]): YcbSpec(
            ycb_id=e["id"], url=e["url"], mass=float(e["mass"]), size=tuple(e["size"]),
            rest_quat=tuple(e["rest_quat"]), offset=tuple(e["offset"]),
            hull=tuple(tuple(p) for p in e["hull"]),
        )
        for e in entries
    }


def get_spec(name: str) -> YcbSpec:
    """Look up by short name ('cracker_box'), alias ('soup_can') or id ('003_cracker_box')."""
    cat = load_catalogue()
    key = _ALIASES.get(name, short_name(name))
    if key not in cat:
        raise KeyError(f"unknown YCB object {name!r}; available: {', '.join(sorted(cat))} "
                       f"(aliases: {', '.join(sorted(_ALIASES))})")
    return cat[key]


def yaw_quat(yaw_rad: float) -> tuple:
    """(w, x, y, z) rotation about +Z."""
    return (math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2))


def write_ycb_usd(spec: YcbSpec, path: str, friction: float) -> None:
    """Write a USDA wrapper: rigid body with the real mass, the remote textured mesh
    as visual (rotated into the rest pose and centred) and the convex hull as collider.
    Written to a temp file and renamed, so a planner and a world starting together
    never read a half-written file.
    """
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    from scipy.spatial import ConvexHull

    tmp = f"{path}.{os.getpid()}.tmp.usda"
    stage = Usd.Stage.CreateNew(tmp)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(spec.mass)

    mat = UsdShade.Material.Define(stage, "/Object/PhysicsMaterial")
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys_mat.CreateStaticFrictionAttr(friction)
    phys_mat.CreateDynamicFrictionAttr(friction)
    phys_mat.CreateRestitutionAttr(0.0)

    visual = UsdGeom.Xform.Define(stage, "/Object/visual")
    visual.AddTranslateOp().Set(Gf.Vec3d(*spec.offset))
    w, x, y, z = spec.rest_quat
    visual.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
    visual.GetPrim().GetReferences().AddReference(spec.url)

    hull = spec.hull
    faces = ConvexHull(hull).simplices
    col = UsdGeom.Mesh.Define(stage, "/Object/collision")
    col.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in hull]))
    col.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    col.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in faces.flatten()]))
    col.CreatePurposeAttr(UsdGeom.Tokens.guide)          # collider only, never rendered
    prim = col.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.strongerThanDescendants, "physics")
    stage.Save()
    os.replace(tmp, path)


def make_ycb_object_cfg(name: str, pos, yaw_rad: float = 0.0, friction: float = 0.5,
                        disable_gravity: bool = False, linear_damping: float = 0.5):
    """RigidObjectCfg for one YCB object with its bounding-box centre at `pos`."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    spec = get_spec(name)
    os.makedirs(USD_DIR, exist_ok=True)
    usd_path = os.path.join(USD_DIR, f"{spec.ycb_id}_mu{friction:g}.usda")
    write_ycb_usd(spec, usd_path, friction)
    return RigidObjectCfg(
        prim_path="PLACEHOLDER",  # replaced by IsaacLabWrapper
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=disable_gravity,
                linear_damping=linear_damping,
                angular_damping=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos), rot=yaw_quat(yaw_rad)),
    )
