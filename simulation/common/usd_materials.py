"""USD material helpers for simulation robot assets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


Color = Tuple[float, float, float]


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    color: Color
    roughness: float = 0.55
    metallic: float = 0.0


DEFAULT_ROBOT_MATERIALS = {
    "arm": MaterialSpec("ArmMainOrange", (0.95, 0.38, 0.06), roughness=0.46, metallic=0.05),
    "arm_dark": MaterialSpec("ArmJointBlack", (0.015, 0.015, 0.016), roughness=0.50, metallic=0.20),
    "hand_palm": MaterialSpec("HandPalmBlackMetal", (0.012, 0.012, 0.014), roughness=0.30, metallic=0.85),
    "hand_finger": MaterialSpec("HandFingerBlackMetal", (0.018, 0.017, 0.016), roughness=0.28, metallic=0.90),
}


def _make_preview_material(stage, root_path: str, spec: MaterialSpec):
    from pxr import Gf, Sdf, UsdShade

    material_path = "%s/%s" % (root_path.rstrip("/"), spec.name)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, "%s/PreviewSurface" % material_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*spec.color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(spec.roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(spec.metallic))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _material_key_for_path(path: str) -> str:
    lowered = path.lower()
    if "mechhand_" in lowered:
        if "mechhand_base_link" in lowered:
            return "hand_palm"
        return "hand_finger"
    if any(token in lowered for token in ("shoulder", "wrist", "base_link")):
        return "arm_dark"
    return "arm"


def apply_robot_materials(stage, robot_root_path: str = "/World/AUBO_i5_MechHand") -> Dict[str, int]:
    """Bind colored preview materials to robot visual geometry.

    Returns counts per material key. Collision meshes are left unmodified.
    """
    from pxr import Gf, Usd, UsdGeom, UsdShade

    candidate_roots = []
    root = stage.GetPrimAtPath(robot_root_path)
    if root and root.IsValid():
        candidate_roots.append(root)
    visual_root = stage.GetPrimAtPath("/visuals")
    if visual_root and visual_root.IsValid() and str(visual_root.GetPath()) not in {str(prim.GetPath()) for prim in candidate_roots}:
        candidate_roots.append(visual_root)
    if not candidate_roots:
        raise RuntimeError("no robot visual roots found; tried %s and /visuals" % robot_root_path)

    material_root = "/World/Looks"
    materials = {key: _make_preview_material(stage, material_root, spec) for key, spec in DEFAULT_ROBOT_MATERIALS.items()}
    colors = {key: Gf.Vec3f(*spec.color) for key, spec in DEFAULT_ROBOT_MATERIALS.items()}
    counts = {key: 0 for key in DEFAULT_ROBOT_MATERIALS}

    seen_paths = set()
    for candidate_root in candidate_roots:
        for prim in Usd.PrimRange(candidate_root):
            path = str(prim.GetPath())
            if path in seen_paths:
                continue
            seen_paths.add(path)
            lowered = path.lower()
            if lowered == "/colliders" or lowered.startswith("/colliders/"):
                continue
            is_mesh = prim.IsA(UsdGeom.Mesh)
            is_subset = prim.GetTypeName() == "GeomSubset"
            has_material_binding = any(prop.GetName().startswith("material:binding") for prop in prim.GetProperties())
            if not (is_mesh or is_subset or has_material_binding):
                continue
            key = _material_key_for_path(path)
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            binding_api.UnbindAllBindings()
            binding_api.Bind(materials[key], UsdShade.Tokens.strongerThanDescendants)
            binding_api.Bind(materials[key], UsdShade.Tokens.strongerThanDescendants, UsdShade.Tokens.preview)
            if is_mesh:
                gprim = UsdGeom.Gprim(prim)
                gprim.CreateDisplayColorAttr([colors[key]])
                gprim.CreateDisplayOpacityAttr([1.0])
            counts[key] += 1
    return counts
