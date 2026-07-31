#!/usr/bin/env python3
"""Build a single-tree AUBO i5 + MechHand URDF.

The output URDF keeps the AUBO i5 as the arm chain and appends the MechHand as a
prefixed subtree connected by a fixed joint. This gives Isaac Sim a clean robot
tree that can be imported as one articulation.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    space = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = space + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = space
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = space


def prefixed_name(name: str, prefix: str) -> str:
    return name if name.startswith(prefix) else prefix + name


def patch_mesh_filename(filename: str, source_urdf: Path, absolute_mesh_paths: bool) -> str:
    if filename.startswith("package:///meshes/"):
        suffix = filename[len("package:///meshes/"):]
        mesh_path = source_urdf.parent / "meshes" / suffix
        return str(mesh_path.resolve()) if absolute_mesh_paths else "package://MechHand_deg45_V2_URDF/meshes/" + suffix
    if filename.startswith("package://aubo_description/"):
        suffix = filename[len("package://aubo_description/"):]
        mesh_path = _REPO_ROOT / "aubo_description" / suffix
        return str(mesh_path.resolve()) if absolute_mesh_paths else filename
    return filename


def patch_meshes(root: ET.Element, source_urdf: Path, absolute_mesh_paths: bool) -> None:
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename:
            mesh.set("filename", patch_mesh_filename(filename, source_urdf, absolute_mesh_paths))


def copy_aubo_children(
    aubo_root: ET.Element,
    output_root: ET.Element,
    aubo_urdf: Path,
    absolute_mesh_paths: bool,
    default_arm_effort: float,
    default_arm_velocity: float,
) -> None:
    for child in list(aubo_root):
        if child.tag == "link" and child.attrib.get("name") == "world":
            continue
        if child.tag == "joint" and child.attrib.get("name") == "world_joint":
            continue
        copied = copy.deepcopy(child)
        if copied.tag == "joint" and copied.attrib.get("type") in {"revolute", "continuous", "prismatic"}:
            limit = copied.find("limit")
            if limit is not None:
                if float(limit.attrib.get("effort", "0") or 0.0) <= 0.0:
                    limit.set("effort", "%.6g" % float(default_arm_effort))
                if float(limit.attrib.get("velocity", "0") or 0.0) <= 0.0:
                    limit.set("velocity", "%.6g" % float(default_arm_velocity))
        patch_meshes(copied, aubo_urdf, absolute_mesh_paths)
        output_root.append(copied)


def copy_prefixed_hand_children(hand_root: ET.Element, output_root: ET.Element, hand_urdf: Path, prefix: str, absolute_mesh_paths: bool) -> str:
    root_link_name: Optional[str] = None
    child_links = set()
    for joint in hand_root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.attrib.get("link"):
            child_links.add(child.attrib["link"])

    for link in hand_root.findall("link"):
        name = link.attrib.get("name", "")
        if name and name not in child_links:
            root_link_name = name
            break
    if root_link_name is None:
        root_link = hand_root.find("link")
        if root_link is None or not root_link.attrib.get("name"):
            raise RuntimeError("MechHand URDF has no root link")
        root_link_name = root_link.attrib["name"]

    for child in list(hand_root):
        if child.tag not in {"link", "joint", "material", "gazebo", "transmission"}:
            continue
        copied = copy.deepcopy(child)
        if copied.tag == "link":
            copied.set("name", prefixed_name(copied.attrib["name"], prefix))
        elif copied.tag == "joint":
            copied.set("name", prefixed_name(copied.attrib["name"], prefix))
            parent = copied.find("parent")
            if parent is not None and parent.attrib.get("link"):
                parent.set("link", prefixed_name(parent.attrib["link"], prefix))
            joint_child = copied.find("child")
            if joint_child is not None and joint_child.attrib.get("link"):
                joint_child.set("link", prefixed_name(joint_child.attrib["link"], prefix))
            limit = copied.find("limit")
            if limit is not None:
                if float(limit.attrib.get("effort", "0") or 0.0) <= 0.0:
                    limit.set("effort", "5.0")
                if float(limit.attrib.get("velocity", "0") or 0.0) <= 0.0:
                    limit.set("velocity", "3.0")
        patch_meshes(copied, hand_urdf, absolute_mesh_paths)
        output_root.append(copied)
    return prefixed_name(root_link_name, prefix)


def add_fixed_mount_joint(output_root: ET.Element, parent_link: str, child_link: str, xyz: str, rpy: str) -> None:
    joint = ET.Element("joint", {"name": "aubo_wrist3_to_mechhand_base", "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})
    ET.SubElement(joint, "parent", {"link": parent_link})
    ET.SubElement(joint, "child", {"link": child_link})
    output_root.append(joint)


def names(root: ET.Element, tag: str) -> List[str]:
    return [elem.attrib["name"] for elem in root.findall(tag) if elem.attrib.get("name")]


def find_duplicate_names(values: Iterable[str]) -> List[str]:
    seen = set()
    duplicates = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def validate_tree(root: ET.Element) -> None:
    link_names = set(names(root, "link"))
    duplicate_links = find_duplicate_names(names(root, "link"))
    duplicate_joints = find_duplicate_names(names(root, "joint"))
    if duplicate_links:
        raise RuntimeError("duplicate link names: %s" % ", ".join(duplicate_links))
    if duplicate_joints:
        raise RuntimeError("duplicate joint names: %s" % ", ".join(duplicate_joints))
    for joint in root.findall("joint"):
        joint_name = joint.attrib.get("name", "<unnamed>")
        parent = joint.find("parent")
        child = joint.find("child")
        parent_link = parent.attrib.get("link") if parent is not None else None
        child_link = child.attrib.get("link") if child is not None else None
        if parent_link not in link_names:
            raise RuntimeError("joint %s parent link not found: %s" % (joint_name, parent_link))
        if child_link not in link_names:
            raise RuntimeError("joint %s child link not found: %s" % (joint_name, child_link))


def build_combined_urdf(args: argparse.Namespace) -> Path:
    aubo_urdf = Path(args.aubo_urdf).expanduser().resolve()
    hand_urdf = Path(args.hand_urdf).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not aubo_urdf.is_file():
        raise SystemExit("AUBO URDF not found: %s" % aubo_urdf)
    if not hand_urdf.is_file():
        raise SystemExit("MechHand URDF not found: %s" % hand_urdf)

    aubo_root = ET.parse(aubo_urdf).getroot()
    hand_root = ET.parse(hand_urdf).getroot()
    output_root = ET.Element("robot", {"name": args.robot_name})
    copy_aubo_children(
        aubo_root,
        output_root,
        aubo_urdf,
        args.absolute_mesh_paths,
        args.default_arm_effort,
        args.default_arm_velocity,
    )
    hand_root_link = copy_prefixed_hand_children(
        hand_root,
        output_root,
        hand_urdf,
        args.hand_prefix,
        args.absolute_mesh_paths,
    )
    add_fixed_mount_joint(output_root, args.parent_link, hand_root_link, args.mount_xyz, args.mount_rpy)
    validate_tree(output_root)

    indent_xml(output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(output_root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aubo-urdf", default=str(_REPO_ROOT / "aubo_description" / "aubo_i5.urdf"))
    parser.add_argument("--hand-urdf", default=str(_REPO_ROOT / "MechHand_deg45_V2_URDF" / "MechHand_deg45_V2.urdf"))
    parser.add_argument("--output", default=str(_REPO_ROOT / "simulation" / "assets" / "aubo_i5_mechhand" / "aubo_i5_mechhand.urdf"))
    parser.add_argument("--robot-name", default="aubo_i5_mechhand")
    parser.add_argument("--parent-link", default="wrist3_Link", help="AUBO link that the hand base is fixed to")
    parser.add_argument("--hand-prefix", default="mechhand_", help="prefix applied to all MechHand links and joints")
    parser.add_argument("--mount-xyz", default="0 0 0", help="fixed joint origin xyz from parent link to hand base")
    parser.add_argument("--mount-rpy", default="0 0 0", help="fixed joint origin rpy from parent link to hand base")
    parser.add_argument("--default-arm-effort", type=float, default=150.0, help="effort limit used when AUBO joints specify zero")
    parser.add_argument("--default-arm-velocity", type=float, default=3.14, help="velocity limit used when AUBO joints specify zero")
    parser.add_argument("--absolute-mesh-paths", action="store_true", help="write absolute mesh filenames for Isaac imports without ROS package resolution")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_path = build_combined_urdf(args)
    print("[combined-urdf] wrote %s" % output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
