#!/usr/bin/env python3
"""Drive support_right hand in Isaac Sim from RPi 5-channel encoder UDP events.

Run with Isaac Sim Python:

    /isaac-sim/python.sh simulation/scripts/teleop_support_right_from_rpi.py

By default this listens to prism RPi event packets on UDP 0.0.0.0:60701 and maps
Enc1..Enc5 directly from encoder angles (degrees) to 5 main joints of
support_right. Distal joints can be coupled by mimic rules in the YAML config.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import yaml


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_repo_path(value: str) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def resolve_config_path(value: str) -> Path:
    """Resolve config path with a fallback to simulation/configs for bare names."""
    raw = str(value or "").strip()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidate = Path.cwd() / path
    if candidate.is_file():
        return candidate.resolve()
    fallback = REPO_ROOT / "simulation" / "configs" / path
    if fallback.is_file():
        return fallback.resolve()
    # Keep a deterministic absolute path for the final error message.
    return candidate.resolve()


def patch_package_mesh_uris(urdf_path: Path, patched_path: Path) -> Path:
    """Write a patched URDF whose mesh filenames are absolute paths."""
    src = urdf_path.read_text(encoding="utf-8")
    remap_dir = patched_path.parent / (patched_path.stem + "_mesh_alias")
    remap_dir.mkdir(parents=True, exist_ok=True)

    def sanitize_mesh_path(mesh_path: Path) -> Path:
        """Create an alias path for mesh files whose stem is USD-unsafe."""
        stem = mesh_path.stem
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", stem):
            return mesh_path
        safe_stem = re.sub(r"[^A-Za-z0-9_]", "_", stem)
        if not safe_stem or not re.match(r"^[A-Za-z_]", safe_stem):
            safe_stem = "mesh_" + safe_stem
        alias_path = remap_dir / (safe_stem + mesh_path.suffix)
        if alias_path.exists():
            return alias_path
        try:
            os.symlink(str(mesh_path), str(alias_path))
        except Exception:
            shutil.copy2(mesh_path, alias_path)
        return alias_path

    def replace(match: re.Match) -> str:
        uri = match.group(1)
        if uri.startswith("package:///meshes/"):
            rel = uri[len("package:///meshes/"):]
            candidate = urdf_path.parent / "meshes" / rel
            if candidate.is_file():
                return 'filename="%s"' % str(sanitize_mesh_path(candidate.resolve()))
        if uri.startswith("package://"):
            rel = uri[len("package://"):]
            candidate = REPO_ROOT / rel
            if candidate.is_file():
                return 'filename="%s"' % str(sanitize_mesh_path(candidate.resolve()))
            if rel.startswith("support_right/"):
                candidate = REPO_ROOT / "simulation" / rel
                if candidate.is_file():
                    return 'filename="%s"' % str(sanitize_mesh_path(candidate.resolve()))
        return match.group(0)

    patched = re.sub(r'filename="([^"]+)"', replace, src)
    patched_path.parent.mkdir(parents=True, exist_ok=True)
    patched_path.write_text(patched, encoding="utf-8")
    return patched_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "simulation" / "configs" / "support_right_rpi_5ch.yaml"),
        help="YAML mapping config",
    )
    parser.add_argument("--urdf", default=None, help="override robot URDF path")
    parser.add_argument("--prim-path", default=None, help="override imported robot prim path")
    parser.add_argument("--event-host", default=None, help="UDP bind host override")
    parser.add_argument("--event-port", type=int, default=None, help="UDP bind port override")
    parser.add_argument("--headless", action="store_true", help="run Isaac Sim headless")
    parser.add_argument("--no-preview", action="store_true", help="exit once robot loads and DOFs validate")
    parser.add_argument("--print-dofs", action="store_true", help="print articulation DOF names at startup")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _channel_value(payload: dict, channel: int):
    value = payload.get(channel)
    if value is not None:
        return value
    return payload.get(str(channel))


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(values: Deque[float]) -> float:
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def filter_channel_values(
    raw_values: Dict[int, float],
    channel_histories: Dict[int, Deque[float]],
    prev_filtered: Dict[int, float],
    input_mode: str,
    median_window: int,
    deadband_deg: float,
    deadband_norm: float,
) -> Dict[int, float]:
    """Apply median smoothing + input deadband per channel."""
    mode = str(input_mode or "angle_deg").strip().lower()
    deadband = deadband_deg if mode == "angle_deg" else deadband_norm
    out: Dict[int, float] = dict(prev_filtered)

    for channel, value in raw_values.items():
        history = channel_histories.setdefault(channel, deque(maxlen=max(1, int(median_window))))
        history.append(float(value))
        filtered = _median(history)
        prev = prev_filtered.get(channel)
        if prev is not None and abs(filtered - prev) < float(deadband):
            out[channel] = float(prev)
        else:
            out[channel] = float(filtered)
    return out


def apply_joint_output_limits(
    targets: Dict[str, float],
    prev_applied: Dict[str, float],
    deadband_rad: float,
    max_speed_rad_s: float,
    dt: float,
) -> Dict[str, float]:
    """Apply joint deadband and slew-rate limits to suppress steady-state jitter."""
    out: Dict[str, float] = dict(prev_applied)
    max_speed = max(0.0, float(max_speed_rad_s))
    deadband = max(0.0, float(deadband_rad))
    dt_sec = max(0.0, float(dt))

    for joint_name, desired in targets.items():
        d = float(desired)
        prev = float(prev_applied.get(joint_name, d))

        if abs(d - prev) < deadband:
            d = prev

        if max_speed > 0.0 and dt_sec > 0.0:
            max_step = max_speed * dt_sec
            delta = d - prev
            if delta > max_step:
                d = prev + max_step
            elif delta < -max_step:
                d = prev - max_step

        out[joint_name] = float(d)

    return out


def parse_encoder_channels(event: dict, input_mode: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    mode = str(input_mode or "angle_deg").strip().lower()

    if mode == "angle_deg":
        angles = event.get("angles") or {}
        for channel in range(1, 6):
            val = _safe_float(_channel_value(angles, channel))
            if val is None:
                val = _safe_float(event.get("enc%d_angle_deg" % channel))
            if val is None:
                val = _safe_float(event.get("enc%d_angle" % channel))
            if val is None:
                continue
            out[channel] = val
        return out

    positions = event.get("positions") or {}
    for channel in range(1, 6):
        val = _safe_float(_channel_value(positions, channel))
        if val is None:
            continue
        out[channel] = max(0.0, min(1.0, val))
    return out


def map_channels_to_joint_targets(
    channel_values: Dict[int, float],
    channels_cfg: dict,
    mimic_cfg: List[dict],
    prev_targets: Dict[str, float],
    alpha: float,
    input_mode: str,
) -> Dict[str, float]:
    targets = dict(prev_targets)
    mode = str(input_mode or "angle_deg").strip().lower()

    for channel in range(1, 6):
        cfg = channels_cfg.get(str(channel)) or channels_cfg.get(channel)
        if not cfg:
            continue
        value = channel_values.get(channel)
        if value is None:
            continue
        joint = str(cfg.get("joint", "")).strip()
        if not joint:
            continue
        invert = bool(cfg.get("invert", False))

        if mode == "angle_deg":
            src_min = float(cfg.get("source_min_deg", 0.0))
            src_max = float(cfg.get("source_max_deg", 360.0))
            if src_max <= src_min:
                continue
            p = (float(value) - src_min) / (src_max - src_min)
            p = max(0.0, min(1.0, p))
        else:
            p = max(0.0, min(1.0, float(value)))

        if invert:
            p = 1.0 - p

        min_rad = float(cfg.get("min_rad", 0.0))
        max_rad = float(cfg.get("max_rad", 0.0))
        raw = min_rad + p * (max_rad - min_rad)
        old = float(targets.get(joint, raw))
        targets[joint] = old + alpha * (raw - old)

    for rule in mimic_cfg:
        source_joint = str(rule.get("source_joint", "")).strip()
        target_joint = str(rule.get("target_joint", "")).strip()
        if not source_joint or not target_joint:
            continue
        source_value = targets.get(source_joint)
        if source_value is None:
            continue
        ratio = float(rule.get("ratio", 1.0))
        offset = float(rule.get("offset_rad", 0.0))
        targets[target_joint] = float(source_value) * ratio + offset

    return targets


def main() -> int:
    args = parse_args()
    cfg_path = resolve_config_path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(
            "config not found: %s\n"
            "Hint: use --config simulation/configs/support_right_rpi_5ch.yaml or --config support_right_rpi_5ch.yaml"
            % cfg_path
        )
    config = load_config(cfg_path)

    robot_cfg = config.get("robot") or {}
    udp_cfg = config.get("rpi_udp") or {}
    channels_cfg = config.get("channels") or {}
    mimic_cfg = list(config.get("mimic") or [])
    control_cfg = config.get("control") or {}
    input_mode = str(control_cfg.get("input_mode", "angle_deg")).strip().lower()

    urdf_path = resolve_repo_path(args.urdf or robot_cfg.get("urdf", "simulation/support_right/urdf/support_right.urdf"))
    if not urdf_path.is_file():
        raise SystemExit("URDF not found: %s" % urdf_path)

    prim_path = str(args.prim_path or robot_cfg.get("prim_path", "/World/SupportRightHand"))
    event_host = str(args.event_host or udp_cfg.get("host", "0.0.0.0"))
    event_port = int(args.event_port or udp_cfg.get("port", 60701))
    alpha = float(control_cfg.get("smoothing_alpha", 0.35))
    alpha = min(1.0, max(0.0, alpha))
    median_window = max(1, int(control_cfg.get("median_window", 5)))
    input_deadband_deg = max(0.0, float(control_cfg.get("input_deadband_deg", 0.6)))
    input_deadband_norm = max(0.0, float(control_cfg.get("input_deadband_norm", 0.005)))
    joint_deadband_rad = max(0.0, float(control_cfg.get("joint_deadband_rad", 0.012)))
    max_joint_speed_rad_s = max(0.0, float(control_cfg.get("max_joint_speed_rad_s", 3.0)))
    recv_timeout = max(0.0005, float(control_cfg.get("recv_timeout_sec", 0.002)))
    log_rate_hz = max(0.1, float(control_cfg.get("log_rate_hz", 2.0)))

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})

    import omni.kit.commands
    import omni.timeline
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    patched_urdf = urdf_path.with_suffix(".isaac_patched.urdf")
    patch_package_mesh_uris(urdf_path, patched_urdf)

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")
    import_config.merge_fixed_joints = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.import_inertia_tensor = True
    import_config.create_physics_scene = False
    import_config.set_default_drive_type(1)
    import_config.set_default_drive_strength(8.0e5)
    import_config.set_default_position_drive_damping(1.0e4)

    status, imported_prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(patched_urdf),
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status or not imported_prim_path:
        raise RuntimeError("URDFParseAndImportFile failed: %s" % patched_urdf)

    if str(imported_prim_path) != prim_path:
        try:
            omni.kit.commands.execute("MovePrim", path_from=str(imported_prim_path), path_to=prim_path)
            imported_prim_path = prim_path
        except Exception as exc:
            print("[teleop] WARN: MovePrim failed (%s -> %s): %s" % (imported_prim_path, prim_path, exc))

    physics = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    physics.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics.CreateGravityMagnitudeAttr().Set(9.81)

    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(2400.0)
    key.CreateAngleAttr(1.2)

    dome = UsdLux.DomeLight.Define(stage, "/World/FillLight")
    dome.CreateIntensityAttr(280.0)

    World(stage_units_in_meters=1.0).reset()
    omni.timeline.get_timeline_interface().play()
    for _ in range(6):
        simulation_app.update()

    articulation = SingleArticulation(prim_path=str(imported_prim_path))
    articulation.initialize()
    dof_names = list(articulation.dof_names)
    print("[teleop] imported articulation: %s" % str(imported_prim_path), flush=True)
    print("[teleop] dof_count=%d" % len(dof_names), flush=True)
    if args.print_dofs:
        print("[teleop] DOFs (%d): %s" % (len(dof_names), ", ".join(dof_names)), flush=True)

    name_to_index = {name: i for i, name in enumerate(dof_names)}
    mapped_joint_names: List[str] = []
    for channel in range(1, 6):
        c = channels_cfg.get(str(channel)) or channels_cfg.get(channel)
        if not c:
            continue
        mapped_joint_names.append(str(c.get("joint", "")).strip())
    for rule in mimic_cfg:
        mapped_joint_names.append(str(rule.get("target_joint", "")).strip())
    mapped_joint_names = [name for name in mapped_joint_names if name]

    missing = [name for name in sorted(set(mapped_joint_names)) if name not in name_to_index]
    if missing:
        raise RuntimeError("configured joints not found in articulation DOFs: %s" % ", ".join(missing))

    if args.no_preview:
        print("[teleop] robot loaded and mapping validated. exiting due to --no-preview", flush=True)
        simulation_app.close()
        return 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(recv_timeout)
    sock.bind((event_host, event_port))

    print("[teleop] listening RPi UDP: %s:%d" % (event_host, event_port))
    print("[teleop] input mode: %s" % input_mode)
    print("[teleop] controlling joints: %s" % ", ".join(sorted(set(mapped_joint_names))))
    if input_mode == "angle_deg":
        print("[teleop] filter: median_window=%d input_deadband_deg=%.3f joint_deadband_rad=%.4f max_joint_speed_rad_s=%.3f" % (
            median_window,
            input_deadband_deg,
            joint_deadband_rad,
            max_joint_speed_rad_s,
        ))
    else:
        print("[teleop] filter: median_window=%d input_deadband_norm=%.4f joint_deadband_rad=%.4f max_joint_speed_rad_s=%.3f" % (
            median_window,
            input_deadband_norm,
            joint_deadband_rad,
            max_joint_speed_rad_s,
        ))
    print("[teleop] press Ctrl-C to stop")

    q = np.zeros(len(dof_names), dtype=np.float64)
    try:
        q[:] = articulation.get_joint_positions()
    except Exception:
        pass

    joint_targets: Dict[str, float] = {}
    for name in dof_names:
        joint_targets[name] = float(q[name_to_index[name]])
    applied_targets: Dict[str, float] = dict(joint_targets)
    channel_histories: Dict[int, Deque[float]] = {}

    last_channels_raw: Dict[int, float] = {}
    last_channels: Dict[int, float] = {}
    last_print_time = 0.0
    last_apply_time = time.monotonic()
    event_count = 0

    try:
        while simulation_app.is_running():
            simulation_app.update()

            for _ in range(8):
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    break
                except BlockingIOError:
                    break
                try:
                    event = json.loads(data.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(event, dict):
                    continue
                values = parse_encoder_channels(event, input_mode=input_mode)
                if values:
                    last_channels_raw = values
                    event_count += 1

            if last_channels_raw:
                last_channels = filter_channel_values(
                    raw_values=last_channels_raw,
                    channel_histories=channel_histories,
                    prev_filtered=last_channels,
                    input_mode=input_mode,
                    median_window=median_window,
                    deadband_deg=input_deadband_deg,
                    deadband_norm=input_deadband_norm,
                )

            if len(last_channels) >= 5:
                joint_targets = map_channels_to_joint_targets(
                    last_channels,
                    channels_cfg=channels_cfg,
                    mimic_cfg=mimic_cfg,
                    prev_targets=joint_targets,
                    alpha=alpha,
                    input_mode=input_mode,
                )

                now_apply = time.monotonic()
                dt = now_apply - last_apply_time
                last_apply_time = now_apply
                applied_targets = apply_joint_output_limits(
                    targets=joint_targets,
                    prev_applied=applied_targets,
                    deadband_rad=joint_deadband_rad,
                    max_speed_rad_s=max_joint_speed_rad_s,
                    dt=dt,
                )

                for joint_name, value in applied_targets.items():
                    idx = name_to_index.get(joint_name)
                    if idx is None:
                        continue
                    q[idx] = float(value)
                articulation.set_joint_positions(q)

            now = time.time()
            if now - last_print_time >= 1.0 / log_rate_hz:
                last_print_time = now
                if last_channels:
                    if input_mode == "angle_deg":
                        ordered = ["enc%d=%.1fdeg" % (i, last_channels.get(i, float("nan"))) for i in range(1, 6)]
                    else:
                        ordered = ["enc%d=%.3f" % (i, last_channels.get(i, float("nan"))) for i in range(1, 6)]
                    print("[teleop] events=%d %s" % (event_count, " ".join(ordered)))
                else:
                    print("[teleop] waiting for encoder packets...")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())