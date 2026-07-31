#!/usr/bin/env python3
"""Apply contrasting materials to the AUBO i5 + MechHand USD asset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--robot-usd",
        default=str(_REPO_ROOT / "simulation" / "assets" / "aubo_i5_mechhand" / "aubo_i5_mechhand.usd"),
        help="input robot USD",
    )
    parser.add_argument("--out-usd", default=None, help="output USD; defaults to overwriting --robot-usd")
    parser.add_argument("--robot-root", default="/World/AUBO_i5_MechHand")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    robot_usd = Path(args.robot_usd).expanduser().resolve()
    out_usd = Path(args.out_usd).expanduser().resolve() if args.out_usd else robot_usd
    if not robot_usd.is_file():
        raise SystemExit("robot USD not found: %s" % robot_usd)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    from pxr import Usd
    from simulation.common.usd_materials import apply_robot_materials

    stage = Usd.Stage.Open(str(robot_usd))
    if stage is None:
        raise SystemExit("could not open USD: %s" % robot_usd)
    counts = apply_robot_materials(stage, args.robot_root)
    out_usd.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out_usd))
    print("[robot-materials] wrote: %s" % out_usd)
    print("[robot-materials] bound prims: %s" % ", ".join("%s=%d" % item for item in sorted(counts.items())))

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
