# PRISM Simulation Pipeline

This directory contains the staged simulation pipeline that replaces the older
single-file Isaac Sim replay scripts.

## One-command Replay Video Pipeline

Run corrected-frame conversion, robot motion planning, robot material overrides,
and planned-motion replay video export in one command:

```bash
/isaac-sim/python.sh simulation/scripts/run_replay_pipeline.py \
  --trial-dir data/raw/task_20260730_103856_ball_picking/trial_000012 \
  --calib-json configs/devices/charuco_4cam_result.json
```

The default output directory is:

```text
data/processed/simulation/<task_dir>/<trial_dir>/
```

The final video defaults to `planned_motion_overhead.mp4` next to
`planned_motion.csv`. Use `--video-path`, `--camera-pos`, `--camera-look-at`,
`--stride`, `--max-frames`, `--replay-stride`, or `--replay-max-frames` to tune
the run. Use `--dry-run` to print the underlying stage commands without running
them.

## Stage 1: Correct Raw Trajectory

Convert a raw trial trajectory into corrected-frame artifacts without requiring
Isaac Sim runtime APIs:

```bash
/isaac-sim/python.sh simulation/scripts/trajectory_corrector.py \
  --trial-dir data/raw/task_20260730_103856_ball_picking/trial_000012 \
  --calib-json configs/devices/charuco_4cam_result.json
```

By default outputs are written to:

```text
data/processed/simulation/<task_dir>/<trial_dir>/
  corrected_trajectory.csv
  corrected_leds.csv
  corrected_meta.json
  corrected_preview.png
```

Use `--out-dir` to choose another output directory, and `--use-raw` to use raw
pose/LED columns instead of smoothed columns when both are present.

## Planned Stages

## Robot Asset: AUBO i5 + MechHand URDF

Build a single robot-tree URDF by appending the MechHand subtree to the AUBO i5
`wrist3_Link` through a fixed joint:

```bash
python3 simulation/scripts/build_combined_urdf.py
```

The default output is:

```text
simulation/assets/aubo_i5_mechhand/aubo_i5_mechhand.urdf
```

The MechHand links and joints are prefixed with `mechhand_` to avoid collisions
with the arm's `base_link`. Use `--mount-xyz` and `--mount-rpy` to tune the
fixed transform between the AUBO wrist and the hand base.

## Planned Stages

Import the combined URDF into Isaac Sim, save a USD asset, and inspect the robot
in the viewport:

```bash
/isaac-sim/python.sh simulation/scripts/build_robot_usd.py
```

For headless validation only:

```bash
/isaac-sim/python.sh simulation/scripts/build_robot_usd.py --headless --no-preview
```

The default USD output is:

```text
simulation/assets/aubo_i5_mechhand/aubo_i5_mechhand.usd
```

The USD build applies contrasting preview materials by default: orange AUBO
links with black joints and black-metal MechHand parts. To recolor an existing USD without
rebuilding the URDF import, run:

```bash
/isaac-sim/python.sh simulation/scripts/apply_robot_materials.py
```

## Planned Stages

## Stage 2: Plan Robot Motion

Before replaying planned joints, inspect the Isaac Sim scene and imported robot
asset:

```bash
/isaac-sim/python.sh simulation/scripts/setup_scene_preview.py \
  --corrected-trajectory data/processed/simulation/task_20260730_103856_ball_picking/trial_000012/corrected_trajectory.csv
```

For command-line validation only:

```bash
/isaac-sim/python.sh simulation/scripts/setup_scene_preview.py --headless --no-preview \
  --corrected-trajectory data/processed/simulation/task_20260730_103856_ball_picking/trial_000012/corrected_trajectory.csv
```

The script saves `simulation/scenes/aubo_i5_mechhand_preview.usd` with the robot,
lighting, table, corrected/world axes, workspace box, and optional trajectory
curve.

## Stage 3: Plan Robot Motion

Solve AUBO arm IK and expand MechHand gestures into per-frame joint targets:

```bash
/isaac-sim/python.sh simulation/scripts/plan_robot_motion.py \
  --corrected-trajectory data/processed/simulation/task_20260730_103856_ball_picking/trial_000012/corrected_trajectory.csv \
  --gestures data/raw/task_20260730_103856_ball_picking/trial_000012/hand/sdk_commands.csv
```

The default output is `planned_motion.csv` next to the corrected trajectory.
By default the planner follows corrected-frame position targets. Add
`--use-orientation` for stricter pose IK, and use `--max-frames N` for quick
smoke tests.

## Stage 4: Replay Planned Motion

Feed planned joint targets into the combined AUBO i5 + MechHand articulation:

```bash
/isaac-sim/python.sh simulation/scripts/replay_planned_motion.py \
  --planned-motion data/processed/simulation/task_20260730_103856_ball_picking/trial_000012/planned_motion.csv
```

Record the replay from the fixed overhead camera at `(0, 0.2, 0)`, looking at
`(0, 0, -0.9)`:

```bash
/isaac-sim/python.sh simulation/scripts/replay_planned_motion.py --record-video \
  --planned-motion data/processed/simulation/task_20260730_103856_ball_picking/trial_000012/planned_motion.csv
```

If the opposite side is needed, use `--camera-pos 0 -0.2 0`. The default video is
written next to the plan as `planned_motion_overhead.mp4`.

For command-line validation only:

```bash
/isaac-sim/python.sh simulation/scripts/replay_planned_motion.py --headless --no-preview --max-frames 10 \
  --planned-motion data/processed/simulation/task_20260730_103856_ball_picking/trial_000012/planned_motion.csv
```

The replay maps CSV columns to Isaac DOFs by joint name, not by order. The final
stage is saved to `simulation/scenes/aubo_i5_mechhand_replay.usd`.
