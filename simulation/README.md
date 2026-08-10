# PRISM Simulation Pipeline

This directory contains the staged simulation pipeline that replaces the older
single-file Isaac Sim replay scripts.

## Stage 0: Simulated Self-Collection (Raw-like)

When no real cameras are connected, generate synthetic task/trial data in a
layout similar to real collection outputs:

```bash
python3 simulation/scripts/collect_sim_trial.py \
  --task-name dexhand_sim \
  --num-trials 3 \
  --duration-sec 10 \
  --fps 30
```

Default output:

```text
data/raw/task_<timestamp>_<task_name>/
  task_metadata.yaml
  hand_sdk_commands_timeline.csv
  trial_000001/
    metadata.yaml
    trajectory/trajectory_led.csv
    trajectory/rigid_pose_6d.csv
    hand/sdk_commands.csv
    hand/rpi_commands.csv
```

This collector supports built-in motion patterns (`circle`, `figure8`,
`lissajous`, `line`), configurable trajectory/noise parameters, and synthetic
hand gesture events (`--gesture-seq`). Generated CSV headers are aligned with
the offline reconstruction schema so downstream analysis and correction tools
can reuse the same readers.

## Stage 0B: Task-Driven Collection (Pick and Place)

Use task semantics (approach, grasp, lift, transfer, release) instead of generic
curves:

```bash
python3 simulation/scripts/collect_sim_task.py \
  --task-type pick_place \
  --task-name pick_place_sim \
  --num-trials 3 \
  --skip-planning
```

This writes raw-like trial outputs under `data/raw/task_*/trial_*/` and also
writes task-phase trajectories to:

```text
data/processed/simulation/<task>/<trial>/
  corrected_trajectory.csv
  task_phases.csv
```

To run planning + Isaac replay execution after generation, use Isaac Python and
enable replay explicitly:

```bash
/isaac-sim/python.sh simulation/scripts/collect_sim_task.py \
  --task-type pick_place \
  --task-name pick_place_exec \
  --num-trials 1 \
  --execute-replay --record-video --headless
```

If planning fails in a non-Isaac environment, run collection-only mode with
`--skip-planning` first, then replay later with Isaac Sim.

When `--record-video` is enabled, camera output is written directly into each
trial camera folder (default camera name `sim_overhead`):

```text
data/raw/task_*/trial_*/cameras/
  sim_overhead.mp4
  sim_overhead_timestamps.csv
```

By default, replay/video failures are treated as non-fatal warnings so task
collection files are still kept. Add `--strict-replay` to make replay failure
abort the run.

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
`planned_motion.csv`. The default robot config enables orientation-aware IK for
the AUBO arm, so wrist joints are used to follow both corrected-frame position
and orientation. Use `--no-use-orientation` for faster position-only previews,
or `--orientation-weight`, `--max-iters`, `--tolerance`, `--stride`,
`--max-frames`, `--replay-stride`, or `--replay-max-frames` to tune the run. Use
`--dry-run` to print the underlying stage commands without running them.

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
By default the planner reads `planning.use_orientation` from
`simulation/configs/aubo_i5_mechhand.yaml`, which enables stricter pose IK in the
default config. Use `--no-use-orientation` for faster position-only previews, or
`--use-orientation` to force pose IK when another robot config disables it. Use
`--max-frames N` and `--stride N` for quick smoke tests.

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

## Stage 5: RPi Encoder Teleop for support_right

Drive the third-generation support_right hand directly from RPi encoder UDP events
(`prism.rpi_hand_event.v1`), with 5-channel mapping to main joints and optional
distal-joint coupling. Default input mode is direct encoder angle (`angles`).

```bash
/isaac-sim/python.sh simulation/scripts/teleop_support_right_from_rpi.py \
  --config simulation/configs/support_right_rpi_5ch.yaml
```

By default the teleop listener binds `0.0.0.0:60701`, matching the RPi
`io_interface.py` default event target. You can override at runtime:

```bash
/isaac-sim/python.sh simulation/scripts/teleop_support_right_from_rpi.py \
  --config simulation/configs/support_right_rpi_5ch.yaml \
  --event-host 0.0.0.0 --event-port 60701 --print-dofs
```

Important: the legacy RPi event log was edge-triggered by SDK pose sends. For
continuous angle teleop, run RPi `io_interface.py` with telemetry streaming:

```bash
python3 -m prism.devices.rpi.io_interface --disable-hand-trigger --event-stream-hz 30
```

This continuously emits UDP packets containing `angles` (Enc1..Enc5) so Isaac
teleop can follow hand motion in real time, independent of SDK command events.

The teleop config now includes anti-jitter controls for near-static poses:

```yaml
control:
  median_window: 5            # per-channel median window
  input_deadband_deg: 0.6     # ignore tiny encoder changes in angle mode
  smoothing_alpha: 0.35       # first-order smoothing
  joint_deadband_rad: 0.012   # ignore tiny output joint changes
  max_joint_speed_rad_s: 3.0  # slew-rate limit for output joints
```

Tuning guidance:

- If still shaky at rest: increase `input_deadband_deg` and `joint_deadband_rad`.
- If response feels too slow: reduce `median_window`/`joint_deadband_rad` or raise
  `max_joint_speed_rad_s`.

Tune channel-to-joint angle ranges and mimic rules in:

```text
simulation/configs/support_right_rpi_5ch.yaml
```

Quick validation (load robot + verify joint names, then exit):

```bash
/isaac-sim/python.sh simulation/scripts/teleop_support_right_from_rpi.py \
  --config simulation/configs/support_right_rpi_5ch.yaml \
  --headless --no-preview --print-dofs
```

Path convenience: if you pass only a filename like
`--config support_right_rpi_5ch.yaml`, the script will also try
`simulation/configs/` automatically.

When Isaac logs many startup warnings, use the script's own markers to judge
success: `imported articulation`, `dof_count=...`, and
`robot loaded and mapping validated`.
