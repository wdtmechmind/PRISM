# MechHand V3 Real Hardware Integration Plan

This plan is derived from MechHand V3 daemon SDK manual and aligned with the current PRISM simulation teleop pipeline.

## 1. SDK Facts Confirmed

- Daemon runs as headless service and exposes TCP command server.
- Frame format:
  - client to daemon: `@<CMD><VALUE>&`
  - daemon to client: `@_<CMD><RESULT>&`
- Connectivity probe:
  - request: `@CC<>&`
  - ready: `@_CC<1>&`
- Direct joint-angle command:
  - request: `@RM<J1,J2,J3,J4,J5>&`
  - units: degree x 10
  - range: `-900..900`
  - no-op per joint: `-32768`
- Typical runtime settings:
  - speed: `SV<...>`
  - force: `SF<...>`
  - running timeout: `SRT<...>`
- Common error contract: `@_<CMD><ERR:<reason>>&`

## 2. Target Architecture in PRISM

Use a two-layer control model:

- Layer A (already done): RPi encoder telemetry producer over UDP.
  - Source: [src/prism/devices/rpi/io_interface.py](src/prism/devices/rpi/io_interface.py)
  - Continuous stream option: `--event-stream-hz`
- Layer B (next): MechHand V3 SDK TCP client in host process.
  - New module suggestion: `src/prism/devices/hand/v3_daemon_client.py`
  - Responsibility:
    - connection/reconnect
    - frame encode/decode
    - timeout and retry
    - error classification

Control data flow on real hardware:

1. RPi sends encoder angles over UDP.
2. Host receives and filters (already in simulation teleop logic).
3. Host maps to 5-channel target angles.
4. Host converts radians to degree x 10 integers.
5. Host sends `RM<J1..J5>` to daemon over TCP.

## 3. Interface Mapping Rules

From current support_right config mapping to SDK joint order:

- SDK order is fixed: `J1=侧摆, J2=拇指, J3=食指, J4=中指, J5=无名`.
- Keep a single mapping function for real-hardware output:
  - clamp radians per finger range
  - convert to degrees
  - convert to int tenths
  - clamp `-900..900`

Important rule:

- Do not send high-rate gesture commands and RM commands concurrently.
- Real-time mode should use RM pipeline only.

## 4. Safety Controls (Must-Have)

Before first real motion, enforce all:

- Rate limiting for outgoing RM (recommended 20-30 Hz to start).
- Per-joint max delta per cycle.
- Deadman timeout:
  - if no fresh encoder input for N ms, stop updating and hold position.
- Startup handshake:
  - require `CC=1` before arming.
- Soft limits in host side stricter than SDK limit.
- Emergency stop command path:
  - immediate open posture or no-op hold fallback.

## 5. Implementation Steps

Step 1: Build SDK TCP client module

- Add `MechHandV3Client` with:
  - `connect()`
  - `send(cmd, value)`
  - `query_connected()` for `CC`
  - `move_rm(j1, j2, j3, j4, j5)`
  - `set_speed()/set_force()/set_runtime()`
- Add framed parser robust to sticky packets and partial packets.

Step 2: Add real-hardware teleop entrypoint

- New script suggestion: `tools/teleop_mechhand_v3_from_rpi.py`
- Reuse existing filter chain from simulation teleop:
  - median window
  - input deadband
  - joint deadband
  - slew rate limit
- Output target to daemon RM command.

Step 3: Add YAML for real hardware

- New config suggestion: `configs/deployment/mechhand_v3_teleop.yaml`
- Include:
  - daemon host/port
  - update rate
  - filter parameters
  - radian-to-degree ranges
  - startup speed/force/runtime

Step 4: Dry-run mode

- Log RM outputs without sending.
- Confirm mapping directions and ranges against expected finger motion.

Step 5: Low-speed first motion

- Use conservative values:
  - speed 20%
  - force 20-30%
  - runtime 2-3 s
- Validate one finger at a time.

Step 6: Closed-loop stability tuning

- Increase update rate gradually.
- Tighten or loosen deadbands from observed jitter.
- Record command/response latency stats.

## 6. Proposed Initial Parameters

- Control rate: 25 Hz.
- Outgoing RM hold timeout: 250 ms.
- Median window: 5.
- Input deadband: 0.6 deg.
- Joint deadband: 0.01-0.015 rad.
- Max joint speed: 2.5-3.5 rad/s.

## 7. Acceptance Checklist (Round 1)

- `CC` check passes continuously for 10 minutes.
- No command parser errors under 25 Hz RM stream.
- No unexpected finger twitch at idle for 60 seconds.
- Emergency stop path verified.
- Reconnect behavior verified after daemon restart.

## 8. Next Action in This Repo

When you confirm, implement in this order:

1. `src/prism/devices/hand/v3_daemon_client.py`
2. `tools/teleop_mechhand_v3_from_rpi.py`
3. `configs/deployment/mechhand_v3_teleop.yaml`
4. Unit tests for frame parsing and clamp/convert logic.
