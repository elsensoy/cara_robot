# Known gaps — Cara

Flat inventory, for future reference. No narrative, no "how we found this" —
just where the holes are.

## Sensing / IMU

- No calibration step verifies the BNO055's physical mounting matches the
  assumed body frame (`AXIS_MAP_CONFIG` register never touched, still on
  power-on default).
- Slow IMU drift is not detected — distinct from the frozen-data check,
  which only catches exact bit-for-bit repeats.
- No sensor redundancy anywhere (one IMU, one aggregate power rail) — "two
  sensors disagree" can't currently arise because there's nothing to
  disagree with.
- Loop rate is never actually measured. 50Hz is the design target; nothing
  logs achieved period or jitter.
- No delay-margin analysis for the controller against growing IMU latency —
  only the freshness check gates whether a sample is used, nothing bounds
  what happens to control stability as delay grows.
- `still_cmd_eps_rad`, `resting_wobble_rad_s`, and the various hold-time
  constants in `ImuGuard` are reasoned defaults, never tuned against real
  IMU noise on real hardware.

## Power / current sensing

- INA219 shunt value, max current per servo, and resulting shunt voltage
  drop were never independently derived for Cara — inherited from the
  stock Adafruit breakout and its example 32V/2A calibration constant.
- Aggregate rail thresholds (idle/warn/critical) are bench-tuned once, not
  re-derived systematically.
- Whether the shunt's own series resistance measurably affects servo
  performance under load has never been quantified.
- Zero physical INA219 chips have ever been wired together and read back —
  the per-joint current-sensing *software* path is real; the *physical
  topology* is completely unvalidated, not even at 2 channels.
- `Ina219MultiPower::read()` timestamps an entire per-joint sweep with one
  shared value, not per channel — channels aren't actually synchronized,
  and the data doesn't say so.
- No cross-check between the aggregate rail reading and the sum of
  per-joint readings when both are present.
- INA219 addressing tops out at 16 per bus; no mux or multi-bus support
  exists in code (though grouping by function — waist+hips only, ~9
  channels — may avoid ever needing one).

## Actuation / mechanical

- No encoders anywhere in the system. Commanded position stands in for
  measured position everywhere — this is the single largest structural gap
  in the stack.
- A servo that's electrically fine but mechanically jammed or disconnected
  is fully undetectable.
- Arduino death is undetectable — the serial command protocol has no
  acknowledgment; the Jetson keeps writing into the void.
- No real E-stop, torque cutoff, or tuck-posture behavior is implemented.
  These exist only as intent in an older design document. A fall-detection
  method (`check_safety_governor`) is fully written in `cara_body_node.py`
  but never called from anywhere.

## Build / infra

- `jetson/control/CMakeLists.txt` currently reads `project(car_control CXX)`
  — missing the "a." Everything else (executable name, directory, ROS
  package) is `cara_control`. Looks like an accidental typo, unconfirmed,
  unfixed.
- `cara_control_node.cpp` (the ROS wrapper) has never actually been built —
  no ROS 2/rclcpp available in this environment. Reviewed by pattern-match
  against proven code in the same file, not compiled.
- The ROS node's `per_joint_addrs` param has no `DeclareLaunchArgument`
  wrapper — settable via a params file only, not a launch CLI override.

## Diagnostics architecture

- No live comparison between COMMAND, the URDF/MuJoCo model's expected
  response, and MEASUREMENT exists. The model itself exists in
  `cara_description/`; it has never been queried by the running
  `jetson/control` pipeline.
- The sensor-diagnostics layer (C++, `jetson/control`) and the RL
  environment (Python/MuJoCo, `cara_env.py`) are fully disconnected
  codebases. The RL observation vector has no health channel at all.
- None of the fault modes the diagnostics layer was built to catch (frozen
  sensor, stale telemetry, motion mismatch, current anomaly) have ever been
  injected into an RL training run.

## Locomotion / RL

- Dynamic (non-quasi-static) walking is incomplete. A real ankle
  software-PD instability was root-caused and a fix found (contact-gated
  damping) but never wired into the live DCM controller — paused, not
  finished.
- The gait phase-transition fragility found in the DCM controller (U35) is
  unresolved: even the strongest possible reference-level correction fails
  to fix it, and the actual physical-state cause (velocity/momentum/contact
  at the transition) hasn't been characterized.
- No RL policy has been trained to a working walking gait yet (multiple
  ARS/PPO attempts converged to safe-but-not-walking local optima).
- The residual-RL comparison this was building toward — teacher alone vs.
  teacher+residual under disturbance — has never been run; gated on the
  phase-transition finding above.
- No domain randomization has been done (mass/CoM, friction, actuator
  strength, delay, sensor noise, pushes) — still just a roadmap item.
- No adaptation/robustness policy work has started (recurrent/history
  policy vs. a fixed baseline under held-out changes).
- No sim-to-real transfer has ever been attempted for any of the walking or
  RL work.

## Design/manufacturing

- Dynamics parameters across the model (PD gains, effort ceilings, ground
  friction) are provisional/TODO-marked pending real servo selection.
- Real inter-axis hip offsets are deferred — the hip is still modeled as
  three coincident axes (a spherical approximation).
- No actuators have been selected for the eventual 20-DoF hardware build;
  CAD/manufacturing work is parallel, not blocking, but also not started
  in earnest.

## Documentation / assumption mismatches

- The documented 20-DoF joint table (waist 3, neck 3, shoulders 6, hips 6,
  ears 2) has no separate "eyes" servo — vision is a fixed USB camera.
  Worth confirming this hasn't changed rather than assuming it.
