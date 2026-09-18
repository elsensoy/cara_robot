# cara_control — Jetson-side health → observation → control signal path

The non-RL prototype of the deployment-side pipeline. It routes a learned-style
**actuator health signal** into a policy observation and shows, qualitatively,
that the controller reacts to it. Same loop, same `Observation`/`Action` types
run in simulation and on Cara — the only swap later is `HandwrittenController`
→ exported RL policy.

```
BNO055 ──► ImuGuard ──┐
                      ├──► ObservationBuilder ──► Controller ──► SafetyFilter ──► servos
INA219 (+ per-joint) ──┴──► HealthEstimator ──┘
                          system_health ∈ [0,1]
```

`ImuGuard` and `HealthEstimator`'s `telemetry_state` are the "can I trust
this sensor" layer, separate from what the sensor is reporting: staleness, a
frozen/derivative-implausibility check (real sensor noise doesn't repeat
bit-for-bit), and a commanded-vs-measured motion cross-check, all behind a
`PersistenceGate` so one dropped read doesn't flip a trust verdict either
way. See `diagnostics.hpp` and the "Diagnostics" section below.

## The `system_health` vs per-servo distinction

| | without `--per-joint-addrs` (default) | with it wired |
|---|---|---|
| `HealthState::system` | **the trustworthy field** — one scalar from the aggregate servo-rail INA219 | unchanged |
| `HealthState::per_servo` | mirrors `system`; `per_servo_valid == false` | real per-joint EMA health, `per_servo_valid == true` |
| `Observation` | appends `system_health` as the last element (`OBS_SIZE = 14`) | unchanged — the per-joint vector isn't in the observation yet, only in `HealthState` for now |

The per-joint current-sensing path (`PerJointPowerSource`, `Ina219MultiPower`)
is real, wired, and exercised in `--sim` by default (`SimPerJointPower`). On
hardware it stays a no-op (`per_servo_valid == false`, mirroring `system`
exactly as before) until `--per-joint-addrs` names real INA219 addresses —
one per servo, needs the A0–A5 pins strapped to distinct addresses.

## Build

```bash
cd jetson/control
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Needs `libi2c-dev` (already on the Jetson). On a machine without it:
`cmake -S . -B build -DCARA_WITH_HARDWARE=OFF` builds the sim-only binary.

## Run

```bash
# Simulation — no hardware. Scripted actuator-health fault every 16 s.
./build/cara_control --sim --duration 17

# On Cara — real BNO055 + INA219, commands out to the Arduino.
./build/cara_control --hw --serial /dev/ttyUSB0
```

Flags: `--rate HZ` (default 50), `--duration S` (0 = forever), `--bus N`
(default 1), `--ina-addr 0x45`, `--imu-addr 0x28`, `--per-joint-addrs
0xNN,...` (7 addresses, hw only, see above), `--test-imu-freeze` /
`--test-power-dropout` (sim only, see "Diagnostics" below).

### What to look for in `--sim`

As simulated servo-rail current ramps 240 → 2760 mA and the rail sags
5.05 → 4.50 V, `health` falls 1.00 → ~0.12, the controller `gain` follows it
down ~0.98 → ~0.32, and the `action` amplitudes shrink (shoulder ±0.19 →
±0.06 rad) — then everything recovers. That is the behavioural response to the
health input, with no reward function involved.

## Diagnostics

Log columns (`--sim`, `--duration 20`): `sev` is `HealthEstimator`'s existing
power-level severity (`ok`/`warn`/`critical`); `tlm` is its new
`telemetry_state` — do we trust that severity at all right now
(`ok`/`degraded`/`fault`); `imu` is `ImuGuard`'s own verdict on the IMU
specifically; `pj` is the minimum per-joint health (`-1.00` when no
per-joint source is present).

Three things are checked before anything downstream trusts a sample:

- **Staleness** — a sample whose timestamp is older than `max_age_s` counts
  as bad, same as a failed read.
- **Frozen / derivative-implausible data** — `ImuGuard` compares consecutive
  IMU samples bit-for-bit; real sensor noise doesn't repeat exactly, so
  several identical readings in a row (`frozen_repeat_trip`, default 5) is
  treated as a suspected fault, not a legitimately still robot.
- **Commanded-vs-measured motion mismatch** — the last safety-filtered servo
  command is a second, independent prediction of how the robot should be
  moving. If it's been holding still for `still_hold_s` and the IMU keeps
  reporting angular velocity above `resting_wobble_rad_s`, that's flagged.

All three feed one `PersistenceGate` (`diagnostics.hpp`): trips `fault`
once bad readings have persisted continuously for `trip_s` (default 0.06s),
clears back to `ok` only once good readings have persisted continuously for
`clear_s` (default 0.20s) — quick to distrust, slower to trust again.
`HealthEstimator` runs its own gate on power-telemetry validity; once that
trips, `system` decays toward `stale_health_floor` (default 0.5) instead of
holding whatever the last good reading said, forever.

The gate is duration-based, not tick-counted, on purpose: this loop
nominally runs at a fixed rate, but a tick-counted policy silently
redefines its own real-time meaning whenever scheduling jitter changes how
long a tick actually takes (see the timing cluster: what happens if one
iteration suddenly takes 80ms). Timing it against the caller's own
monotonic clock (`now` in `ImuGuard::update`, the sample's own `t_s` in
`HealthEstimator`, which the driver sets whether or not the read succeeded)
keeps fault semantics defined in wall-clock time regardless of how fast the
loop happens to be running at that moment. At the nominal 50Hz rate the
defaults above reproduce the same effective behavior as the tick counts
they replaced (3 ticks / 10 ticks) — only the mechanism changed.

Exercise both detectors with no hardware: `--sim --test-imu-freeze` latches
the simulated IMU output for part of each 16s cycle; `--sim
--test-power-dropout` makes the simulated power source report failed reads
for part of each cycle. Both recover on their own once the window closes —
watch the `imu`/`tlm` columns flip to `fault` and back.

## Components

| File | Role |
|---|---|
| `types.hpp` | `NUM_SERVOS`, joint table (from `arduino/main.cpp`), `Observation`/`Action`/`HealthState` layout, `PerJointCurrentSample`, `ImuDiagnostics` |
| `diagnostics.hpp/.cpp` | `PersistenceGate` (hysteresis), `ImuGuard` (staleness + frozen-data + motion-mismatch, gated) |
| `HealthEstimator` | EMA + threshold logic ported from `tests/cara_power_monitor.py`, mapped to `[0,1]`; own telemetry `PersistenceGate`; per-joint EMA when a `PerJointPowerSource` is present |
| `ObservationBuilder` | commanded joint pos + projected gravity + base ang-vel + `system_health` |
| `HandwrittenController` | dumb: scales gait amplitude by health and IMU wobble. **The seam the RL policy replaces.** |
| `SafetyFilter` | clamps to per-joint limits + rate-limits — last line before the servos |
| `sources_sim.cpp` | synthetic IMU/power/gait with a shared fault timeline; `SimPerJointPower`; `--test-imu-freeze`/`--test-power-dropout` injection |
| `sources_hw.cpp` | `Ina219Power`, `Ina219MultiPower` (per-joint), `Bno055Imu` (on `tests/imu_test.cpp`'s `I2CDevice`), `SerialOutput` (`S<ch>,<deg>` to the Nano) |

## ROS 2 integration (`ros2_ws/src/cara_control`)

`ros2_ws/src/cara_control` is a thin `ament_cmake` wrapper that compiles the
pipeline sources from this directory (single source of truth) and runs them on a
50 Hz timer. Observation-only — it never commands servos — so it is safe to run
alongside `cara_stack.launch.py`.

**Not yet wired into the ROS node:** `ImuGuard` and the per-joint power path.
`cara_control_node.cpp` still calls `health_.update(ps)` with no per-joint
sample and consumes `imu_src_->read()` directly, so it compiles unchanged
against the extended `HealthEstimator` (the new parameter defaults to
`nullptr`) but doesn't get the new trust checks — only the standalone
`cara_control` binary does. Wiring them in is mechanical (same pattern as
`main.cpp`'s loop) but deliberately not done here yet, to keep this change
reviewable in one piece.

```bash
# in the container
cd /workspace/ros2_ws
colcon build --packages-select cara_control      # add --cmake-args -DCARA_WITH_HARDWARE=OFF if libi2c-dev is missing
source install/setup.bash
ros2 launch cara_control cara_control_sim.launch.py          # source:=hw for real sensors
```

### Watch the health signal move

```bash
ros2 topic echo /cara/health/system            # 1.0 -> ~0.1 -> 1.0 on the sim cycle
ros2 run rqt_plot rqt_plot /cara/health/system/data /cara/control/gain/data

# force a fault on demand instead of waiting for the 16 s cycle:
ros2 topic pub --once /cara/sim/fault std_msgs/Float32 "{data: 1.0}"
ros2 topic pub --once /cara/sim/fault std_msgs/Float32 "{data: -1.0}"   # release
```

| Topic | Type | Meaning |
|---|---|---|
| `/cara/health/system` | `Float32` | health scalar, 0..1 |
| `/cara/health/state` | `String` | `ok` \| `warn` \| `critical` |
| `/cara/health/servo_rail_current_ma` | `Float32` | smoothed servo-rail current |
| `/cara/health/servo_rail_voltage_v` | `Float32` | smoothed servo-rail voltage |
| `/cara/control/gain` | `Float32` | controller's response to health (0.25..1.0) |
| `/cara/control/observation` | `Float32MultiArray` | full obs vector (len 14) |
| `/cara/control/action` | `Float32MultiArray` | joint targets, rad (len 7) |
| `/joint_commands` (sub) | `JointTrajectory` | joint targets the controller tracks (default `setpoint_topic`); `""` = internal demo gait |
| `/cara/sim/fault` (sub) | `Float32` | override sim fault level; <0 releases |

Node params: `source` (`sim`\|`hw`), `rate_hz` (50), `i2c_bus` (1), `ina_addr`
(0x45), `imu_addr` (0x28), `setpoint_topic` (`/joint_commands`),
`setpoint_timeout_s` (0.5).

## Migration to a learned policy

1. Train in Isaac Lab with `system_health` (or `thermal_state`) in the observation.
2. Export to ONNX.
3. Add an `OnnxController : Controller` that runs ONNX Runtime / TensorRT and
   maps the output vector to `Action::target_rad`. Swap it in `main.cpp`.
4. Nothing else in the loop changes.
