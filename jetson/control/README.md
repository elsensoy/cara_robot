# cara_control — Jetson-side health → observation → control signal path

The non-RL prototype of the deployment-side pipeline. It routes a learned-style
**actuator health signal** into a policy observation and shows, qualitatively,
that the controller reacts to it. Same loop, same `Observation`/`Action` types
run in simulation and on Cara — the only swap later is `HandwrittenController`
→ exported RL policy.

```
BNO055 ──┐
         ├──► ObservationBuilder ──► Controller ──► SafetyFilter ──► servos
INA219 ──┴──► HealthEstimator ──┘
             system_health ∈ [0,1]
```

## The `system_health` vs per-servo distinction (kept on purpose)

| | now | later |
|---|---|---|
| `HealthState::system` | **the trustworthy field** — one scalar from the aggregate servo-rail INA219 | unchanged |
| `HealthState::per_servo` | mirrors `system`; `per_servo_valid == false` | filled once telemetry allows per-joint attribution; flip the flag |
| `Observation` | appends `system_health` as the last element (`OBS_SIZE = 14`) | append the `per_servo` vector, grow `OBS_SIZE` by `NUM_SERVOS` |

`HealthEstimator::update()` and `ObservationBuilder::build()` each carry a
`LATER:` comment at the exact spot the per-servo path plugs in.

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
(default 1), `--ina-addr 0x45`, `--imu-addr 0x28`.

### What to look for in `--sim`

As simulated servo-rail current ramps 240 → 2760 mA and the rail sags
5.05 → 4.50 V, `health` falls 1.00 → ~0.12, the controller `gain` follows it
down ~0.98 → ~0.32, and the `action` amplitudes shrink (shoulder ±0.19 →
±0.06 rad) — then everything recovers. That is the behavioural response to the
health input, with no reward function involved.

## Components

| File | Role |
|---|---|
| `types.hpp` | `NUM_SERVOS`, joint table (from `arduino/main.cpp`), `Observation`/`Action` layout |
| `HealthEstimator` | EMA + threshold logic ported from `tests/cara_power_monitor.py`, mapped to `[0,1]` |
| `ObservationBuilder` | commanded joint pos + projected gravity + base ang-vel + `system_health` |
| `HandwrittenController` | dumb: scales gait amplitude by health and IMU wobble. **The seam the RL policy replaces.** |
| `SafetyFilter` | clamps to per-joint limits + rate-limits — last line before the servos |
| `sources_sim.cpp` | synthetic IMU/power/gait with a shared fault timeline |
| `sources_hw.cpp` | `Ina219Power`, `Bno055Imu` (on `tests/imu_test.cpp`'s `I2CDevice`), `SerialOutput` (`S<ch>,<deg>` to the Nano) |

## ROS 2 integration (`ros2_ws/src/cara_control`)

`ros2_ws/src/cara_control` is a thin `ament_cmake` wrapper that compiles the
pipeline sources from this directory (single source of truth) and runs them on a
50 Hz timer. Observation-only — it never commands servos — so it is safe to run
alongside `cara_stack.launch.py`.

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
