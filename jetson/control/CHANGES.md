# Sensor diagnostics hardening — what changed, and why

**Origin:** this work didn't start from a design review — it started from
prepping answers to a SpaceX-interview question set about Cara's sensor
architecture (`cara-questions-spacex.txt`). Answering those questions
honestly meant reading the actual pipeline code line by line instead of
describing what it was supposed to do, and several questions had no good
answer in the code as it stood. Each of those became a concrete, scoped
piece of work instead of a vague "add more robustness" gesture. This file
records which question exposed which gap, and exactly what was built in
response, so the reasoning stays attached to the code and doesn't have to
be reconstructed later from the diff alone.

Everything below lives in `jetson/control/` (the shared C++ pipeline) and
`ros2_ws/src/cara_control/` (the ROS 2 wrapper around it).

---

## 1. "What if the sensor returns an old value?" → timestamp/sequence monitoring

**What it exposed:** `ImuSample`/`PowerSample` carried a timestamp, but
nothing ever checked it. A read that returned a technically-valid but
stale value would flow straight through — staleness wasn't a first-class
check, it was an absent one.

**What we built:** every sample now carries a `seq` (incremented once per
read call, good or bad), and `ImuGuard` checks sample age against
`max_age_s` as one of its inputs, alongside the read's own `valid` flag.
Age is now something the pipeline actively looks at, not something that
happens to be recorded and never consulted.

## 2. "How can you tell 'sensor froze' apart from 'robot isn't moving'?" → derivative-plausibility check

**What it exposed:** this one had no answer at all. A BNO055 returning a
technically-successful I²C transaction with the exact same bytes as last
time — a stuck register, a wedged bus, a cached value — looked identical
to a real, valid reading of a robot standing still. Nothing distinguished
them.

**What we built:** `ImuGuard` compares consecutive IMU samples bit-for-bit
(roll/pitch/yaw/gyro). Real sensor noise doesn't repeat exactly — even a
motionless robot's IMU output has quantization dither in the low bits — so
several identical readings in a row (`frozen_repeat_trip`, default 5
consecutive ticks) gets flagged as suspected-frozen rather than trusted as
"genuinely still."

## 3. "Cross-check IMU-implied motion against commanded motion" → the second, independent signal

**What it exposed:** the servo command sent out every tick is itself a
prediction of how the robot should be moving, and nothing compared it
against what the IMU actually reported. A mismatch — commanded to hold
still, IMU insists otherwise — was invisible.

**What we built:** `ImuGuard` tracks the last safety-filtered command and,
once it's been holding still for `still_hold_s`, checks whether the IMU's
angular-velocity magnitude stays above `resting_wobble_rad_s`. Sustained
disagreement between "what we told it to do" and "what it says it did" is
now a checked condition, not an assumption that they always agree.

## 4. "Per-joint current sensing, replacing the aggregate-only placeholder" → the plumbing that was waiting for it

**What it exposed:** `HealthState::per_servo` and `per_servo_valid` already
existed in the code specifically as a placeholder — mirroring the
aggregate scalar, flagged `false`, with comments marking exactly where
real per-joint data should plug in once it existed. It was an honest,
named gap, not a hidden one — but it was still a gap.

**What we built:** a full `PerJointPowerSource` interface, an
`Ina219MultiPower` hardware driver (one INA219 per servo, opt-in via
`--per-joint-addrs` / the `per_joint_addrs` ROS param), and
`SimPerJointPower` so the path is exercised by default in `--sim` with no
hardware at all. `HealthEstimator` now runs a real per-joint EMA and
current-to-health mapping when a per-joint source is present, and flips
`per_servo_valid = true` — the placeholder's own contract, honored.
Per-joint thresholds are provisional (not bench-calibrated, same caveat the
aggregate thresholds once carried) — marked as such in the code, not
presented as final numbers.

## 5. "A persistence-threshold state machine, not reacting to a single dropped read" → replacing binary valid/invalid everywhere

**What it exposed:** every fault path in the original code reacted to
exactly one bad sample: `HealthEstimator` held its last estimate forever
on any invalid read with no escalation, and `ObservationBuilder` reset
straight to a hardcoded neutral on the very first invalid IMU sample, with
no distinction between "one transient blip" and "this sensor is actually
gone."

**What we built:** `PersistenceGate` (`diagnostics.hpp`) — a small,
reusable consecutive-bad/consecutive-good hysteresis. It trips `fault`
after `trip` consecutive bad ticks (default 3) and clears back to `ok`
only after `clear` consecutive good ones (default 10) — deliberately
asymmetric: quick to distrust, slower to trust again. `ImuGuard` uses one
to decide whether to hold the last-known-good IMU sample through a short
grace period or fall back to the existing safe default; `HealthEstimator`
uses a separate one for power-telemetry validity, and once *that* trips,
`system` health decays toward a conservative floor (`stale_health_floor`,
default 0.5) instead of silently reporting whatever the last good reading
said, forever.

---

## Files touched

**Shared pipeline (`jetson/control/`)** — single source of truth, compiled
into both the standalone `cara_control` binary and the ROS node:

| File | What changed |
|---|---|
| `include/cara_control/types.hpp` | `seq` on `ImuSample`/`PowerSample`; new `PerJointCurrentSample`, `ImuDiagnostics`; `HealthState::telemetry_state` |
| `include/cara_control/diagnostics.hpp` (new) | `PersistenceGate`, `ImuGuard` |
| `src/diagnostics.cpp` (new) | `ImuGuard::update()` — staleness + frozen-data + motion-mismatch, gated |
| `include/cara_control/pipeline.hpp` / `src/pipeline.cpp` | `HealthEstimator` gains a telemetry `PersistenceGate`, stale-floor decay, and the per-joint EMA path |
| `include/cara_control/sources.hpp` | `PerJointPowerSource` interface; `makeIna219MultiPower`, `makeSimPerJointPower`, `makeNullPerJointPower`; sim test-injection hooks |
| `src/sources_hw.cpp` | `Ina219MultiPower`; shared `ina219Init`/`ina219ReadInto` (de-duplicated from `Ina219Power`); `seq` on both hardware drivers |
| `src/sources_sim.cpp` | `SimPerJointPower`, `NullPerJointPower`; `--test-imu-freeze`/`--test-power-dropout` injection, reusing the existing fault-timeline cadence |
| `include/cara_control/i2c_device.hpp` | added a move constructor/assignment (needed so `Ina219MultiPower` can hold a `std::vector<I2CDevice>`) |
| `src/main.cpp` | wires `ImuGuard` + per-joint source into the loop; `--per-joint-addrs`, `--test-imu-freeze`, `--test-power-dropout` CLI flags; extended log line |
| `CMakeLists.txt` | added `diagnostics.cpp` to the build |
| `README.md` | new "Diagnostics" section, updated per-servo table, updated flags/components |

**ROS wrapper (`ros2_ws/src/cara_control/`)** — wired to match `main.cpp`:

| File | What changed |
|---|---|
| `src/cara_control_node.cpp` | `ImuGuard` + `PerJointPowerSource` in the constructor and `step()`; new `per_joint_addrs` param; new topics: `/cara/imu/state`, `/cara/health/telemetry_state`, `/cara/health/per_servo`, `/cara/health/per_servo_valid`; `RCLCPP_WARN_THROTTLE` on frozen/mismatch/telemetry-fault |
| `CMakeLists.txt` | added `diagnostics.cpp` to the build |
| `launch/cara_control_sim.launch.py` | documented `per_joint_addrs` default (empty) |

---

## What was actually verified, and what wasn't

**Verified:** both configurations of the standalone pipeline
(`CARA_WITH_HARDWARE=0` and `=1`, the latter syntax-checked against a
minimal `i2c/smbus.h` stub since this machine has no real INA219/BNO055
attached) compile clean under `-Wall -Wextra`. The `--sim` binary was run
directly: normal operation is unchanged from before this work, and
`--test-imu-freeze` / `--test-power-dropout` were both run end-to-end —
the `imu`/`tlm` log columns correctly escalate to `fault` and recover on
their own once the injected fault window closes.

**Not verified:** `cara_control_node.cpp` (the ROS wrapper) — this
environment has no ROS 2 / rclcpp installed, so that file was reviewed
line-by-line against patterns already proven elsewhere in the same file
(the existing `RCLCPP_WARN_THROTTLE` usage, the existing
`Float32MultiArray` assign-from-`std::array` pattern) but not actually
built. No real hardware exists to test `Ina219MultiPower` or the real
`Bno055Imu` frozen-data path against; only the sim-side equivalents were
exercised. Per-joint current thresholds are provisional, not bench
measured.

## What's still an open, honestly-named gap

- BNO055 axis-mapping/mounting calibration — still assumed, not verified
  (the `AXIS_MAP_CONFIG` register is still never touched).
- The motion-mismatch thresholds (`still_cmd_eps_rad`, `resting_wobble_rad_s`,
  hold times) are reasoned defaults, not tuned against real IMU noise on
  real hardware.
- No cross-check yet between the *aggregate* and *per-joint* current
  readings when both are present (e.g. per-joint sum vs. aggregate,
  as an internal consistency check on the wiring itself).
- The ROS node's `per_joint_addrs` param has no `DeclareLaunchArgument`
  wrapper (array-valued launch CLI args are awkward in ROS 2 launch) — it's
  settable via a params file, not a `ros2 launch ... per_joint_addrs:=...`
  command-line override.
