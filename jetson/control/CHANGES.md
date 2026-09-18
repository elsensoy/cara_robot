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

**What I built:** every sample now carries a `seq` (incremented once per
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

**What I built:** `ImuGuard` compares consecutive IMU samples bit-for-bit
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

**What I built:** `ImuGuard` tracks the last safety-filtered command and,
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

**What I built:** a full `PerJointPowerSource` interface, an
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

**What I built:** `PersistenceGate` (`diagnostics.hpp`) — a small,
reusable bad/good hysteresis. `ImuGuard` uses one to decide whether to hold
the last-known-good IMU sample through a short grace period or fall back to
the existing safe default; `HealthEstimator` uses a separate one for
power-telemetry validity, and once *that* trips, `system` health decays
toward a conservative floor (`stale_health_floor`, default 0.5) instead of
silently reporting whatever the last good reading said, forever.

### Revision: the gate itself was still rate-dependent

**What it exposed (a follow-up question, not a new one from the original
list):** the first version of `PersistenceGate` counted consecutive ticks —
`trip = 3`, `clear = 10` — which quietly assumes every tick takes the same
20ms this loop nominally runs at. That's exactly the fragility the timing
cluster warns about (*"suppose one iteration suddenly takes 80ms — what
happens?"*): a tick-counted gate silently redefines its own real-time
meaning whenever scheduling jitter changes how long a tick actually takes.
Three slow ticks and three fast ticks trip the same "fault" state after
very different amounts of wall-clock time — the gate's behavior wasn't
actually pinned to anything physical.

**What I built:** `PersistenceGate` now tracks *how long* bad (or good)
readings have persisted continuously, driven by the caller's own monotonic
timestamp (`ImuGuard`'s `now`; `HealthEstimator` uses the sample's own
`t_s`, which the driver sets whether or not the read succeeded) instead of
a call count. `Config` changed from `{int trip; int clear;}` (ticks) to
`{double trip_s; double clear_s;}` (seconds); defaults (0.06s / 0.20s) were
chosen to reproduce the old tick counts' behavior exactly at the nominal
50Hz rate, so this is a mechanism fix, not a behavior retune — verified by
rerunning both `--test-imu-freeze` and `--test-power-dropout` and checking
the trip/recovery timestamps in the log were unchanged.

## 6. "How are you addressing twenty INA219s on the bus?" → a real ceiling, surfaced and documented, deliberately not built

**What it exposed:** this question came from stress-testing the per-joint
current-sensing work (item 4) at 20 servos instead of Cara's current 7, and
it doesn't have a code fix — it has a wall. An INA219 has exactly 16 usable
I²C addresses (the A0/A1 pins each strap to one of 4 references, 4×4=16).
Twenty servos, one chip each, on one bus is not a configuration problem, it's
arithmetically impossible. `Ina219MultiPower` has no notion of an I²C mux or
a second bus; it simply never hits this wall at 7 joints, which is exactly
why it's easy to miss until someone asks the 20-servo version of the
question. Answering it honestly also surfaced a second, real, currently-
shipped gap: `Ina219MultiPower::read()` polls every channel sequentially in
one loop but stamps the *entire* `PerJointCurrentSample` with one shared
`t_s` taken before the sweep starts — so "how synchronized are those
measurements" has an honest answer of "not synchronized, and not labeled as
such in the data today."

**What I did about it — deliberately not a redesign:** documented, not
built, per explicit instruction not to rush a redesign before an interview
under time pressure. The right preparation for this cluster turned out to
be a three-way classification, applied consistently rather than invented
per-question:

| Claim | Status |
|---|---|
| Software interface implemented | ✅ — `PerJointPowerSource`, `Ina219MultiPower`, the per-joint `HealthEstimator` path, `SimPerJointPower` |
| Physical multi-sensor topology validated | ❌ — zero real INA219 chips have ever been wired together and read back, not even two |
| Future hardware architecture (needed at 20, not 7) | ❌ — an I²C mux or multiple buses, real shunt sizing for Cara's actual current range, per-channel timestamping: none of this is code, and isn't pretended to be |

One boundary worth recording explicitly, since it's easy to conflate with
the above: this is all about per-joint **current** sensing. Cara has exactly
one IMU for the whole body and there is no plan for more — per-joint
orientation sensing is a much harder, more expensive scaling problem than
per-joint current sensing, and the two shouldn't be implied to scale
together just because both start with "per-joint."

### Refinement: the 20 servos aren't one uniform scaling problem

**What it exposed:** the framing above treated "20 servos" as a single flat
number to fit under the 16-address ceiling. That's wrong — the 20 DoF in
the joint table (`README.md`'s topology table: waist 3, neck 3, shoulders
6, hips 6, ears 2) aren't equally relevant to either per-joint current
sensing or the IMU-based motion cross-check, and treating them as
equivalent overstates how hard the scaling problem actually is.

- **Waist (3) + hips (6) = 9** are the clear must-instrument set: the
  highest-load joints, the direct balance actuators, and exactly the
  segment the IMU's projected-gravity/angular-velocity signal is actually
  about. This is also where `ImuGuard`'s commanded-vs-measured motion
  cross-check is meaningful — a hip or waist joint moving is precisely what
  should (or shouldn't) show up as trunk motion.
- **Ears (2)** are the clear opposite case, and this isn't a guess — the
  ear-inertia study already measured **zero** effect on whole-body standing
  tilt or the weight-shift envelope. An ear twitch doesn't perturb trunk
  orientation, so gating it through the same IMU-based motion cross-check
  used for a hip would be a category error, not just unnecessary caution.
- **Neck (3)** is genuinely mixed, not a clean yes/no: neck motion isn't a
  balance actuator the way a hip is, but head mass and position do move the
  whole-body center of mass in the dynamics model, so it can't be waved off
  the same way ears can.
- **Shoulders (6)** sit in between — moderate load, not directly
  balance-actuating, but not clearly cosmetic either.

**What this changes about item 6's answer:** the address-ceiling problem
was overstated. Waist+hips alone (9) fits comfortably under 16; even adding
shoulders (15) still fits on one bus with no mux at all. Ears are the clear
candidate to exclude from per-joint current sensing entirely — small,
low-torque, already shown not to matter to the signals this sensing exists
to protect. The honest revision isn't "here's how I'd build the mux," it's
that grouping by **functional/balance relevance** rather than by servo
count might make the mux question moot, depending on where the line gets
drawn for neck and shoulders. Still not built — the classification table
above still holds — but the "future hardware architecture" bucket now has
a much narrower actual problem in it than "address 20 things."

One number worth flagging rather than assuming: the original 20-DoF
breakdown has no separate "eyes" DoF — vision is a fixed USB camera, not an
actuated eye servo, per the hardware table and the joint topology table
both. If eye servos exist in a newer design than what these docs describe,
that's worth confirming rather than carrying forward an assumption.

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
- `Ina219MultiPower::read()` timestamps the whole per-joint sweep once, not
  per channel — fine for a rough health signal, not honest if anything ever
  needs to correlate a current spike to a specific commanded event.
- Per-joint current sensing is I²C-address-limited to 16 channels per bus
  (the INA219's own A0/A1 addressing range) with no mux/multi-bus support —
  a hard ceiling for any future 20-servo build, not just a wiring
  inconvenience at today's 7.
