# Answers to cara-questions-spacex.txt

Direct answers, in the original file's order, written the way I'd actually
say them out loud — first person throughout, because the prompt for this
conversation is specifically projects I made significant technical
contributions to. Where the honest answer is "that's a gap," I say so
plainly rather than invent robustness Cara doesn't have; where I closed a
gap while preparing this, I say what I actually did about it, not "we."

Grounded against the real code in `jetson/control/`, `arduino/`, `ros2_ws/`,
and `cara_description/` — see `jetson/control/CHANGES.md` for the full,
dated account of what changed and why while I was preparing these answers.

---

## The six framing questions

**"Tell me about Cara."**

Cara is a 20-DoF companion-robot project. I've built two things toward it.
The first is a working embedded control rig — a Jetson driving 7 servos
through an Arduino, with a BNO055 IMU and an INA219 current/voltage sensor
feeding a health estimate into a hand-written controller. The second, and
the deeper of the two, is a simulation track in MuJoCo where I built the
robot's kinematic and dynamic model from a single YAML source, validated it
stage by stage — standing, weight-shift, single-support balance, stepping —
then attempted a real dynamic walk with a capture-point controller. That
walk doesn't fully close yet; I root-caused why, and I'm now building an RL
environment on top of the validated model instead of continuing to hand-tune
the walking controller.

**"What did you personally build?"**

On the embedded side: the C++ signal pipeline in `jetson/control/` — a raw
Linux i2c-dev/SMBus wrapper, BNO055 and INA219 drivers, an EMA-based health
estimator, a hand-written gain controller, a rate-limiting safety filter,
and a sim/hardware source abstraction so the identical loop runs on a
laptop or on Cara. I also just went back through that pipeline and added a
telemetry-trust layer on top of it — staleness monitoring, a frozen-data
detector, a commanded-vs-measured motion cross-check, and per-joint current
sensing plumbing — which I'll get into below, because reviewing my own
sensor code that closely for this interview is exactly what surfaced most
of it. On the simulation side: the YAML→URDF/MJCF generation pipeline, the
kinematics/dynamics validators, the balance and stepping controllers, the
capture-point walking model, and the investigation that isolated a
saturation-induced limit cycle in the ankle's torque loop as the actual
blocker to dynamic walking.

**"Walk me through the architecture."**

Two loops. The physical loop runs at 50 Hz: the BNO055 (orientation +
angular velocity) and the INA219 (servo-rail current/voltage) are read over
I²C each tick; the IMU reading now passes through a guard I added that
checks staleness, frozen data, and commanded-vs-measured consistency before
anything downstream trusts it. An EMA turns the power reading into a health
scalar; an observation builder assembles a fixed-layout vector; a
hand-written controller scales gait amplitude by health and IMU wobble; a
safety filter clamps to joint limits and rate-limits the step; the result
goes out as a short serial command to an Arduino Nano, which re-clamps and
writes PCA9685 registers. That whole loop is single-threaded and
synchronous. The ROS 2 loop is a set of independent nodes — IMU publisher,
body/policy node, actuator node, a second PCA9685 node for head gaze, a
health monitor — talking over topics at their own rates, with one explicit
staleness guard on the command topic (0.5s timeout, falls back to neutral).

**"What runs on the Jetson versus the Arduino?"**

The Jetson does everything that thinks: I²C sensor reads, filtering, the
controller, the new fault-detection layer, the ROS 2 graph. The Arduino
Nano does nothing that thinks. It parses one line of serial text, clamps it
to that joint's hard min/max from a local table, and writes a PCA9685
register. No sensing, no state, no decisions on the Arduino at all.

**"Why did you split it that way?"**

Blast radius. Every failure mode I can think of on this system — a ROS node
crashing, an EMA diverging, a bad policy output — happens upstream of the
Arduino. Putting the joint-limit clamp on the very last piece of hardware
before the actuator means even if everything above it is broken, the
Arduino still refuses an out-of-range angle. It's also a good fit
practically: the PCA9685 is a dedicated I²C PWM chip with its own
oscillator, so once it's told a pulse width it free-runs that channel at
50 Hz on its own — Jetson-side timing jitter doesn't directly become PWM
jitter.

**"What was the hardest technical problem?"**

Not on the hardware — in the MuJoCo dynamic-walking controller. I built a
capture-point (DCM) controller on top of a linear-inverted-pendulum model
of Cara, and the planner was provably correct, but the walk toppled on the
second forward step no matter how I retuned gains. I chased two wrong
hypotheses first — the swing-foot target, then the gait-initiation
amplitude — before instrumenting per-substep error logging and isolating
the actual fault: reproducing it against a perfectly static ankle target
with zero swing motion, which ruled out everything I'd been tuning. The
real cause was a discrete relay limit cycle in the ankle's torque loop once
it saturated — every damping gain in a whole range produced identical
chatter, and only dropping toward an analytically-derived torque-headroom
bound broke it, and only while that foot was unloaded. I validated the fix
but deliberately didn't wire it into the live controller before pivoting to
building an RL environment, so I'd call it a real, closed root-cause with
an intentionally-parked fix, not a finished feature.

---

## Architecture

**Why did you choose the BNO055?**

It does sensor fusion onboard — accelerometer, gyro, and magnetometer
combined into a fused orientation estimate on the chip's own MCU, output
over I²C as Euler angles, a quaternion, gyro, and linear acceleration. I get
a usable orientation without writing and tuning my own attitude filter.

**Why use I²C?**

It's a 2-wire bus that lets multiple low-bandwidth devices share one bus at
different addresses — my BNO055 (0x28) and INA219 (0x45) sit on the same
bus. The data rates here are nowhere near what would push me to SPI, and
Linux has native support (`/dev/i2c-N` plus SMBus ioctls) so I didn't need
a custom driver.

**How does an I²C transaction actually work?**

The master addresses the target (7-bit address plus a read/write bit), the
target ACKs. For a register read I first write the register pointer as a
normal write transaction, then issue a repeated start and read back N
bytes, ACKing each byte except the last, then stop. My `I2CDevice` wrapper
does this through SMBus calls — a single-byte read for status registers,
a block read for actual telemetry.

**What is the BNO055's I²C address?**

`0x28` — COM3 pulled low, the default everywhere I use it. If I ever added
a second one it would need COM3 pulled high for `0x29`.

**How are registers read?**

A single-byte SMBus read for things like the chip-ID/status registers, and
a block read (auto-incrementing on the device) for actual telemetry.

**Why do a burst read rather than individual reads?**

Two reasons. Coherence: Euler heading/roll/pitch are three consecutive
16-bit registers, and a burst read gets all six bytes as one snapshot
instead of three separate transactions that could straddle an internal
register update and tear. And overhead: every discrete transaction pays
address-plus-register-pointer-plus-repeated-start latency, so one 6-byte
block read replaces what would be six single-byte round trips. I do exactly
two block reads per sample — Euler, then gyro.

**What data are you retrieving?**

In the C++ pipeline: Euler angles (roll/pitch/yaw) and gyro (angular
velocity), 12 bytes across two burst reads. There's a separate, simpler ROS
node (`cara_imu_node.py`) that instead reads quaternion, gyro, and linear
acceleration to publish a standard `sensor_msgs/Imu` — I'd be upfront that
those are two independent drivers reading different registers, not one
shared abstraction yet.

**What units does the device return?**

Euler: 1/16 degree per LSB. Gyro: 1/16 degree/s per LSB. Quaternion: Q14
fixed point, 1/16384 per unit. Linear acceleration: 1/100 m/s² per LSB. I
convert Euler and gyro to radians and rad/s in code.

**What coordinate frame is that data expressed in?**

The BNO055's own internal axis convention — there's a register
(`AXIS_MAP_CONFIG`) that lets you remap it, and I never touch it, so I'm
running on the power-on default. I assume the sensor frame is the robot
body frame with no extrinsic mounting rotation. That's a real, named
assumption, not something I've verified with a calibration step.

**How do you transform it into the robot's frame?**

From roll and pitch I build a body-frame projected-gravity unit vector
directly: `g = [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]`.
Standard, but it inherits the assumption above — no independent check that
the physical mounting actually matches it.

**Why did you sample at 50 Hz?**

It's the one number that's consistent end to end — the C++ pipeline's
default rate, the ROS body node's control timer, and the MuJoCo RL
environment's control frequency are all 50 Hz. That's deliberate: a policy
trained in sim should see the same control frequency it'll see on hardware.

**How did you verify you were actually achieving 50 Hz?**

Honestly — I haven't instrumented this. The loop computes its own elapsed
time and sleeps to the next scheduled tick, but there's no logged histogram
of actual achieved period or jitter. I'd rather say that plainly than claim
a verification I haven't done. A production version would log real
inter-sample intervals and flag deviation past some tolerance.

**What happens if a read takes too long?**

The loop doesn't try to catch up. It schedules the next tick as a fixed
offset from the last one; if it's already past that when it checks, it
resyncs to "now" instead of sleeping a negative duration, trading one
skipped update for not compounding delay into every future tick. This
matters more than it used to, because I recently made my fault-detection
gate duration-based instead of tick-counted specifically so a slow tick
doesn't quietly redefine what "three bad readings" means in wall-clock time
— more on that under the timing questions below.

**What happens if an I²C read fails?**

Each driver call is independently wrapped in try/catch; a failed read logs
and returns a sample marked invalid. That invalid flag now feeds a
persistence-gated fault check I added rather than being reacted to on its
own — one bad read holds the last good value through a short grace period;
only sustained failure escalates to an actual fault state that the rest of
the pipeline treats differently.

**What happens if the sensor returns an old value?**

This is the one where reviewing my own code for this interview actually
changed something. While going through the pipeline I realized my
timestamp existed on every sample but wasn't actually part of the validity
decision anywhere — a technically-successful read with a stale timestamp
would flow straight through. I added explicit freshness monitoring: every
sample's age gets checked against a max-age bound as one of the inputs to
the same fault gate that watches read failures, so "old but technically
valid" is now a checked condition instead of an invisible one.

**How can you tell "sensor froze" apart from "robot isn't moving"?**

The harder case was a successful bus transaction returning plausible but
stale data — a stuck register or a cached value looks electrically
identical to a real reading of a still robot. I closed part of that gap by
comparing consecutive IMU samples bit-for-bit; real sensor noise doesn't
repeat exactly, even standing still, so several identical readings in a row
now gets flagged as suspected-frozen instead of trusted as "genuinely
still." I'd still call this partial — it catches exact repeats, not a
sensor that's frozen at a slightly-varying-but-wrong value, and it doesn't
poll the chip's own data-ready status register, which would be the more
complete fix.

**How do you detect noisy data?**

On the IMU side, honestly, I don't — I rely entirely on the BNO055's own
onboard fusion to have already smoothed it; there's no outlier rejection on
my side. On the power side there's an EMA, but that's smoothing, not
detection — a genuine outlier still gets filtered in, nothing flags it as
suspicious on the way.

**What filtering are you doing?**

Only on the power channel — a first-order exponential moving average,
separate gains for current and voltage. The IMU stream isn't filtered by my
code at all; I consume the BNO055's onboard-fused output directly.

**Why an alpha-beta filter instead of a Kalman filter?**

I'd correct the premise here rather than just answer it: what's actually in
my code is a plain EMA, not a true alpha-beta filter — an alpha-beta filter
tracks a two-state model, position and velocity, with fixed gains on both
error terms; my EMA only smooths one scalar with one gain. I used it
because the power channel doesn't need a state estimate, just a slow-moving
trend for a threshold decision, and an EMA is one line with an obvious time
constant. If I did need to track position and velocity as cheaply as
possible without a Kalman filter's covariance propagation, alpha-beta is
the right middle ground — I don't need it here because the BNO055's onboard
fusion already is a Kalman-style estimator running on the sensor's own
silicon.

**How did you choose the filter parameters?**

Empirically, off a bench power-monitor test script, not derived from a
noise-model calculation.

**What latency does filtering introduce?**

For a discrete EMA, the effective lag scales with `(1-alpha)/alpha`
samples. At my current-channel alpha and 50 Hz, that's roughly one sample —
about 20ms to meaningfully settle toward a step change. Small, but not
zero.

**What happens to your controller if IMU latency grows?**

My controller uses angular velocity directly as a stability signal, so a
stale measurement means it's reacting to where the robot was, not where it
is — added phase lag in a closed feedback loop, which past some delay
margin turns a damping term into a destabilizing one. I don't currently
measure or bound IMU sample age inside the control decision itself beyond
the staleness check I just described gating whether the sample is used at
all — there's no separate delay-margin analysis on the controller side, and
I'd name that as a real gap rather than imply I've done that analysis.

---

## Timing

**Your IMU samples at 50 Hz but your control loop operates at another
frequency. What happens between measurements?**

On the real hardware they're the same rate by construction — one loop, one
read, one compute, one write, every 20ms. On the MuJoCo side there
genuinely is a split I built on purpose: physics steps at 500 Hz, and one
RL policy decision holds for 10 physics substeps before the next
observation. Between policy decisions, the actuators just keep tracking
whatever target was last set — the physics engine integrates forward on its
own faster clock.

**Suppose one iteration suddenly takes 80 ms. What happens?**

No watchdog treats 80ms as anomalous versus merely late — the loop resyncs
its schedule to now and moves on. Where this actually matters now: I
recently rewrote my fault-detection gate from counting consecutive ticks to
tracking elapsed wall-clock time, specifically because a tick-counted
policy silently redefines its own meaning when a tick takes 80ms instead of
20ms — three slow ticks and three fast ticks used to trip the same "fault"
state after very different amounts of real time. Now it's timed against a
monotonic clock, so an 80ms iteration is handled correctly rather than
quietly miscounted.

**Are your loops synchronous or asynchronous?**

The C++ pipeline is synchronous and single-threaded — read, estimate,
build observation, compute, safety-filter, write, every tick, blocking on
each call. The ROS 2 graph is asynchronous between nodes — separate
processes on separate timers, talking over topics with no shared clock —
but each node's own callback is synchronous.

**How do you timestamp measurements?**

At the moment the driver's read call returns, using a monotonic clock —
not at the moment the chip physically latched the sample. That's a real
simplification: the BNO055 driver doesn't read any onboard latch timestamp,
so it's assuming I²C read latency is small and roughly constant.

**Do you use measurement time or processing time?**

Processing time, explicitly — see above.

**What happens when perception, telemetry, and actuation operate at
different rates?**

The clearest real example: my body node's control tick runs on its own
50 Hz timer and reads whatever the independently-rated IMU subscription
last wrote, with no freshness check on that specific path. If that topic
were slow or bursty, the control tick would silently use however-old that
value is — an unguarded shared-state pattern between two independently
clocked callbacks that I haven't closed everywhere yet, even though I've
closed the equivalent gap in the core C++ pipeline.

**How do you prevent stale data from being consumed?**

Mostly, until recently, I didn't. The one exception before this pass was a
single guard on the ROS command topic — if it goes stale past half a
second, the controller falls back to neutral instead of replaying an old
setpoint forever. I've now generalized that idea into the core sensor path
too: both the IMU and the power telemetry go through duration-based
persistence gates before anything downstream trusts them.

**Would you use threads? Why or why not?**

I haven't needed them on the embedded side — the per-iteration work fits
comfortably inside a 20ms budget single-threaded, and single-threaded means
no locking and no shared-state races to reason about, which matters most
exactly where I most need to trust the telemetry path. Where I do have
concurrency is the ROS 2 graph, and that's process-level, not thread-level
— separate nodes over topics, which avoids data races by construction but
replaces them with the staleness problem above.

**What shared data exists between your nodes?**

Nothing shared in memory — ROS 2 nodes are separate processes. The closest
thing to "shared state" is whatever's most recently arrived on a subscribed
topic, which is really per-node cached state updated asynchronously by
callbacks, not shared state in the concurrency-hazard sense.

**What happens if two components access the same state simultaneously?**

Within one process it doesn't arise — everything's single-threaded. Across
processes there's no shared memory to race on; the closest analogue is two
publishers both trying to command the same servo topic, which I avoid
structurally by giving each hardware output exactly one owning node, rather
than by locking.

---

## Hardware

**Why PCA9685 rather than driving servos directly?**

It's a dedicated I²C PWM chip with its own oscillator — I write a pulse
width once over I²C and it free-runs that channel at 50 Hz indefinitely.
Neither the Arduino nor the Jetson has enough clean hardware PWM channels
to drive seven-plus servos directly and reliably, and offloading PWM
generation to dedicated silicon decouples PWM timing from host scheduling
entirely.

**Why two PCA9685 boards?**

What I've actually got wired today is one PCA9685 on the Arduino driving
the seven body joints, plus a second one addressed directly by the Jetson
purely for head pan/tilt — two boards for ownership reasons (body servos go
through the Arduino's safety clamp, head servos need lower latency than the
serial round-trip allows), not because seven-plus-two channels exceeded one
board's sixteen. At real scale the actual reason you'd need multiple boards
is the address space — each board only exposes sixteen channels per I²C
address, and you'd strap the address pins to run a second one on the same
bus.

**How does PWM command a servo? What does the pulse width represent?**

Standard RC convention: a roughly 20ms period with a high pulse somewhere
in the 500-2500 microsecond range encoding a target angle, not power — the
servo's own electronics decode pulse width into a position command and
close the loop to it internally.

**What happens if your commanded joint angle exceeds physical limits?**

It's clamped twice, independently, in two different languages on two
different machines: the Jetson clamps to the joint table in radians before
the command ever leaves it, and the Arduino clamps again to the same limits
in degrees before writing the PCA9685 register. That's deliberate
defense-in-depth so a bug upstream can't physically overdrive a joint.

**Why are the servos powered separately from the Jetson?**

Servo stall and inrush current, and the switching noise that comes with it,
can sag voltage or inject transients well outside what the Jetson's own 5V
budget is designed to absorb. Isolating the supplies keeps a servo-side
electrical event from becoming a compute-side brownout.

**Why do they need a common ground? What would happen without it?**

Without a shared 0V reference, the logic level the PCA9685 sees isn't
referenced to the same ground the rest of the circuit uses — PWM edges and
I²C signal integrity effectively float relative to each other, which is
exactly what causes unstable PWM and IMU noise.

**Why did you add a large capacitor?**

Bulk capacitance at the servo rail buffers the near-instantaneous current
spike of stall or simultaneous-start current, which a BEC's own control
loop can't respond to fast enough on its own — the capacitor sources that
spike locally instead of it appearing as a voltage sag on the rail.

**What happens when several servos start simultaneously?**

Instantaneous current sums toward roughly stall current times the number
of servos moving at once — the real worst-case sizing scenario, not
steady-state holding current. That's exactly the scenario my health signal
exists to see: current ramping and voltage sagging together, driving the
health estimate down and the controller's gain down in response.

**What's servo stall current? How would you estimate your maximum current
requirement?**

I don't have a per-component number for this — the thresholds I do have in
code are empirically bench-measured for the *aggregate* rail, not derived
from a per-servo datasheet spec. I haven't done the analytical sizing
(worst-case stall current times the number of servos that could plausibly
move at once, plus BEC-efficiency margin), and I'd rather say that directly
than back-fill a number I don't actually have.

**What would a brownout look like in software? How would you distinguish
it from a crash?**

A brownout shows up as bus voltage dropping toward my critical threshold
while the node keeps running fine and the health label escalates smoothly
— a graceful, visible degradation. A crash instead makes the whole node
disappear from the ROS graph, topics stop publishing, and because the
PCA9685 free-runs its last commanded pulse in hardware, the servos just
freeze at their last pose rather than going limp. So the distinguishing
signal is "does the health metric degrade smoothly, or does the topic go
silent" — I don't have a dedicated brownout detector, but the existing
telemetry happens to make the two look different if you're watching for it.

---

## Failure modes

**What happens if the IMU dies?**

Now, specifically: a read failure or a detected staleness/frozen-data
condition feeds the same persistence gate, and once it trips, the
observation builder falls back to a safe neutral (identity gravity, zero
angular velocity) instead of trusting garbage. Before this pass, a single
bad read reset straight to that fallback with no grace period at all; now
it holds the last good sample through a short window first, so one
transient blip doesn't immediately zero out the stability signal.

**What happens if the Arduino dies?**

Not detected. The Jetson keeps writing to the serial port regardless, and
the write's return value is explicitly discarded — the command protocol has
no acknowledgment at all, so there's no way to know commands stopped
landing. This is still a real, open gap.

**What happens if the Jetson dies?**

The PCA9685 holds the last commanded pulse per channel in hardware, with no
host refresh needed — Cara holds her last pose rather than going limp. A
useful property, but one of the chip, not something I engineered.

**What happens if one servo jams?**

Still not detectable on real hardware today. I built the software interface
for per-joint current sensing — a real driver, a real health-estimation
path — but zero physical INA219 chips have actually been wired per joint,
so this is an implemented interface without a validated physical topology
behind it yet, and I'd be careful not to blur that distinction.

**What if a servo responds correctly electrically but the mechanical joint
doesn't move?**

Fully undetectable. There are no encoders anywhere in the stack — commanded
position stands in for measured position everywhere in the code, and that
comment is literally in the source. This is probably the single biggest
structural gap in the whole system.

**What if I²C becomes intermittent?**

Better handled now than before. Each read used to be reacted to
independently — a lone failed read on the power side got silently held
forever with no escalation, while the IMU side reset all the way to neutral
on the very first bad sample, which was inconsistent between the two paths.
Both now go through the same kind of duration-based persistence gate:
brief intermittency gets absorbed through a grace period, and only
sustained intermittency escalates to an actual fault state.

**What if your IMU slowly drifts instead of completely failing?**

Still not detected, and I want to be precise about why: the frozen-data
check I added catches bit-identical repeats, but a slowly, smoothly
drifting-but-wrong reading never repeats exactly, so it sails right through
that check. Nothing cross-checks orientation against a second reference or
against commanded motion over a long horizon. This is a real, distinct gap
from the frozen-data case, not the same problem solved twice.

**What if two sensors disagree?**

Doesn't currently arise — there's no sensor redundancy at all today, one
IMU and one aggregate power rail. If I added a second one later there's no
voting or consistency logic yet to build on.

**Which failure is most dangerous?**

The mechanically-stuck-but-electrically-fine joint, specifically because of
the missing encoders — every other check in the system implicitly assumes
commanded equals actual, so that one failure is invisible to all of them at
once rather than caught by any single check.

**What's your safe state?**

Narrower than I'd like, but more real than it was a week ago: rate limiting
in the safety filter, and now a persistence-gated fallback to neutral on
sustained sensor loss instead of an unbounded hold. There's no implemented
tuck posture, torque cutoff, or E-stop logic anywhere in the running code —
those exist only as design intent in an older design document, and there's
a method in one of my ROS nodes for fall detection that's fully written but
never actually called from anywhere. I'd say that plainly rather than
imply it's wired in.

**What failures can you detect right now?**

A failed I²C transaction. Stale telemetry, timed against a monotonic clock
rather than counted in ticks. Frozen/derivative-implausible IMU data —
bit-identical consecutive readings. A commanded-vs-measured motion
mismatch — commanded to hold still while the IMU keeps reporting motion. A
stale command topic. An aggregate power excursion, mapped to a health
scalar and severity label.

**What failures can't you detect right now?**

Per-joint mechanical faults, because there's no encoder feedback and no
physically-validated per-joint current sensing yet, even though the
software interface for the latter exists. Slow sensor drift, as distinct
from frozen data. Cross-sensor disagreement, because there's no redundant
sensing to disagree. IMU mounting/axis-alignment error — I've never
verified the physical mounting matches the assumed frame. Arduino death,
because the command protocol has no acknowledgment. I'd rather list these
out plainly than let "I added a fault-detection layer" imply I closed every
gap in one pass — I closed specific, named ones.

---

## Debugging methodology

**"How would you test the controller without having the robot connected?"**

This is something I actually do, not a hypothetical — my sensor and power
sources are abstract interfaces with both simulated and hardware
implementations, and the binary takes a flag that swaps them; the identical
controller code runs either way. I recently added two dedicated sim-only
fault-injection flags specifically so I could validate the new
fault-detection logic end to end with zero hardware attached — one latches
the simulated IMU output to prove the frozen-data detector trips and
recovers correctly, the other makes the simulated power source report
failed reads to prove the same for the telemetry persistence gate. On the
MuJoCo side it goes further — every walking-controller milestone was
validated against quantitative regression checks, entirely in simulation,
before any hardware existed to test against.

**"Cara worked yesterday. Today she falls immediately to the left. What
do you do?"**

I'd walk through the actual investigation I ran on exactly this shape of
problem — a walking controller that had been passing regression checks
suddenly toppling reproducibly on a specific step.

1. **Observe** — it wasn't random; it failed at the identical point on
   every run.
2. **Localize** — checked for unintended config or model drift first; I
   keep a hard gate that generated model files stay byte-identical across
   changes, which ruled out an accidental geometry change immediately.
3. **Hypothesize** — my first guess was the swing-foot target, my second
   was gait-initiation amplitude. Both were real, testable hypotheses, and
   both turned out wrong.
4. **Instrument** — I added per-substep error logging rather than
   continuing to guess from the outside.
5. **Isolate** — I reproduced the identical failure signature against a
   perfectly static target with no swing motion at all, which ruled out
   everything upstream in one step.
6. **Reproduce deterministically** — I built a flag-gated diagnostic that
   snapshots the exact failing state and replays it under different gains
   from that identical restored point every time, so I had a controlled
   comparison instead of "run and hope."
7. **Fix** — the fix wasn't "tune harder," it was analytic: a
   torque-headroom bound explained why a whole range of gains produced
   identical behavior.
8. **Verify** — I reran the entire existing regression suite to confirm
   the fix didn't silently break something else.

On the real hardware rig, the same checklist maps onto real signals I
actually have — joint symmetry is true by construction, IMU sanity is a
check against a known-flat reference pose, supply voltage reads directly
off the health topic — but "are the left-side servos actually reaching the
commanded position" is currently unanswerable on real hardware, again
because there are no encoders. That's the one gap that would slow me down
fastest in a real version of this scenario, and I'd say so rather than
describe a check I can't actually run.

---

## Not from the original question set, but a near-certain follow-up

Ben's own note on this: *"INA219 addressing becomes a real architectural
issue at that scale. You'll likely need multiple buses, an I²C mux, or some
other topology rather than simply hanging every sensor from one bus."* He
also listed the natural next questions — max current per servo, shunt
sizing, voltage drop, whether the measurement hardware itself affects servo
performance, sampling/conversion rate, whether I can sample fast enough,
synchronization, whether I even need simultaneity, why per-servo rather
than grouped domains, and what a current anomaly tells me that position
doesn't. His advice was not to rush a redesign before the interview, but to
know precisely which parts are software interface implemented, which are
physical topology validated, and which are future hardware architecture —
and I'm keeping that distinction explicit throughout, rather than letting
any one answer bleed into the others.

**How are you addressing twenty INA219s on the bus?**

Today's code doesn't try to, and the honest reason is sharper than "that's
a lot of wiring": the INA219 has exactly 16 usable I²C addresses — its
A0/A1 pins each strap to one of four references (GND, V_S+, SDA, SCL),
giving 4×4=16 combinations. Twenty servos on one bus, one chip each, cannot
be addressed at all — that's not a configuration problem, it's an
arithmetic ceiling. My `Ina219MultiPower` driver just opens one I²C device
per address on a single bus; it has no notion of a mux or a second bus, and
it would simply fail — a real address collision, or running out of unique
addresses — if I pushed it past sixteen. My current seven-joint rig never
hits this wall, which is exactly why it's easy to miss until someone asks
the twenty-servo version of the question.

**What would actually fix it?**

My honest first answer isn't a mux — it's to ask whether I actually need
twenty independent current measurements at all. The defensible starting
design is one current-sense channel per limb or power rail:

```
Left leg rail
Right leg rail
Left arm rail
Right arm rail
Torso / head
```

Five channels, not twenty — comfortably under the sixteen-address ceiling,
no mux or second bus required. The real engineering question isn't "how do
I wire twenty sensors," it's "do I actually need twenty measurements to
localize a useful fault," and until proven otherwise I think the answer is
probably not. If a rail's current spikes while the controller is heavily
commanding one joint on that rail, that's already strong diagnostic
evidence without per-joint granularity. I'd only reach for finer
granularity — and then an I²C mux or multiple buses to support it — if a
real fault case later shows up that a rail-level reading can't localize.
That's a more defensible engineering progression than maximizing sensor
count up front, and it's a materially different, better answer than "I'd
need a mux for twenty."

**But not all twenty servos are the same problem in the first place.** The
documented joint topology is waist (3), neck (3), shoulders (6), hips (6),
ears (2) — twenty total. Waist and hips, nine joints, are the highest-load,
direct balance actuators — the clear must-instrument set, and exactly the
segment the IMU's own signal is about. Ears, two joints, are the clear
opposite: 10-gram servos, and the ear-inertia study I ran already measured
zero effect on whole-body standing tilt or the weight-shift envelope — not
worth a current-sense channel, let alone per-joint attribution. Neck is
genuinely mixed: it isn't a balance actuator, but head mass and position do
move the whole-body center of mass in my dynamics model, so it isn't
cosmetic the way ears are. Once I group by function instead of treating all
twenty as equivalent, waist plus hips alone is nine of sixteen addresses;
even adding shoulders is fifteen. The mux question may simply not come up,
depending on where the line lands for neck and shoulders. One thing worth
saying plainly rather than assuming: the documented twenty-DoF breakdown
has no separate "eyes" servo at all — vision is a fixed USB camera, not an
actuated eye joint, in the design I'm working from.

**What's the maximum current through each servo? How did you size the
shunt? What's the voltage drop across it?**

None of these were independently derived for Cara, and I'd rather say that
directly. The per-joint thresholds in my code are explicitly commented as
provisional — roughly the bench-measured *aggregate* rail envelope divided
across joints, not a per-servo datasheet or bench number. The shunt itself
is whatever ships on the Adafruit INA219 breakout, read out with a register
value I inherited directly from the well-known Adafruit "32V range, gain
of 8, 12-bit, continuous" calibration constant rather than choosing it for
Cara's actual current levels. I haven't calculated the resulting shunt
voltage drop at Cara's real per-joint currents — a rough estimate is a few
tens of millivolts at a few hundred milliamps into a typical 0.1-ohm
breakout shunt, but that's back-of-envelope, not measured.

**Could your measurement hardware itself affect servo performance?**

Yes, plausibly, and I haven't quantified it. Every shunt sits in series
with its servo's actual power path — it's added resistance in the supply
line, not a passive tap. At higher currents, exactly the stall or
simultaneous-start scenario my health signal exists to catch, that
resistance produces a real voltage drop right at the servo, at precisely
the moment torque matters most. I'd flag that as a real, open risk rather
than wave it away.

**What's the INA219 sampling/conversion rate? Can you actually sample
twenty channels quickly enough? How synchronized are those measurements?
Do you need simultaneous measurements?**

The configuration I'm using implies roughly 532 microseconds per
conversion, alternating shunt and bus voltage, so a fresh combined reading
every roughly 1.1 milliseconds per chip — that's my read of a well-known
register constant, not something I've put a scope on. More importantly,
reviewing this for the interview surfaced a real gap in my own code: my
per-joint driver polls every channel sequentially in one loop, but it
stamps the entire sample with one shared timestamp taken before the sweep
starts, not a timestamp per channel. At seven channels the sequential I²C
overhead is probably a small fraction of my 20-millisecond control budget;
at twenty channels, with the same three-transaction-per-channel pattern, I
don't have a measured number either way. So the honest answer to "how
synchronized are those measurements" is: not synchronized at all today, and
not labeled as such in the data. Whether I actually need simultaneity
depends on the question I'm asking with the data — a rough per-joint health
signal tolerates a few milliseconds of skew fine; attributing a specific
current spike to a specific commanded motion event would need much tighter
synchronization than my current design provides.

**Why per-servo sensing rather than grouped current domains?**

Per-servo is what my code targets today because that's what the
aggregate-vs-per-servo distinction in my health state was built for from
the start — real fault *attribution*, not just "the rail." But once I stop
treating all twenty joints as interchangeable, the better answer isn't
"per-servo versus grouped" as one global choice, it's to group by function.
Waist and hips are the must-instrument set; ears are the clear exclusion;
shoulders and neck are judgment calls. That reframing is what turns "twenty
sensors" into "five rails, escalate only where a real fault demands it,"
which is the answer I gave above.

**What does a current anomaly tell you that servo position doesn't?**

This is the strongest answer in the cluster, because it's not
hypothetical: I have zero position feedback anywhere in the stack — no
encoders; my own commanded-position type literally has a comment saying it
stands in for measured position until Cara has encoders. Current is
actually the only signal in my entire pipeline that responds to physical
*load* rather than to commanded intent. A jammed joint commanded to move
looks electrically and positionally identical to a healthy one in every
other signal I have; it would only show up as anomalous stall current.
That's the specific, load-bearing reason per-joint current sensing is worth
the addressing headache at all, not something I'd add "for completeness."

**On not redesigning this before the interview:** I agree with that advice,
and I'd rather say so directly than scramble to sketch a mux topology under
time pressure. The useful preparation isn't a redesign — it's exactly this
three-way split, being able to say precisely which of "the interface
exists," "the topology is physically real," and "the hardware plan exists"
is true for any given piece, without letting any of the three bleed into
the others. And one boundary worth stating before it's asked as a trap:
none of this is about per-joint *orientation* sensing. I have exactly one
IMU for the whole body and no plan for more — per-joint current sensing and
per-joint IMU sensing are not the same ask, and I wouldn't want "scaling
per-joint sensing" to imply more IMUs are coming. Balance is a whole-body
property; current draw is a per-joint one; they don't scale the same way,
and even that one IMU's relevance isn't uniform across the body — it's a
signal about the trunk and legs, not about an ear twitch.

## A layered observability architecture — designed, not built

This is the reframe that actually resolves the cluster above, and it came
from a good piece of feedback: stop asking "how many current sensors," and
ask instead what independent evidence sources exist, and what
*disagreement between them* tells me that no single source can.

**The five layers, and what's actually true for each today:**

- **Per-joint command/state** — my commanded-position type and joint
  table. Real, exists today.
- **Body motion** — the BNO055 IMU, a global inertial response. Real,
  exists today, and now gated by the freshness/frozen-data/motion-mismatch
  checks I added.
- **Electrical load** — per-joint or grouped current sensing. The software
  interface is real; the physical topology is not validated on any real
  hardware yet.
- **Expected behavior** — my URDF/MuJoCo dynamics model predicting what a
  command should produce. The model itself exists in my simulation work,
  but it has never been connected to the live embedded pipeline as a
  real-time comparison. This is the one genuinely new, unbuilt piece.
- **Learned behavior** — an RL policy produces motion, but diagnostics stay
  outside it. This is already true structurally in what I've built — not
  something I still need to build.

**Triangulating command, model, and measurement is the actual new
capability.** Today I have command and measurement, but nothing computes
what a command *should* produce and checks the measurement against it. My
own dynamics model already answers that question offline; it's just never
been wired into the live diagnostics loop. The payoff is concrete. Say I
command a right knee flexion. The model expects the pelvis orientation to
shift slightly and the right foot's trajectory to change. If the IMU shows
no corresponding motion and the right-leg current spikes, my hypothesis
space becomes a jammed actuator, a stalled servo, or a detached linkage —
not "the IMU failed," which is what a single-sensor view would default to.
Or say I command the robot to hold neutral. The model expects near-zero
angular velocity. If the IMU instead shows large angular velocity while
current stays normal, my hypotheses become an external disturbance, an
unstable configuration, or a model-controller mismatch — current staying
normal is what rules out an actuator-load explanation. Three independent
sources is what makes the disagreement between them informative; two
sources just leaves me guessing which one is wrong.

**The policy is not the diagnostician, and I want to be precise that this
is already true, not a change I need to make.** My pipeline already only
flows health *into* the observation and the safety envelope —
`HealthEstimator` feeds `ObservationBuilder` feeds the controller feeds the
safety filter, one direction. Nothing in my architecture, not the
hand-written controller and not a future learned policy, is ever asked to
decide whether a servo is electrically failing. I'd rather state that
plainly than let a good architectural principle sound like a gap I just
discovered.

**The one gap this reframe actually makes concrete, and it's the most
honest thing I can say about where this stands:** my sensor-diagnostics
work and my RL environment are two codebases that don't talk to each other
at all. Everything about staleness, frozen data, and motion-mismatch
detection lives in my C++ embedded pipeline. My RL environment is a
completely separate Python/MuJoCo codebase, and its observation vector
today has no health signal in it whatsoever — twelve joint positions,
twelve velocities, projected gravity, angular velocity, a desired speed,
and the previous action. "Health-conditioned locomotion" is not a training
run away from where I am; it's an integration I haven't started.

**The roadmap that follows from that, none of it started yet:** add a
health channel to the RL observation, then inject the fault modes my
existing domain-randomization plan already gestures at — reduced actuator
strength, delayed servo response, stuck joints, increased friction, noisy
or biased IMU readings, dropped sensor samples, brownout-like actuator
degradation — and ask whether a health-aware policy degrades gracefully
instead of continuing the same gait into a fall. Only after that would I
validate any of it on hardware. Simulation fault injection, then health
estimation, then health-conditioned locomotion, then hardware validation,
in that order.

**If Ben asks directly how I'd resolve the twenty-servo question, this is
how I'd put it:** the software path now supports per-joint health, but I
haven't validated a twenty-channel current-sensing topology on the physical
robot. Rather than assume the obvious answer is twenty sensors on one bus,
my next step is determining the minimum sensing granularity that actually
localizes useful faults — starting from five grouped rails, using the model
comparison to add resolution only where a real fault case demands it. Once
that observability is established in simulation, I can inject the same
degradation modes into the RL environment and validate a health-aware
controller before adding any hardware complexity. That's not "I don't have
a solution" — it's "I know what's implemented, I know what's still
unknown, and I know how I'd resolve the unknown experimentally."
