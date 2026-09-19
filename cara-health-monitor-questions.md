# HealthEstimator — likely questions, and honest answers

Same format as `cara-questions-spacex-answers.md`: first person, grounded in
the actual code in `jetson/control/`. Where something isn't implemented, I
say so and give the plan I'd actually follow, rather than describing it as
if it exists.

The short version of my own view going in: this is a **well-behaved
heuristic, not a validated estimator**, and most of these questions are
sharp precisely because the gap between those two things is where the
interesting engineering sits.

---

## What the score is

**What does `system_health ∈ [0,1]` actually represent?**

It's a normalized, conservative margin-to-threshold score over the servo
rail's electrical state — not a probability, not a physical quantity, and
not a calibrated estimate of anything. Concretely, I compute two linear
normalizations and take the worse of the two:

```
current_health = 1 − (I_ema − I_idle) / (I_critical − I_idle)
voltage_health =     (V_ema − V_critical) / (V_nominal − V_critical)
system_health  = clamp01( min(current_health, voltage_health) )
```

So it's 1.0 when the rail is drawing idle current at nominal voltage, and
0.0 when either channel has reached its critical threshold. The thresholds
(idle 250mA, warn 2500mA, critical 3100mA; nominal 5.0V, warn 4.8V,
critical 4.4V) are bench-measured on the aggregate rail — that's the entire
empirical basis for the number.

**Why compress health into one scalar?**

Two honest reasons, one good and one merely convenient. The good one: it
gives the controller a single monotone knob with an unambiguous direction —
lower means "do less" — which means the consuming code doesn't have to know
anything about current, voltage, or thresholds. It also drops cleanly into
a policy observation vector as one more float, which matters because the
whole point of that pipeline is that a learned policy eventually replaces
the hand-written controller without the interface changing. The merely
convenient one: it was the smallest thing that could demonstrate the signal
path end to end, and I'd rather be honest that scope drove the shape here
as much as principle did.

**What information gets lost by doing that?**

Quite a lot, and I think this is the most legitimate criticism of the
design:

- **Which channel caused it.** A health of 0.4 from overcurrent and a 0.4
  from voltage sag are indistinguishable to the consumer. I keep
  `current_ema_ma` and `voltage_ema_v` as diagnostic fields alongside the
  scalar, so it's recoverable from the telemetry — but not from the number
  the controller actually consumes.
- **Which joint.** `per_servo` exists as a field but mirrors the aggregate
  scalar with `per_servo_valid = false` unless real per-joint sensing is
  wired, which it isn't on hardware.
- **Rate and direction.** 0.6 falling fast and 0.6 recovering look
  identical. There's no derivative term anywhere.
- **Dwell time / history.** A rail that's been at 0.5 for ten minutes reads
  the same as one that dipped to 0.5 a moment ago. There's no
  thermal-like accumulator, which is exactly the kind of state a real
  actuator-health model would need.
- **Failure mode.** Overcurrent from one stalled servo, from many servos
  moving at once, and voltage sag from a weak battery all collapse to the
  same scalar.

---

## The filter

**Why EMA? What's the equation?**

```
y[n] = y[n−1] + α · ( x[n] − y[n−1] )
```

A first-order IIR low-pass, run separately on current and bus voltage. I
used it because the power channel doesn't need a state estimate — just a
slow-moving trend for a threshold decision. It's one line, it has an
obvious time constant, it needs no matrices, and it costs nothing in the
control loop. If I needed to track position *and* velocity cheaply I'd
reach for an alpha-beta filter; if I needed an adaptive gain under changing
noise I'd reach for a Kalman filter. Neither is justified for smoothing a
scalar rail current.

**How did you choose α?**

Empirically, off a bench power-monitor script — α = 0.5 for current, 0.3
for voltage — not derived from a noise model or a target cutoff frequency.
I'd rather say that plainly. The reasoning behind the asymmetry is that
voltage is the slower, more structural signal (battery sag), while current
is the one I want to see respond to a stall quickly, so it gets less
smoothing. The resulting lag is roughly one sample on current at 50Hz
(≈20ms) and a few samples on voltage. If asked how I'd do it properly:
record the real rail noise floor, pick a cutoff below the fastest fault I
care about detecting and above the ripple I want rejected, and set α from
that — which I haven't done.

---

## Dynamics and stability

**What happens when health suddenly drops?**

The EMA is what keeps "suddenly" from being instantaneous — health can only
move as fast as the filtered current and voltage move, so a true step in
draw shows up over a few samples, not one. But once health has a value, the
controller consumes it immediately and proportionally, with no hysteresis
and no rate limit of its own:

```
gain = (0.25 + 0.75 · health) · stab
target[j] = gain · setpoint[j]        // neutral == 0 rad
```

So a health drop scales all commanded joint amplitudes toward neutral
posture. The only downstream damping is the `SafetyFilter`'s per-joint rate
limit (2.5 rad/s), which bounds how fast a joint can actually move to the
newly-scaled target.

**Why should health modify gait amplitude?**

Because amplitude is the variable that most directly drives the thing being
measured. If the rail is overdrawing, the useful intervention is to demand
less mechanical work, and scaling the commanded motion about neutral is the
bluntest, most predictable way to do that with no model of the gait
required. It also fails in a sensible direction: at gain → 0.25 the robot
is still holding posture near neutral, not going limp.

**Could this create instability?**

Yes, and I think this is the sharpest question on the list. There's a
closed loop here that I have not analyzed:

```
commanded amplitude → mechanical work → rail current
       ↑                                      │
       └────── gain ←── health ←── EMA ───────┘
```

Current draw depends on commanded motion, health depends on current, and
gain depends on health — so the health signal is partly a measurement of my
own control action. That's a negative feedback loop with lag in it (the EMA),
which is the classic recipe for oscillation or a limit cycle: pull back →
current drops → health recovers → push forward → current rises → pull back.
Nothing in the current design damps that on purpose. The EMA lag and the
`SafetyFilter` rate limit happen to slow it down, but neither was designed
for that purpose, and I've never swept the loop for a limit cycle. I'd call
that an unanalyzed risk rather than a solved problem.

**What prevents your safety mechanism from making things worse?**

Three things, only one of which was deliberate:

1. **The 0.25 floor is deliberate.** Gain never reaches zero. On a legged
   robot, zero motion authority isn't safe — it's collapse. A "safety"
   mechanism that removes the ability to hold posture would be actively
   dangerous, so the floor is the one part of this I'd defend as
   intentional design rather than convenience.
2. **Scaling is about neutral, not about zero torque.** Reducing gain moves
   commanded targets toward the neutral pose, which is a defined, stable
   posture — not toward slack.
3. **The `SafetyFilter` is downstream and unconditional.** Joint limits and
   rate limits apply to whatever the controller produces, including
   anything a health-driven gain change produces. The health path cannot
   command something the safety filter would otherwise reject.

What's *missing* from that list is any analysis of the loop above, and any
guarantee that the intervention converges rather than oscillates. I can
argue each piece is individually sensible; I can't yet show the closed loop
is stable.

**What happens if health recovers? Do you need hysteresis?**

Recovery is symmetric and immediate — gain rises as soon as health does,
with no dwell requirement and no separate recovery threshold. And yes, I
need hysteresis, and I don't have it where it matters.

To be precise about what exists: I *do* have time-based hysteresis in the
pipeline, but it's on **telemetry trust**, not on **health level**. The
`PersistenceGate` I added requires bad readings to persist for 0.06s before
declaring a telemetry fault and good readings to persist for 0.20s before
clearing — deliberately asymmetric, quick to distrust and slow to trust
again. Nothing equivalent guards the health scalar or the severity labels.
So if the filtered current hovers right at a threshold, the labels chatter
and the gain wobbles, and the only thing smoothing that is the EMA lag.

---

## Structure

**Why not implement discrete states such as NORMAL / DEGRADED / SAFE?**

Honest answer: I have the labels but not the states. `HealthState::label`
is computed as `ok` / `warn` / `critical` from threshold comparisons — but
I checked, and nothing in the control path consumes it. It's read in
exactly two places: a log line and a ROS `String` publisher. The controller
only ever looks at the continuous scalar. So there is no state machine
driving behavior; there's a continuous gain and an informational label that
happens to be printed next to it.

I'd also push back gently on framing it as either/or — I think the right
design is both. Discrete states give you contracts you can actually reason
about and test ("in SAFE, no new gait is initiated" is verifiable; "gain was
0.31" isn't), while a continuous scalar gives smooth modulation *within* a
state, which avoids the classic problem of a robot visibly lurching at every
state boundary.

**How do INA219 and BNO055 measurements contribute differently?**

They aren't fused, and I want to be precise about that because "health
monitor" makes it sound like they are. `system_health` is **purely
electrical** — INA219 current and voltage only. The BNO055 contributes
nowhere to the health scalar. It enters the controller through a completely
separate term:

```
stab = 1 − clamp(‖ω‖ / 3.0, 0, 0.4)
gain = (0.25 + 0.75 · health) · stab
```

So IMU angular velocity multiplies the health-derived gain but never
changes health itself, and its authority is capped at a 40% reduction. The
IMU also supplies projected gravity and angular velocity to the observation
vector independently. Two separate signals into one multiplication — not a
sensor-fusion estimate.

That separation is deliberate and I'd defend it: they're measuring
different things (electrical load vs. inertial response), they fail in
different ways, and fusing them into one number would destroy exactly the
cross-check that makes a disagreement between them informative.

**How do you distinguish a sensor problem from an actual system-health
problem?**

This was a genuine hole until recently, and closing it is most of what I
did in the last pass. The old behavior: a failed read returned an invalid
sample and `HealthEstimator` simply held its last estimate — forever, with
no escalation. A dead sensor and a perfectly healthy robot were literally
indistinguishable in the output.

Now there are two separate fields with two separate meanings:

- `label` (`ok`/`warn`/`critical`) — **how bad is the power situation**,
  assuming the reading is real.
- `telemetry_state` (`ok`/`degraded`/`fault`) — **do I trust the reading at
  all**, from the persistence gate watching read validity.

And once telemetry goes to `fault`, health stops being held indefinitely
and decays toward a conservative floor (0.5) instead of reporting whatever
the last good reading happened to say. That's the difference between "the
robot is fine" and "I haven't heard from the sensor in a while, so I'm
going to stop acting as if I know."

What still can't be distinguished: a *plausible but wrong* current reading
— a miscalibrated shunt, or a sensor reading low. That looks exactly like a
healthy rail. The cross-check that would catch it (per-joint currents
summing to the aggregate, or current disagreeing with the model's expected
load for the commanded motion) doesn't exist.

---

## The one I'd most want to answer well

**"Why should I trust your health score?"**

You shouldn't — not as a calibrated measure of actuator health, and I'd
rather say that directly than defend it as something it isn't.

What I'd claim for it is narrower and I think defensible. It's **monotone
in the right direction** — it cannot go up when current rises or voltage
sags. It's **conservative by construction** — it takes the worse of the two
channels, not an average, so a single bad channel can't be masked. It
**degrades safely rather than failing open** — the 0.25 gain floor, the
decay-to-floor on lost telemetry, and the unconditional downstream safety
filter all mean a wrong health value produces reduced motion, not unbounded
motion. And it's now **explicitly separated from the question of whether
the telemetry is trustworthy**, which is the failure mode that actually
worried me.

What I would *not* claim: that 0.4 means anything specific about remaining
actuator life, that the thresholds are validated, or that it would catch a
real servo failure — because no servo has ever actually stalled on this
robot with an INA219 attached and recording. The thresholds come from
observing a healthy rail under normal operation and picking numbers above
it. That's a plausible envelope, not a fault signature.

So: trust it as a bounded, conservative, monotone heuristic that will pull
motion back when the rail is unhappy, and not as evidence that a specific
joint is failing. The work that would upgrade it from the first thing to
the second is concrete and I know what it is — it's below.

---

## What isn't implemented, and how I'd do it

### 1. Hysteresis on health level and severity labels

Missing entirely; only telemetry trust has it. Plan: give each severity
threshold separate entry and exit values (Schmitt trigger) — e.g. enter
`warn` at 2500mA but don't leave it until 2300mA — and additionally require
a dwell time before any transition, reusing the existing time-based
`PersistenceGate` rather than writing a second mechanism. A gain rate limit
(cap on d(gain)/dt) is the other half of this, so authority can't step even
if health does.

### 2. Real discrete states, consumed by the controller

Today's labels are informational. Plan: define `NORMAL / DEGRADED / SAFE`
with explicit entry/exit conditions (hysteretic, per above) and, more
importantly, a **behavioral contract per state** — NORMAL: full authority.
DEGRADED: amplitude scaling as now, plus refuse to *initiate* new motion,
only complete what's in progress. SAFE: hold/settle to neutral, reject new
setpoints, require an explicit operator or supervisor action to exit. Then
make the controller consume the state for the discrete decisions and keep
the continuous scalar for modulation within NORMAL/DEGRADED, so behavior
doesn't jump at boundaries.

### 3. Rate and dwell information in `HealthState`

Plan: add `d_health_dt` (from the same EMA machinery) and
`time_in_current_state_s`, so a consumer can distinguish "0.6 and falling
fast" from "0.6 and recovering," and so a thermal-style accumulator becomes
possible later. This is cheap and I'd do it first — it's additive, it
breaks nothing, and it's a prerequisite for most of the rest.

### 4. Closed-loop analysis of the health→gain→current loop

Not done at all. Plan: use the existing sim source, which can already
script arbitrary current profiles, to close the loop artificially — make
simulated current a function of commanded amplitude — then sweep EMA α,
gain-response speed, and the `SafetyFilter` rate limit looking for
oscillation or a limit cycle. This is exactly the kind of thing I'd want to
find in simulation rather than on hardware, and it's the same approach that
found the ankle instability in the walking controller.

### 5. Threshold calibration against a real fault

The largest gap, and not a software problem. Plan: deliberately stall a
servo on the bench with the INA219 attached and recording; capture the real
current signature of a stall, including its transient shape, not just its
steady-state magnitude; set thresholds from that. Until that exists, every
threshold in this estimator is an envelope around healthy behavior rather
than a boundary derived from failure.

### 6. Per-joint attribution and internal consistency

Interface exists, physical topology doesn't. Plan (detailed in
`cara-questions-spacex-answers.md`): grouped current rails first — five
channels, not twenty — and a consistency check comparing the per-rail sum
against the aggregate reading, which would catch the "plausible but wrong"
sensor case that currently sails through.

### 7. Model-based expectation

The missing third evidence source. Plan: compare measured current against
what the URDF/MuJoCo model predicts the commanded motion should cost. A
current reading that's plausible in isolation but wrong for the commanded
motion is only detectable this way, and the model already exists — it's
just never been queried from the live pipeline.
