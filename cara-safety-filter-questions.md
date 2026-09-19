# SafetyFilter — likely questions, and honest answers

Same format as `cara-health-monitor-questions.md`: first person, grounded in
the actual code (`jetson/control/include/cara_control/pipeline.hpp` and
`src/pipeline.cpp`). Where something isn't implemented, I say so and give
the plan I'd actually follow.

Short version going in: this is 14 lines of code, and I found two real,
previously-unnoticed issues in it while re-grounding this file — which is
itself worth saying, because it's a better answer than pretending 14 lines
means nothing could be wrong.

For reference, the whole thing:

```cpp
void SafetyFilter::apply(float dt_s, Action& a) {
    if (!init_) { last_ = a.target_rad; init_ = true; }

    const float max_step = cfg_.max_rate_rad_s * dt_s;

    for (int j = 0; j < NUM_SERVOS; ++j) {
        const float lo = servoDegToRad(kJoints[j].min_deg);
        const float hi = servoDegToRad(kJoints[j].max_deg);
        const float target = std::clamp(a.target_rad[j], lo, hi);
        const float step   = std::clamp(target - last_[j], -max_step, max_step);
        last_[j] += step;
        a.target_rad[j] = last_[j];
    }
}
```

---

## What it does

**What does the SafetyFilter actually do, in one sentence?**

It's the last thing that touches a joint command before it leaves the
Jetson: for every joint it clamps the target to that joint's hard
min/max limits, then rate-limits how far the commanded position is allowed
to move this tick, and remembers where it left off so the next tick ramps
from there rather than from the raw target.

**Why clamp AND rate-limit in the same filter rather than two separate
stages?**

Mostly because they share the same per-joint loop and the same state
(`last_`), and splitting them wouldn't buy independence — the rate limiter
needs to know the *already-clamped* target to ramp toward, so the two are
sequentially dependent regardless of whether they're one function or two.
I wouldn't defend this as a deep design principle, just a reasonable
combination of two things that have to run in that order anyway.

**Why clamp before rate-limiting, not the other way around?**

Because rate-limiting toward an unclamped target could ramp a joint
smoothly and confidently toward a position it's never allowed to reach,
which is worse than clamping first — you'd spend time and motion budget
approaching a target that was always invalid. Clamping first means the
rate limiter's destination is always physically legal; all it's doing is
bounding how fast you're allowed to approach a legal point.

---

## The rate limit

**What's `max_rate_rad_s` and how did you choose it?**

2.5 rad/s, a single constant applied to every joint. I did not derive this
from a servo datasheet or a torque/speed curve — it's a reasonable-sounding
number, not a measured one, and I'd say that plainly rather than imply
otherwise.

**Is the same rate limit applied to every joint? Should it be?**

Yes, uniformly — `cfg_.max_rate_rad_s` is one scalar, not a per-joint
table, even though `kJoints` already carries per-joint metadata that could
hold a per-joint rate. Should it be uniform? Probably not. A hip carrying
body weight and a neck-pitch joint have very different safe angular
velocities in practice — driven by different servo specs, different
inertial loads, and very different consequences if they overshoot. Treating
them identically is the kind of simplification that's fine for a
seven-joint prototype and wrong to carry unexamined into twenty.

**What is `dt_s` here, and what happens if it's wrong?**

It's the caller's measured elapsed time since the last tick — in
`main.cpp`, the real wall-clock delta, falling back to the nominal period
only if that delta comes out non-positive. `max_step` scales directly and
linearly with it: `max_step = max_rate_rad_s * dt_s`. That means a larger
`dt_s` — a slow tick — directly widens how far a joint is allowed to move
in one step. This is intentional and correct for the ordinary case (a
slightly late tick should still let a joint travel its expected distance).
It's also the mechanism behind the next answer.

---

## Initialization and edge cases

**What happens on the very first call?**

`last_` is seeded directly from whatever's in `a.target_rad` — the *raw,
unclamped* incoming action — not from the clamped target the loop is about
to compute two lines later. In the ordinary case this is harmless, because
the very first commanded action comes from the controller tracking a
small-amplitude gait setpoint that's already well inside every joint's
range, so `last_` starts in-bounds and everything is fine from tick one.

**Could a bad first command ever pass through unclamped-in-effect?**

This is the first thing I actually found by re-reading this file closely
rather than describing it from memory, and I think it's worth stating as a
real, if narrow, gap. Walk through it: suppose the very first `Action`
this filter ever sees has `target_rad[j]` outside `[lo, hi]` for some
joint — say a controller bug, or a future learned policy producing a wild
first output. `last_[j]` gets seeded to that out-of-range value. Then, in
the same call, `target` is correctly clamped into range — but `step` is
computed as `clamp(target − last_[j], −max_step, max_step)`, so if the
distance from the bad seed to the valid target exceeds one tick's
`max_step`, the output on that very first tick (`last_[j] + step`) is
still outside `[lo, hi]`. It corrects over the next few ticks, but "the
safety filter's own initialization can emit one out-of-range value" is a
real latent bug in a component whose entire job is to never do that. The
fix is a one-line change: seed `last_` from the *clamped* target, not the
raw input. Not done yet; I'd fix it before calling this filter finished.

**What happens if the loop misses several ticks, or `dt_s` spikes?**

`max_step` grows linearly and without bound with `dt_s` — there's no upper
clamp on the allowed step size. A pathologically large `dt_s` (a clock
jump, a resumed-from-suspend process, several genuinely missed ticks) would
let a single call move a joint arbitrarily far, defeating the entire point
of a rate limiter for exactly the input it should be most defensive about.
This is the same category of gap as the “sudden 80ms tick” question
elsewhere in this project — I fixed the equivalent issue in the
`PersistenceGate` timing (making it duration-based), but I haven't yet
capped `max_step` here the same way. A one-line `dt_s = std::min(dt_s,
dt_cap)` before computing `max_step` would close it.

---

## Independence and defense in depth

**Why duplicate joint limits here when the Arduino also clamps?**

Deliberately — it's the one part of this design I'd defend unreservedly.
Every risky failure mode I can name (a bad controller output, a future
policy's wild action, an EMA that diverged, a ROS node bug) happens
upstream of the Arduino. Clamping again on the very last piece of hardware
before the actuator means even if everything above it is broken, the
Arduino still refuses an out-of-range angle. Two independent clamps, in two
different languages, on two different machines, is the point, not a
redundancy to trim.

**What guarantees the two tables — Jetson `kJoints` and the Arduino's
`joints[]` — actually agree?**

Nothing does, and I think this is the more interesting answer than the
question implies. I checked: `kJoints` in `types.hpp` and `JointLimit
joints[]` in `arduino/main.cpp` are two independently hand-maintained
tables, in two different languages, with the same seven `{channel, min,
neutral, max}` triples typed out twice. Nothing generates one from the
other, and nothing checks they match. If someone tightened a limit on one
side and forgot the other, the two clamps would silently disagree — and
because the Arduino clamps *last*, its table would win regardless of what
the Jetson thinks the limit is, without either side knowing there's a
mismatch. This is a real, currently-unmanaged risk, not a hypothetical one:
defense in depth only works if the two layers agree on what they're
defending against.

**Does the safety filter know about health, IMU, or telemetry state?**

No, by design, and I'd defend that too. It receives only the `Action`
already produced by the controller (which has already incorporated health
and IMU wobble into its gain) and `dt_s`. It has no visibility into
*why* a target is what it is — which keeps it simple enough to reason
about completely, and keeps it correct even if everything upstream of it
is compromised. A safety filter that needs to trust the same telemetry the
rest of the system might be misreading isn't a safety filter, it's another
consumer of the same possibly-bad data.

**Could the controller and the safety filter disagree, and who wins?**

They can't really disagree in the sense of competing — the filter has no
opinion about what the *right* target is, only about what's a *legal* one.
The controller proposes; the filter constrains. If the controller asks for
something outside the joint's range or faster than the rate limit allows,
the filter wins, unconditionally, every time. There's no override path.

---

## Scope — what this filter is not

**Does it know about servo torque or velocity capability, or just
position?**

Just position, indirectly, through the rate limit. There's no torque
awareness at all — nothing here knows or cares how much force is required
to hit a given rate, and the rate limit is a flat angular-velocity cap, not
a torque or current-aware one. A joint under load and the same joint
unloaded get identical treatment.

**Does it prevent self-collision or workspace violations?**

No. Every joint is clamped completely independently — there's no coupling
between joints at all, so nothing here would catch, for example, a left-arm
target and a torso target that are individually legal but would collide
with each other. This filter's entire model of "safe" is per-joint range
plus per-joint rate; it has no model of the robot's geometry.

**Does it adapt the rate limit to load or health?**

No — the health signal already scales *amplitude* through the controller's
gain before this filter ever sees the action, but the rate limit itself is
a fixed constant regardless of health, current draw, or anything else. A
degraded-health robot moving at reduced amplitude still gets the same
angular-velocity ceiling as a fully healthy one.

---

## Testing and verification

**How would you test this filter in isolation? Have you?**

Honestly: no unit tests exist for this filter, or for anything else in
this pipeline. I checked — there's no `tests/` directory in this repo at
all, despite two comments elsewhere in the codebase (`pipeline.hpp`,
`sources_hw.cpp`) referring to `tests/cara_power_monitor.py` and
`tests/imu_test.cpp` as if they exist. They don't, in the current tree.
Everything I've verified about this pipeline has been through running the
actual `--sim` binary and reading its output, not through isolated unit
tests. For `SafetyFilter` specifically, a real unit test would be
straightforward and cheap — feed it a sequence of targets and dt values
with no other machinery involved, and assert the output never exceeds
`[lo, hi]` or moves faster than `max_rate_rad_s * dt`, including
specifically the two edge cases above (a bad first value, a huge `dt_s`).
I haven't written it.

---

## The one I'd most want to answer well

**"Why should I trust this filter to actually keep the robot safe?"**

For its actual, narrow job — never let a commanded angle leave the range
the joint can physically reach, and never let it get there faster than a
fixed rate — I'd trust it, with two named exceptions I'd fix first (the
first-tick seeding bug, and the unbounded `dt_s`). It's simple enough that
I can reason about every line of it, it has no dependency on any signal
that could itself be wrong, and it sits at the very end of the chain with
no override.

What I would not claim is that it keeps the *robot* safe, full stop —
because "safe" for a legged robot is bigger than "no joint exceeds its own
range at its own rate." It doesn't know about the other joints, so it can't
stop a self-collision. It doesn't know about load, so it can't stop a rate
that's fine unloaded and dangerous under a stall. It doesn't know about the
Arduino's independent copy of the same limits, so it can't guarantee the
two tables actually agree. And there is no test suite proving any of this,
just a working demo I've watched behave correctly. I'd trust it to do
exactly what it's written to do, and I'd be careful not to let that sound
like more than it is.

---

## What isn't implemented, and how I'd do it

### 1. Fix the first-tick seeding bug

Seed `last_` from the *clamped* target on first use, not the raw input —
one line, closes the "safety filter's own init can emit an out-of-range
value" gap directly.

### 2. Bound `dt_s`

Clamp the `dt_s` fed into `max_step`'s computation to some sane ceiling
(e.g. a few multiples of the nominal period) before using it, so a missed
tick or a clock jump can't produce an arbitrarily large single step. This
is the same category of fix as the duration-based `PersistenceGate` work —
a value that's supposed to represent real elapsed time needs an explicit
sanity bound, not an implicit trust that it'll always be reasonable.

### 3. A single source of truth for joint limits

Stop hand-maintaining `kJoints` (C++) and `joints[]` (Arduino) as two
separate literals. Plan: generate both from one YAML/JSON table at build
time (or, more simply, have the Arduino read its limits over serial from
the Jetson at startup instead of hardcoding them), the same "generated,
never hand-edited" discipline already used for URDF/MJCF in
`cara_description/`. Until then, a lint step that diffs the two tables and
fails CI on mismatch would at least catch drift instead of letting it
silently ship.

### 4. Per-joint rate limits

Move `max_rate_rad_s` from a single `Config` scalar to a per-joint value in
`kJoints`, so a hip and a neck joint can carry different, deliberately
chosen ceilings instead of one number picked for neither.

### 5. Unit tests

Write the isolated test described above — no simulator, no hardware, just
`SafetyFilter::apply()` against a table of (input, dt) pairs with asserted
output bounds, specifically covering the first-call and large-`dt_s` cases
so they can't silently regress once fixed.

### 6. Whole-body / geometric awareness

The largest real gap, and not a small one: nothing here knows the robot's
geometry, so nothing catches a per-joint-legal combination that would
still cause a self-collision. Closing this means consuming the same
URDF/MuJoCo model already built in `cara_description/` as a live
constraint check — the same "connect the model to the live pipeline" gap
named in the health-monitor and observability write-ups, not a new,
unrelated one.

### 7. Load- or health-aware rate limiting

Right now health only scales amplitude, upstream, through the controller.
A more complete design would let the safety filter itself tighten the rate
ceiling as health degrades or current rises — an independent, second line
of defense that doesn't rely on the controller having applied the gain
correctly, rather than the current single-path amplitude scaling.
