# Cara — RL environment (U19) and nominal-locomotion training (U20)

Companion to [`single_support_notes.md`](single_support_notes.md), which documents
the earlier hand-designed DCM walking controller (U7–U18, **paused** — real,
root-caused ankle-instability + landing-geometry findings recorded there, fix
not wired in). As of 2026-09-13 the project's stated goal changed: **RL-readiness
is the target, not finishing the DCM controller.** This file documents that
separate workstream, starting from U19.

## U19 — the environment, frozen as a baseline

`cara_description/scripts/cara_env.py` implements `CaraWalkEnv`, a minimal
Gym-style (`reset()`/`step()`) environment with no dependency on `dcm_walk.py` —
it does not inherit DCM's prescribed footstep schedule; a policy trained on it
must learn its own stepping coordination from scratch.

**Model / configuration**
- Config: `cara_description/config/cara_full_body.yaml`, `dynamic=True` MJCF
  (the same generator (`generate_mjcf.py`) and byte-identical output path used
  everywhere else in the project — no new model, no new geometry).
- Base pose: the `stand_nominal` keyframe (qpos + matching ctrl already baked
  into the MJCF by the existing keyframe emission).
- Actuation: the model's **default position-PD actuators** (`<position>`,
  `gear=1`, `kp`/`dampratio` from `config`, `forcerange=±effort`) on all 12 leg
  joints (`{l,r}_{hip_yaw,hip_roll,hip_pitch,knee_pitch,ankle_pitch,ankle_roll}`).
  This deliberately bypasses the U15 torque-ankle/CoP-realization scheme and
  its still-parked chatter instability (see `single_support_notes.md` U18) —
  the position actuators are a separate, simpler control path with their own
  tracking behavior that has NOT yet been stress-tested under motion or load
  (only standing has been verified so far; see the training-boundary note
  below).
- Physics timestep: `model.opt.timestep = 0.002s` (unchanged project default).
- Policy interval: `control_hz = 50 Hz` → `substeps = 10` physics steps per
  environment `step()` call (matches the "50Hz control" note already in the
  README's RL section).
- Episode length: `episode_seconds = 10.0s` → `max_steps = 500` per episode
  (config default; the self-test and first training run below use shorter
  overrides).

**Action space** — `Box(-1, 1, shape=(12,))`, one entry per actuated leg joint,
same order as `leg_model.actuated_joint_names(spec)`. An action `a` maps to a
joint-position target via
```
target = clip(nominal + a * offset_scale, joint_lower, joint_upper)
offset_scale = action_range_frac * (joint_upper - joint_lower) / 2   # action_range_frac = 0.30, provisional
```
`target` is written directly to `data.ctrl` (the position actuators' own PD
tracks it — no separate low-level controller is layered on top).
`action_range_frac = 0.30` is a **provisional** bound on how far the policy
may move a joint from nominal, chosen to keep early-training actions inside a
plausible stepping range without saturating joint limits immediately.
`# TODO: replace with measured/CAD-derived actuator speed/torque limits once
real servos are chosen — 0.30 is not derived from hardware.`

**Observation space** — `Box(-inf, inf, shape=(43,))`, concatenation of:
- 12 actuated joint positions (`qpos`), 12 actuated joint velocities (`qvel`)
- 3 projected-gravity components (world `-Z` expressed in the pelvis frame —
  the standard IMU-orientation proxy; available on real hardware via an IMU)
- 3 base angular velocity components (`qvel[3:6]`, body frame — an IMU gyro
  reading; available on real hardware)
- 1 desired forward velocity (currently a **fixed** scalar, `desired_vx = 0.10
  m/s`; becomes a commanded input once speed variation is introduced)
- 12 previous action components

Nothing in this list requires privileged simulator state (no ground-truth
contact forces, no absolute world position) — everything is either directly
computable from an IMU + joint encoders, or is the policy's own memory. This
was a deliberate choice so the observation is hardware-transferable in shape,
though the transfer itself is not attempted yet.

**Reward** (per control step, weights in `CaraWalkEnvConfig`):
```
r = w_vel * (-|vx_measured - vx_desired|)
  + w_upright * (projected_gravity_z + 1)          # 0 upright, -2 fully inverted
  + w_effort * (-mean(action^2))
  + w_action_rate * (-mean((action - prev_action)^2))
  + w_collision * (-1 if a non-foot geom touches the floor else 0)
w_vel=1.0  w_upright=0.5  w_effort=0.01  w_action_rate=0.01  w_collision=1.0
```
`vx_measured` is the pelvis subtree CoM x-velocity over the control step.

**Termination** — `terminated` (a real failure, ends the episode with a fixed
`-10` penalty): base tilt (max(|roll|,|pitch|)) `> 40°`, OR pelvis height
`< 0.15m`, OR a non-finite `qpos`/`qvel` (invalid simulator state).
`truncated` (episode timeout, `step_count >= max_steps`): **no penalty** —
this is deliberately kept separate from `terminated` per the training-boundary
check below, since conflating them would teach the policy that running out
the clock is a failure.

## Training-boundary check (done before any training run)

Per the review guidance, checked explicitly rather than assumed:

1. **Timeout vs. fall are distinguished.** `terminated` and `truncated` are
   two separate booleans (`gymnasium`'s 5-tuple `step()` contract, implemented
   without a `gymnasium` dependency — see "no ML deps" note below); the fixed
   fall penalty only applies when `terminated`.
2. **Terminal observations are handled correctly.** On a fall the returned
   observation is the real post-fall state (or an all-zero placeholder only
   if the state actually went non-finite) — not a stale pre-fall observation,
   and not silently dropped.
3. **Reward cannot receive velocity-tracking credit on the terminating step.**
   Fixed during this check: the original implementation computed `r_vel` from
   the same-step CoM velocity even when that step was the fall itself — a
   pitching collapse drags the pelvis CoM forward and would have been paid as
   if it were forward walking. `cara_env.py` `step()` now short-circuits to a
   flat `reward = -10.0` with every reward *component* set to `None` (visible
   in `info["reward_components"]`) whenever `fallen` is true, so no per-term
   weight tuning can accidentally reintroduce this leak later.
4. **Action/observation shapes match the trainer.** The trainer built for
   U20 (`train_ars.py`) reads `env.action_space.shape` / `.low` / `.high` and
   `env.observation_space.shape` directly — no hard-coded dimensions.
5. **No `gymnasium` compatibility adapter was needed.** The project's `.venv`
   has no `gymnasium`/`gym`/`torch` installed (confirmed via `pip list`:
   `mujoco`, `numpy`, `PyYAML`, plus rendering libs `glfw`/`PyOpenGL` — no ML
   framework). Rather than add a new dependency, `cara_env.py` implements the
   Gymnasium `reset()->(obs,info)` / `step()->(obs,reward,terminated,truncated,info)`
   contract directly with a small dependency-free `Box` shim
   (`cara_env.Box`), and U20's trainer is a from-scratch NumPy implementation
   (Augmented Random Search) rather than a PyTorch PPO/SAC baseline. If a
   future trainer needs real `gymnasium.spaces.Box` objects, swapping the
   shim for the real class is a one-line change — nothing else in `cara_env.py`
   depends on it being the shim specifically.

## U20 — first nominal-locomotion training experiment

`cara_description/scripts/train_ars.py` implements Augmented Random Search
(ARS-V2, Mania et al. 2018) — a numpy-only, gradient-free policy search that
has been shown competitive with deep-RL baselines on MuJoCo locomotion tasks,
chosen specifically because it needs no autodiff/deep-learning dependency.
Policy: a single linear layer `action = clip(W @ normalize(obs), -1, 1)`,
`W` shape `(12, 43)`, with an online (Welford) observation mean/std
normalizer updated from every observation seen during rollouts.

Conditions for this first run (per the review guidance — kept deliberately
narrow): fixed masses (default `cara_full_body.yaml`, no randomization), flat
ground, unchanged actuator limits, a single fixed forward-velocity command
(`desired_vx = 0.10 m/s`). No domain randomization, no adaptation. Results and
the accepted/rejected configuration are recorded in this file's changelog
below once the run completes, per the "record the model/configuration"
instruction — this section is updated in place, not superseded, so the
baseline stays traceable.

### Evaluation (separate from training)

`cara_description/scripts/evaluate_policy.py` loads a saved policy and runs
two fixed evaluation suites — zero-speed standing and forward walking — and
reports, without touching training code: episode duration and fall rate;
commanded vs. achieved velocity; alternating foot clearance/touchdown pattern
(from `info["foot_z"]`/`info["foot_touch"]` per foot); joint tracking error
(`info["qpos_err"]`, mean |commanded − actual| over the 12 leg joints); and
actuator saturation (`info["torque_frac"]`, fraction of `forcerange` used).

**The milestone is repeatable forward stepping, not reward alone** — the
evaluation report explicitly checks that forward displacement correlates with
alternating single-foot contact loss (stepping), not a foot sliding while
always in contact, or a forward lean/fall that happens to end past the start
line.

### U20 run log (2026-09-13)

**Two real reward-design bugs found and fixed by inspecting reward components
first, per the review guidance — before touching morphology, actuators, or
the learning algorithm:**

1. **Early-termination exploit.** The environment originally had no per-step
   "alive" reward: every non-terminal step cost `-w_vel*|vx-desired| + small
   penalties < 0`, while a fall cost one fixed `-10` and then *stopped*
   accumulating further negative reward. A run 1 (`--directions 8 --top-b 4
   --episode-seconds 4.0`, 20 iterations, `w_vel=1.0`, no alive bonus) showed
   this directly: episode length under the evaluation policy fell from 200
   steps (iter 0) to 20-40 steps by iter 10, monotonically — the search
   discovered that falling early minimizes accumulated cost. **Fix:** added
   `w_alive=1.0`, a flat `+1.0`/step reward while not fallen (`cara_env.py`
   `CaraWalkEnvConfig.w_alive`). Re-ran the same 20-iteration check after the
   fix: the zero-action policy's return went from -6.12 (60 steps) to +53.88,
   confirming survival is no longer punished.
2. **Velocity term too weak relative to the alive bonus.** With `w_alive=1.0`
   and the original `w_vel=1.0`, standing still scored ~0.9/step and perfect
   velocity tracking scored ~1.0/step at `desired_vx=0.10` — only a 0.1/step
   marginal benefit to attempting to walk, against the risk of forfeiting the
   rest of the episode's alive bonus by falling. Run 2a (300 iterations, 16
   directions, top-8, `step_size=0.015`, `noise_std=0.02`, `w_vel=1.0`)
   confirmed this: `eval_return` plateaued at ~180 (the max achievable by
   never falling) from iteration 0 and never moved off it —
   `evaluate_policy.py` on the resulting policy: **standing (vx=0) FELL in
   0.54s; forward-walk target (vx=0.10) survived the full 4s but made
   essentially zero forward progress (-0.001m) — verdict NOT WALKING.**
   **Fix:** raised `w_vel` to `10.0`, which makes standing-still ~0.0/step
   (neutral) and perfect tracking ~+1.0/step — a real incentive to move,
   while a fall still costs more than riding out the episode at the neutral
   baseline.

**Run 2b (same settings as 2a, `w_vel=10.0`, 300 iterations, ~99s wall
time):** did NOT converge to standing-still this time — `eval_return`
fluctuated noisily around 0 for the full run with no clear upward trend,
falling consistently around step 20-30 of a 200-step episode.
`evaluate_policy.py` result: **both standing and forward-walk evaluations
FELL, in 0.44-0.46s.** The evaluation script's verdict logic was itself
found to have a bug during this check and was fixed before trusting its
output: it originally called a fall a "STEPPING" verdict whenever the
contact pattern alternated and the pelvis moved forward, without checking
whether the episode had actually survived — exactly the failure mode the
milestone definition warns against, since a topple produces foot-scuffing
contact flicker and forward CoM lurch for free. `summarize()` now requires
`not fell and steps >= 0.8*requested_steps` before any "STEPPING" verdict is
possible; a fall is reported as `FELL` regardless of what its contact/motion
numbers look like.

**Actuator-tracking check (independent of RL, per the pivot message's
caveat that standing alone only establishes the static case):** drove the 12
leg joints with a hand-crafted, non-balancing alternating sinusoidal swing
(0.5 Hz, legs 180° out of phase) — not a real gait, just an open-loop motion
to exercise the actuators under load. Result: joint tracking error stayed
small throughout (mean 0.019 rad, max 0.034 rad) even as actuator saturation
climbed high (mean 66%, peaked at 100%, saturated on 17% of steps), and the
robot fell at t=0.92s (no balance strategy in this test, so falling was
expected). The small tracking error argues *against* a fundamental
position-actuator tracking failure; the saturation shows the joints are
working hard to hold that motion under load, consistent with the pivot
message's caveat, but not disqualifying.

**Net assessment:** the milestone (repeatable forward stepping) is **not yet
met** under either reward weighting tried. `w_vel=1.0` converges to a safe,
non-walking local optimum; `w_vel=10.0` avoids that collapse but the search
budget (300 ARS iterations, 16 directions, a from-scratch linear policy, no
curriculum, no randomized initial state) has not found a stable gait within
this "short" experiment's scope. Per the guidance ("if the policy fails,
inspect reward components and actuator tracking before changing morphology
or adding more sophisticated learning"): both reward components and actuator
tracking have now been inspected and one real class of bug (reward
incentive structure) was found and fixed twice over; nothing here points at
morphology or the actuators as the blocker. The open question is scale
(more directions/iterations, a longer or curriculum'd run) versus a
different search/exploration strategy — a decision left for the next
session rather than escalated unilaterally.

### U20 reward audit (before proceeding to PPO)

Per review guidance: "adding an alive bonus and increasing velocity weight
are reward-design changes, not automatically complete fixes." Before
starting any new training run, `cara_description/scripts/reward_audit.py`
computes the actual discounted return (`gamma=0.99`, the value planned for
PPO; `gamma=1.0` reported alongside since that's what ARS effectively
optimized) over the real 500-step (10s) episode horizon for three outcomes:

- **A. standing** — real simulated rollout, zero action, full horizon.
- **B. brief-forward-then-fall** — real simulated rollout (the same
  hand-crafted open-loop swing used in the actuator-tracking check);
  genuinely falls at step 47 having drifted -0.185m.
- **C. sustained target-speed** — the intended behavior. No controller
  produces this yet, so it is computed as an **idealized upper bound**
  (exact `vx=desired_vx`, perfectly upright, modest realistic action
  magnitude) plugged directly into the reward formula — not a simulated
  rollout. Labeled as such; this is the most generous case the reward could
  ever pay a real gait, so if it still doesn't clearly win, that is a
  problem with the reward, not with simulation fidelity.

Ranking alone is not sufficient — a technically-correct order with a thin
margin over "standing" is still an attractor for a noisy gradient-free
search. Re-running the audit against the **old** `w_vel=1.0` weighting
confirms this concretely:

| weighting | A. standing (disc.) | C. sustained (disc.) | margin of C over A | verdict |
|---|---|---|---|---|
| `w_vel=1.0` (original) | 89.33 | 99.25 | **10.0%** | FAIL (thin margin) |
| `w_vel=10.0` (current) | -0.80 | 99.25 | **100.8%** | PASS |

C nominally outranked A even under the old weighting — but only by 10%,
which is exactly consistent with what run 2a actually did: get stuck at
standing despite the ranking being "technically correct." The current
`w_vel=10.0` weighting clears this with a wide margin (standing nets
slightly negative, sustained target-speed nets ~99% of its own value ahead
of it) and is the configuration carried into U21 (PPO). This audit should
be re-run any time the reward function changes, before trusting a new
training run's results.

## U21 — nonlinear PPO baseline (a different learning setup, not an ablation)

Per explicit instruction: keep ARS as the first benchmark rather than
spending substantially more compute on it, and choose a nonlinear PPO
baseline next with a bounded training budget. The absence of torch/gymnasium
in the project's main `.venv` is an environment constraint, not a reason to
add a heavy ML dependency to it — solved with a **separate venv**,
`.venv-rl` (Python 3.11 + `torch` 2.14.0 CPU + `gymnasium` 1.3.0 + `mujoco`
3.13.0). `cara_description/scripts/cara_env.py` still imports neither torch
nor gymnasium — `train_ppo.py` and `evaluate_ppo.py` are the only files that
do, and they only run under `.venv-rl`.

**Implementation** (`train_ppo.py`): follows CleanRL's
`ppo_continuous_action.py` structure (a widely-used, established single-file
PPO reference for continuous control) — a small tanh MLP actor-critic
(64,64) with a state-independent log-std, GAE(λ), the clipped surrogate
objective, advantage normalization, and observation normalization via an
online running mean/std (reward is deliberately **not** normalized, so
returns stay comparable with `reward_audit.py`'s raw-reward analysis).
`CaraWalkGymEnv` is a thin `gymnasium.Env` adapter around `CaraWalkEnv` —
all actual environment logic still lives in `cara_env.py`, untouched.

**Correct timeout handling** — verified empirically before relying on it,
not assumed: `gymnasium.vector.AsyncVectorEnv`'s `NEXT_STEP` autoreset mode
returns the real terminal observation on the step that ends an episode
(confirmed with a scripted 3-step toy env: `step()` returns the genuine
final observation together with `terminated=True`, and only the *following*
`step()` call performs the actual reset). This means GAE bootstrapping can
use that real terminal observation directly: a true fall (`terminated`)
bootstraps with `0` (no continuation value); a timeout (`truncated`)
bootstraps with the critic's value of that real terminal observation instead
of `0` — exactly the distinction the review guidance asked to get right.

**Throughput measured before choosing a budget** (per instruction, not
guessed): single-process env-only stepping is 2736 steps/sec; 8 concurrent
processes reach ~9571 steps/sec aggregate; full PPO end-to-end (rollout +
10 epochs × 8 minibatches of gradient updates per iteration, batch=2048,
8 parallel envs) is **~730 steps/sec** — the optimization step, not
environment stepping, dominates wall time at this scale. From that number,
`total_timesteps=300,000` was chosen as the fixed, bounded per-seed budget
(~410s projected; 375-388s actual across 3 seeds — see below).

**Task kept fixed**, per instruction: same flat-terrain default MJCF,
nominal masses, `action_range_frac=0.30`, `desired_vx=0.10 m/s`. No reward
change and no curriculum introduced during this run.

**Three seeds** (0, 1, 2), same hyperparameters, run sequentially:

| seed | wall time | fall_rate (training, final) | ep. length (final, of 200-step cap) |
|---|---|---|---|
| 0 | 388s | 1.0 | ~19-22 steps |
| 1 | 378s | 1.0 | ~19-22 steps |
| 2 | 387s | 1.0 | ~19-24 steps |

**Every seed shows the same pattern**: episode return improves somewhat
over training (roughly -17 → -1..-5, mostly from reduced effort/action-rate
penalty as actions shrink) but **episode length never grows past ~20-24
steps of a possible 200**, and the training-time fall rate never drops below
1.0 for any of the three seeds, for the entire 300k-step budget. This is a
consistent, reproducible obstacle across seeds — not seed noise.

**Evaluation** (`evaluate_ppo.py`, deterministic/mean action, same
metrics+verdict logic as U20's `evaluate_policy.py`, imported directly
rather than reimplemented):

| seed | scenario | verdict | genuine steps | fwd. distance |
|---|---|---|---|---|
| 0 | standing | NOT WALKING | 0 | +0.013m |
| 0 | forward-walk | FELL | 2 | +0.058m |
| 1 | standing | FELL (1.16s) | 0 | +0.205m |
| 1 | forward-walk | FELL (0.64s) | 2 | +0.148m |
| 2 | standing | FELL (0.70s) | 0 | +0.198m |
| 2 | forward-walk | FELL (0.88s) | **3** | -0.077m |

`genuine steps` = maximal runs of single-leg support lasting ≥60ms (3
control steps) — filters out the contact flicker a topple produces for
free, per the instruction to retain partial-progress metrics even when the
final verdict is a fall. `runs/u21_ppo/ppo_seed2_forward_walk.gif` (verified
frame-by-frame, matching the two GIF-encoder bugs fixed earlier this
session) shows a real single-leg lift attempt before the topple.

**Comparison to the U20 ARS baseline**: PPO's best case (seed 2, 3 genuine
steps) edges out ARS's best case (2 genuine steps), and 2 of 3 PPO seeds
show substantially more forward CoM drift before falling (+0.15-0.21m vs.
ARS's -0.185m/+0.059m). Both baselines still fail the milestone — **neither
achieves a fall rate below 100% or repeatable stepping**. Per the
instruction, this comparison does not isolate which change (nonlinear
policy vs. search method) would help; it establishes that a different
learning setup was tried honestly and under a bounded budget, not that
either approach is closer to solving the task.

**Net assessment**: the reward audit passed before this run, so the reward
is not implicated here. The observed pattern — exploration destroys stance
before any useful transfer occurs, identically across all 3 seeds — is
exactly the trigger condition the review guidance names for considering a
curriculum on command speed (start near `desired_vx=0`, expand gradually).
That decision, and any budget increase, is left for the next session rather
than started unilaterally here. All artifacts (checkpoints, per-seed
training logs, eval reports, the GIF, and this run's exact hyperparameters)
are preserved under `cara_description/runs/u21_ppo/` with `manifest.json` as
the index, matching the same preservation standard set for U20's ARS runs
in `cara_description/runs/u20_ars/`.

## U22 — a desired_vx curriculum (start near 0, expand gradually)

U21 showed the same obstacle in all 3 seeds: fall rate stuck at 1.0, episode
length never past ~20-24 of 200 steps, for the whole budget, regardless of
`desired_vx`. That is exactly the trigger condition the review guidance
names for a curriculum: "if exploration repeatedly destroys stance before
any useful transfer occurs, begin with easier low-speed commands and
gradually expand them."

**Mechanism** (`train_ppo.py --curriculum`): `vx_stages =
[0.0, 0.03, 0.06, 0.10]`. Advancement to the next stage requires, over the
most recent 30 completed episodes, mean episode length ≥80% of the 200-step
horizon **and** fall rate ≤30%, after at least 100,000 env-steps already
spent in the current stage. `desired_vx` is pushed live into all 8 worker
processes via `AsyncVectorEnv.call("set_desired_vx", vx)` — no vector-env
teardown/recreate between stages. Verified with a lenient smoke test before
trusting it on the real run (advancement logic, live propagation, and
checkpoint/history recording all confirmed working). Task and reward
otherwise unchanged from U21 (same audited `v3_alive+vel10` reward).

**Budget**: 800,000 steps, one seed — a new mechanism gets validated on one
seed before committing to a 3-seed budget, same practice as everywhere else
in this project. Wall time: 946.5s (~15.8 min), consistent with the U21
throughput measurement.

**Result: the curriculum never advanced past stage 0** (`desired_vx=0.0`,
pure balance — the *easiest possible* stage) for the entire budget. Episode
length crept up from ~20 steps early in training to a plateau around
~26-30 steps from roughly step 400,000 onward — real but small progress,
well short of the 160-step threshold. **Fall rate stayed at 1.0 for the
entire run**, never dipping below the 30% bar needed to advance even once.

This is a more precise result than U21 could produce on its own: **even
with zero forward-velocity demand, this PPO setup has not learned reliable
balance yet.** The obstacle observed in U21 was never really about learning
to walk forward — it's more fundamental. Deterministic evaluation confirms
this: 1/1 evaluated episode fell at 0.82s (41 steps), actuator saturation
mean 87% / peak 100% (higher than any U21 seed's fixed-vx run), i.e. the
policy is working hard and still failing to hold itself up.
`runs/u22_ppo_curriculum/ppo_curr_seed0_standing.gif` (verified
frame-by-frame) shows a forward-toppling fall from a crouched/leaning
posture.

**Net assessment**: the curriculum mechanism itself is confirmed working
correctly — it simply had no reason to fire, because the underlying
balance problem wasn't solved even at the easiest stage. This shifts the
leading hypothesis: not "walking is hard," but "this policy/setup hasn't
learned to balance at all yet." Candidates worth checking next, in rough
order of cost: (1) whether more raw budget alone eventually cracks stage 0
(untested — 800k steps is not a large budget for from-scratch bipedal
balance); (2) whether `action_range_frac=0.30` gives the policy the right
granularity for fine balance corrections (untested, provisional value from
U19); (3) whether the position-PD actuation path (chosen in U19 specifically
to bypass U15's torque-ankle instability) has weaker disturbance rejection
than assumed — standing alone was verified in U19/U20, but that was a
*static*, non-learning check, not a check under an actively-perturbing
policy. None of these have been investigated yet; this is a decision point,
not a recommendation acted on unilaterally. All artifacts (checkpoint,
training log, eval report, GIF, smoke-test verification) are under
`cara_description/runs/u22_ppo_curriculum/` with `manifest.json` as the
index.

## U23 — bounded diagnostic: why does training move away from standing?

Explicit instruction: investigate action scaling and exploration next, using
the known zero-action standing policy as the control, and do **not** extend
the training budget yet. `cara_description/scripts/diagnose_stage0.py` is a
pure evaluation script (no training) comparing four controllers under
**identical resets, stage-0 reward (`desired_vx=0.0`), episode length
(200 steps), and model config**. Note: `CaraWalkEnv.reset()` is confirmed
(by reading the code, not assumed) to be fully deterministic — no noise is
added to the `stand_nominal` keyframe — so "identical resets" is
automatically satisfied, and 3 of the 4 conditions are exactly reproducible
in a single rollout; only the sampled-action condition has real randomness
and was repeated 10 times.

**Action range, translated to actual radians** (not just the 0.30
fraction): `hip_pitch` and `knee_pitch` get the most authority (±24.1°,
±20.3°), `ankle_roll` the least (±7.6°) — see the manifest for the full
per-joint table.

**Exploration std, translated the same way**: PPO's initial
`actor_logstd=-0.5` gives an action-space std of 0.6065 — which, run through
a quick Gaussian tail calculation *before* touching the simulator, predicts
that at least one of the 12 joints' sampled action exceeds the `[-1,1]`
bound on **~72% of steps**, from the very first step of training, before any
learning happens. This was checked as a hypothesis, then confirmed
empirically.

**Results**:

| controller | steps (of 200) | fell | total return | action \|mean\| |
|---|---|---|---|---|
| 1. zero action | 200 | No | **193.00** | 0.0000 |
| 2. untrained policy, deterministic mean | 200 | No | 191.82 | 0.0025 |
| 3. untrained policy, sampled (10 repeats) | 21.3 avg | **100%** | -5.45 avg | ~0.47-0.52 |
| 4. trained policy (U22), deterministic mean | 41 | Yes | -18.23 | 0.5196 |

**Decision-sequence readout** (following the given rule set exactly):
1. *Zero action fails under training resets?* **No** — reconfirmed, holds
   the full episode. Reset/baseline stability is fine.
2. *Zero action rewarded poorly?* **No** — 193.00 vs. the trained policy's
   -18.23. The objective correctly favors standing; this was never a reward
   problem (consistent with U20's reward audit already passing).
3. *Initial mean succeeds, sampling fails?* **Yes.** The untrained policy's
   deterministic mean essentially *is* the zero-action baseline (action
   magnitude 0.0025, return within 1.2 points of zero-action) — confirming
   initialization starts at nominal stance, not away from it. But sampling
   from that same mean, at the initial std, produces a **100% fall rate in
   21.3 steps on average** — which closely matches the ~20-30 step collapse
   seen throughout real PPO training in U21 and U22. Measured action-clip
   rates (6.5-12.8% of joint-steps) closely track the 9.9% theoretical
   per-joint prediction. Actuator saturation during these sampled rollouts
   runs 90-100%, and peak tilt sits right at the 40° fall threshold
   (41.5-46.0°) — consistent with exploration noise itself driving the
   robot to the failure boundary, not a slow drift.

Steps 4 (actuator path) and 5 (PPO updates/observation normalization) were
**not reached** — their trigger conditions were not met by this evidence.
Condition 4 also shows the trained policy's mean has been pulled *away*
from its good initialization (return -18.23, action magnitude 0.52, versus
the untrained mean's 191.82 / 0.0025) — a plausible consequence of PPO's
gradient estimate being computed almost entirely from noise-corrupted,
falling rollouts throughout training, rather than from the initially-good
mean behavior it started with.

**Conclusion, per the given decision rule**: reduce initial exploration std
first, keeping `action_range_frac=0.30` (the action bounds) unchanged. No
training budget was spent or extended this session — this was diagnosis
only. The concrete next step (a new `--init-logstd` option on
`train_ppo.py`, a lower starting std, then a bounded validation run at the
**same** budget scale as U21/U22, not larger) is proposed but not yet
started, pending confirmation. All artifacts (script, JSON report, full
stdout log, manifest with the complete decision-sequence readout) are under
`cara_description/runs/u23_stage0_diagnostic/`.

## U24 — validation: reduced initial exploration std

Confirmed next step per U23: added `--init-logstd` to `train_ppo.py`'s
`ActorCritic` (default `-0.5`, unchanged for backward compatibility) and ran
a validation with `--init-logstd -1.0` (std=0.3679, ~8% per-step chance any
of 12 joints clips — down from ~72% at the original `-0.5`).
`action_range_frac=0.30` was **not** touched. Same curriculum settings and
**same 800,000-step budget as U22** — not extended, per instruction — for a
direct, controlled comparison where init_logstd is the only variable
changed. Wall time: 995.2s, consistent with U22's 946.5s.

**Result: still did not advance past stage 0** within budget — fall rate
stayed at 1.0 for the entire run. **But this is not a wash**: every tracked
metric improved, measurably, in the same deterministic-mean evaluation used
for U22's checkpoint:

| metric | U22 (`init_logstd=-0.5`) | U24 (`init_logstd=-1.0`) | change |
|---|---|---|---|
| steps survived (of 200) | 41 | 63 | +54% |
| total return | -18.23 | -6.66 | less negative |
| action magnitude | 0.5196 | 0.3811 | -27% |
| action Δ step-to-step | 0.3588 | 0.1154 | -68% (much smoother) |
| action clip rate | 12.8% | 0.5% | -96% relative |
| actuator saturation (mean) | 87.1% | 75.8% | -13% |
| joint tracking error (mean) | 0.0413 rad | 0.0199 rad | -52% |
| peak tilt | 41.5° | 40.4° | still crosses the 40° threshold, barely |

The training curve itself shifted only modestly (episode length plateaued
around ~30-35 steps vs. ~26-30 before — real, not dramatic), but the
**deterministic** policy — what actually gets evaluated and what would run
on hardware — is now close enough to holding stance that peak tilt sits
right at the fall boundary (40.4° vs. the 40° cutoff) rather than well past
it (41.5°, and U21's fixed-vx seeds ranged up to ~43-46°). One incidental
observation: mid-training PPO `clipfrac` ran noticeably higher (~0.20-0.24
vs. near-zero at the same step count in U22) before decaying to 0 via LR
annealing — plausibly a lower-entropy, more "confident" policy producing
larger per-update ratio swings; not flagged as a problem on its own, just
recorded.

**Net assessment**: the exploration-std diagnosis from U23 was correct and
produced consistent improvement everywhere it was checked — but didn't
fully solve stage-0 balance within this budget. That's evidence the
diagnosis was right but incomplete at this budget, not that it was wrong.
Three untested candidates for next, in rough order of cost: more budget
under the same lower std (does stage 0 crack given more steps now that the
destructive-exploration confound is reduced?); an even lower `init_logstd`;
or proceeding to the original decision sequence's step 4 (actuator path —
small smooth joint-target ramps) now that exploration is no longer the
dominant confound. None started — flagged for the next decision. All
artifacts (checkpoint, training log, eval report, and a full re-run of the
U23 four-controller diagnostic against this checkpoint) are under
`cara_description/runs/u24_ppo_lowstd/` with `manifest.json` as the index.
**U24 is preserved as the current best result** and was not overwritten by
anything below.

### Correction: 40.4° vs. 41.5° is not evidence of "close to balance"

Flagged directly: termination fires at 40°, so both U22's and U24's peak-tilt
numbers mostly measure how far the robot crossed the threshold *between
simulation checks*, not how close either policy came to actually holding
balance. Retracted as a "close" claim. The real, defensible U24
improvements are the ones that don't depend on where the fall threshold
sits: +54% survival, -68% action jitter, -96% relative clip rate, -52%
tracking error.

### Instrumentation fixes (before any further training)

Two gaps identified and closed in `train_ppo.py`, neither requiring a new
run to apply going forward:

1. **`actor_logstd` is a trainable `nn.Parameter`** (included in
   `optimizer.parameters()`), so `--init-logstd` only sets its *starting*
   value — lower-at-init and lower-throughout-training are different
   interventions that look identical if only the final checkpoint is
   inspected. Checked retroactively on the existing checkpoints (final
   values only, no full trajectory available for either): U22 (`init=-0.5`)
   ended at per-joint `logstd` -0.68 to -1.08 (std 0.34-0.51); U24
   (`init=-1.0`) ended at -1.16 to -1.55 (std 0.21-0.31) — U24 ended up
   lower on every single joint than U22's endpoint, which is *consistent*
   with U24 having genuinely lower exploration throughout training, not
   just at step 0, though the full trajectory was never logged so this
   isn't proven. **Fixed going forward**: every training-loop history row
   now records `logstd_mean`/`logstd_min`/`logstd_max`, and the per-iteration
   console line prints the current mean logstd — the next run will have the
   real trajectory, not just endpoints.
2. **Resuming a checkpoint would have silently reset the learned
   distribution.** Added `--resume-from`: loads the full model
   `state_dict` (including the actual learned `actor_logstd`, not
   `--init-logstd`) and the observation normalizer from a checkpoint;
   `--init-logstd` is explicitly ignored and this is logged when resuming.
   Optimizer (Adam moment) state is not restored — a fresh optimizer is
   used, which is a known, disclosed limitation, not a silent gap.
   Smoke-tested against the real U24 checkpoint before trusting it: resumed
   `actor_logstd` matched the checkpoint's saved values exactly, and
   training continued (not reinitialized) from there.

## U25 — small, smooth actuator-response probes

Explicit instruction, before spending any more training budget: does Cara
respond predictably to reasonable commands? `probe_actuator_response.py`
(no training) runs four tests, same model/resets/control timing as
everywhere else: **zero-action hold** (per-joint baseline), a **fast**
smooth single-joint excursion-and-return (100ms cubic-smoothstep ramps,
200ms hold — zero endpoint velocity, matching `dcm_walk.py`'s existing
`smooth01` shape), a **slow** version (400ms ramps, isolating command speed
from pose/loading), and a **replay of the actual U24 policy** with the same
per-joint instrumentation for direct comparison. Tested individually (all
11 other actuators held at nominal) on `l_ankle_pitch`, `l_knee_pitch`,
`l_hip_pitch`, at amplitude 0.4 — chosen to bracket U24's own mean
`|action|=0.3811`, not arbitrary.

**Zero-action hold**: 0% saturation on all three joints, tracking error
≤0.005 rad. Clean baseline, no pre-existing saturation issue at any tested
joint.

**Smooth excursions (fast and slow, 6 tests total)**: **0% saturation in
every single one**, tracking error 0.014–0.024 rad (~1–1.4°) throughout,
including in the four tests where the robot ultimately fell. The ankle and
knee excursions did cause falls — expected and disclosed as such: an
isolated, uncompensated single-joint movement with no balance strategy is
inherently destabilizing by design, and per instruction this is explicitly
**not** treated as an actuator failure, because tracking stayed accurate
right up through the fall (mean error never exceeded 0.024 rad in any of
the six tests, whether it fell or not). Hip excursions never fell (peak
tilt 7–8°).

**U24 replay — categorically different**: peak per-joint angular velocity
was **2.5–5× higher** than even the *fast* smooth test (ankle: 6.22 vs.
1.21 rad/s; knee: 3.81 vs. 1.03; hip: 2.78 vs. 1.13), tracking error 2–2.5×
worse, and **real saturation appears for the first time** — 20.6% of steps
on the ankle (longest continuous run 140ms), something none of the six
smooth-command tests produced on any joint at any speed. This isn't sampling
noise: it's the policy's deterministic **mean** output, and the underlying
step-to-step jitter (`action_delta=0.1154` in normalized units per 20ms
step) is a property of the learned mean function itself.

**Decision, per the given rule set**: commands track well under smooth,
moderate-speed motion — this is not an actuator/controller-path defect.
Poor tracking and saturation only appear when replaying the actual learned
policy, whose commands are well outside the range and speed just verified
to work cleanly. This points to **"commands track well but the posture
destabilizes Cara: return to policy coordination and reward incentives"** —
specifically, nothing in the current reward (`w_action_rate=0.01`, quite
small relative to `w_alive=1.0`) or architecture discourages the
high-frequency, high-velocity action pattern the policy has converged to,
and that pattern — not the actuators — appears to be what's actually
driving saturation and tracking error. No fix implemented and no new
training run started this session — diagnosis only, matching the U23
pattern. All artifacts (script, full per-joint JSON, stdout log, manifest
with the complete decision readout) are under
`cara_description/runs/u25_actuator_probe/`.

## U26 — bounded reward comparison: the last local smoothness experiment

Explicit framing: U25 narrowed the problem but did not *prove*
`w_action_rate=0.01` was too weak — its effect depends on the action
differences, aggregation, and other terms, and some of the observed
jitter may be the policy reacting to a developing fall rather than a cause
of it. One bounded, decision-ruled comparison, set before training, meant
to be the last local smoothness experiment before a strategy call.

**Frozen this run**: `CaraWalkEnv` mechanics untouched — same actuators,
`action_range_frac=0.30`, curriculum settings, termination. The *only*
varied factor between the two branches is `w_action_rate`.

**Penalty chosen from measured contributions, not the coefficient's
size**: `analyze_reward_contributions.py` measured the actual raw
(unweighted) `r_rate` distribution from U24 — both the deterministic mean
and, more importantly, **stochastic sampling using U24's own learned std**
(what PPO's training gradient actually sees). P75 raw `r_rate` under
sampling was **0.315** — nearly 11× the deterministic value (0.029),
because sampling noise adds real per-step jitter on top of whatever the
mean itself does. Targeting "P75 sampled jitter costs ~10% of the per-step
alive bonus" gives `w_action_rate ≈ 0.318`; rounded to **0.3** (a ~30×
increase from 0.01, derived from data, not a guess at the coefficient's
scale).

**Instrumentation hardened first**: `train_ppo.py` now saves optimizer
state in checkpoints, and `--resume-from` restores it when present —
verified against U24's checkpoint, which predates this and can therefore
only be **warm-started** (policy, critic, learned `actor_logstd`, and
observation normalizer restored; optimizer state is not, since U24 never
saved it). Both U26 branches are disclosed as warm starts, not exact
resumptions, per instruction. Any future resume from a post-U26 checkpoint
will be exact.

**Two branches, identical 400,000-step budget, same seed, same starting
checkpoint (U24)**:

| metric (deterministic-mean, 4s episode) | U24 baseline | branch A (unchanged, 0.01) | branch B (stronger, 0.3) |
|---|---|---|---|
| steps survived | 63 | 56 | **74** |
| tracking error (mean) | 0.0199 rad | 0.0215 rad | **0.0150 rad** |
| actuator saturation (mean) | 75.8% | 72.9% | **62.4%** |
| raw `r_rate` P75 (deterministic) | 0.029 | 0.021 | **0.0056** |

Branch A — 400,000 *more* steps under the unchanged reward — did not
improve substantially; survival actually **regressed slightly** relative
to the U24 checkpoint it started from (63→56 steps), despite identical
training conditions otherwise. Branch B improved on every tracked metric
simultaneously: +17.5% survival, -5.2× deterministic-mode jitter, -25%
tracking error, -18% relative saturation. Under sampled (training-mode)
behavior the smoothness gain was more modest (~16% lower P75 `r_rate`,
since sampling noise itself — nearly unchanged between branches, std
converged to within 0.5% of each other — dominates that number regardless
of the mean's own smoothness). Training curves for both branches show
`fall_rate` stuck at 1.0 for the entire 400k-step continuation; branch B's
episode-length plateau shifted modestly higher (~36-41 vs. ~33-36).

**Per the decision rule set before training**: branch B fires *"stronger
penalty improves survival and reduces disruptive commands → keep it and
test the standing milestone."* Branch A does not fire *"unchanged
continuation improves substantially."* **Decision: `w_action_rate=0.3` is
adopted as the new default going forward.**

**Standing milestone, tested as instructed**: ≥90% full-episode survival
over 30 evaluation resets, using branch B's checkpoint. `CaraWalkEnv.reset()`
is confirmed deterministic (U23), so 30 identical resets of a deterministic
policy would just repeat one trial — this test used the policy's own
learned stochastic sampling across 30 independent repeats instead, the only
way 30 resets produce meaningfully different outcomes in the current
environment (pending the separately-defined perturbation test). **Result:
0/30 (0%) — NOT MET.** Mean survival 37.3 of 200 steps, 100% fall rate.
Reported plainly as the expected outcome at this stage, not a failure
specific to branch B's decision — the milestone gate exists precisely so
this can be checked honestly rather than assumed.

All artifacts (both checkpoints — now warm-startable exactly, going
forward — training logs, reward-contribution analysis for both branches,
the milestone-test script and its full 30-repeat log) are under
`cara_description/runs/u26_reward_comparison/` with `manifest.json` as the
index. No further coefficient tuning or mechanical diagnosis was started
this session, per instruction — this was the last local smoothness
experiment before a strategy decision.

### Two reporting corrections carried into U27

1. **U26's branch B is adopted provisionally, not established.** One bounded
   comparison supports `w_action_rate=0.3` over the original `0.01` — it
   does not establish reliability across training seeds. Nothing here has
   run that seed sweep yet.
2. **Saving optimizer state alone does not guarantee exact resumption.**
   U26's `--resume-from` restores model weights, `actor_logstd`, the
   observation normalizer, and (when present) optimizer state — but RNG
   stream position, the LR/entropy schedule's exact position, and rollout
   buffer/environment state are not all preserved. U26's continuations are
   best understood as **close, disclosed warm starts**, not bit-exact
   resumptions, even where optimizer state was restored. (U27, below, does
   not resume from anything, so this is a correction to U26's framing, not
   a defect found in U27.)

## U27 — strategy change: initialize near the known standing solution

The question changes: not "can PPO rediscover walking from scratch," not
"repair a policy that already falls," but **"can PPO preserve a known
successful stance and learn recovery around it?"** Cara already has a
nominal standing behavior (zero action); the learner should first preserve
it, then learn recovery around it — not required to rediscover it, and not
required to copy DCM's gait.

**1. Zero-mean-init, verified not assumed.** `ActorCritic(zero_mean_init=True)`
zeroes the final `actor_mean` layer's weight *and* bias exactly (hidden
layers keep ordinary orthogonal init) — not the previous `std=0.01`
orthogonal init, which U23 measured as *close to* but not exactly zero
(action magnitude 0.0025). Checked directly: `actor_mean(x) == 0` for
arbitrary random inputs, and a full 200-step deterministic rollout through
the complete observation/action/reward pipeline reproduced the zero-action
baseline's return to floating-point tolerance (193.0046, both ways) with an
identical step count.

**2. Exploration std chosen empirically, from the zero-mean policy
specifically** (not reused from U23/U24's non-zero-mean sweep, since the
question here is different: not "does this avoid catastrophic clipping" but
"do rollouts usually stay upright long enough to be useful"). Swept
`init_logstd` from -1.5 to -3.5 (15 sampled repeats each, zero perturbation):
-1.5 through -2.5 all gave **0% full-episode survival**; **-3.0 gave 73.3%
full survival**, with even the falls lasting a long time (minimum 123 of
200 steps); -3.5 gave 100% survival but *zero* fall exposure — judged too
conservative, since the instruction asked for "usually," not "always," and
some failure exposure is needed to learn from. Chose **-3.0**.

**3. Training**: fixed `desired_vx=0.0` (stage-0 only, no curriculum
machinery — the goal is standing, not progressing speed stages),
`w_action_rate=0.3` (U26's adopted value, with the reliability caveat
above), fresh start (no resume), 400,000-step bounded budget (set before
starting, matching U26's per-branch scale), single seed. Added periodic
evaluation (`--eval-every 5`): a deterministic-mean rollout from a nominal
reset every 5 iterations, with a separate best-by-standing-performance
checkpoint saved independently of the final one.

**Result — decisively different from every prior approach**: training-time
`fall_rate` hit **0.0 by iteration 16 of 195** (global step 32,768 — **8.2%
of budget**) and stayed there for essentially the rest of training (two
single-iteration blips to 0.05–0.1 that immediately recovered). The
periodic deterministic eval hit the maximum 200 steps by **iteration 5**
(global step 10,240 — **2.6% of budget, 11.6s of wall time**) and never
needed to improve again — the best-checkpoint mechanism locked in almost
immediately. For contrast: every prior approach (U20's from-scratch ARS,
U21-U22's from-scratch PPO with and without curriculum, U24's reduced
exploration std, U26's reward-weight variants) stayed stuck falling within
20–75 steps of 200, for its *entire* 300k–800k-step budget, no exceptions.

### Three evaluations (kept separate, per the correction above)

**1. Deterministic policy, nominal reset** — 200/200 steps, no fall.
**Learning preserves the standing solution**, matching zero-action exactly.

**2. Sampled policy, nominal reset (30 repeats)** — this is the test that
measures **action-sampling robustness**, correctly relabeled: 80.0%
full-episode survival, 20.0% fall rate, mean 190.1/200 steps. Contrast with
U23: the original (non-zero-mean) untrained policy's sampled rollouts fell
**100%** of the time in a mean of 21.3 steps. This policy, after training,
survives the full episode 80% of the time under its own sampling noise —
**exploration is tolerable at this std.**

**3. Deterministic policy vs. zero action, matched perturbed resets** — the
actual recovery question. First swept perturbation magnitude *against zero
action itself*, since the instruction warns that if both succeed equally
the test hasn't demonstrated anything: the passive position-PD baseline is
**fully robust (100% survival) up to ~3 rad/s** of initial joint-velocity
noise and only starts degrading past 4 rad/s — at 0.5 rad/s (an initial
guess) both the policy and zero action survived 100% of 20 resets,
confirmed uninformative rather than assumed adequate. Settled on **6.0
rad/s** (zero-action ~70% survival there — room to show a difference either
way), 40 matched resets:

| | full-episode survival | mean steps |
|---|---|---|
| learned policy | 70.0% | 157.1 |
| zero action (same resets) | 67.5% | 152.9 |

Per-reset head-to-head: policy better on 9, zero better on 3, tied on 28 —
**a real but modest recovery edge**, not dramatic, but a consistent,
directional signal (wins 3× more often than it loses when they differ), not
noise in either direction.

**Net assessment**: the strategy change succeeded decisively on its first,
necessary goal — preserving the known standing solution, something no
from-scratch or continued-training approach achieved across six prior
workstreams. The actual new milestone (recovery) has a real, if modest,
positive answer at this budget. Not yet done, flagged for next: expanding
the *same* perturbation family's magnitude/duration range gradually (per
instruction, not combined with payload/friction/actuator randomization);
checking this result's reliability across additional seeds (the explicit
reporting qualification applies here too — one run, one seed); or extending
training budget now that the destructive-early-fall pattern is gone and
there's visible headroom before the 200-step ceiling in both the
sampled-nominal and perturbed conditions. All artifacts — the untrained
initial checkpoint, the best and final trained checkpoints, full training
history, the exploration-std sweep, the perturbation-magnitude sweep against
zero action, and the three-evaluation report — are under
`cara_description/runs/u27_standing_init/` with `manifest.json` as the
index.

## U28 — three-seed reliability check

### Correction accepted: the "9 wins/3 losses" framing was imprecise

Survival rate was 70.0% vs. 67.5% — **28 vs. 27 successful episodes, one
additional success**. The 9/3 head-to-head count mixed two different
strengths of evidence: a **survival flip** (one side reached the full
episode, the other didn't) and a **time-to-fall-only** difference (both
fell, one merely lasted a few steps longer). `evaluate_u28.py` reports these
separately from here on, with the win criterion stated exactly: `policy
steps > zero_action steps` on the *same* reset, broken into survived/both,
flip-toward-policy, flip-toward-zero, and time-to-fall-only in each
direction.

### Discrepancy investigated first, using only existing data — no run launched

*Training fall_rate near 0 vs. 80% sampled-eval survival.* Traced to a real
methodology flaw, not a real inconsistency: U27's checkpoint-selection
metric was a single deterministic rollout capped at 200 steps, which
**saturated the moment it first hit the ceiling (iteration 5 of 195, 2.6%
of budget) and never updated again**, silently freezing an under-converged
early snapshot as "best" while training continued improving for 190 more
iterations. Confirmed directly: `u27_seed0_best.pt` (iter 5) has
`obs_norm.count=10,240` and `actor_logstd` std≈0.049; the FINAL checkpoint
(iter 195) has `obs_norm.count=399,360` (39× more data) and std≈0.041
(quieter). Re-evaluating the FINAL checkpoint on the identical sampled-
nominal test gave **100%** survival, matching the training log's near-zero
`fall_rate` exactly. Not a real discrepancy in learning — a checkpoint-
selection artifact, now fixed (below) before anything new was launched.

### Checkpoint-selection fix, and a limitation it exposed

`train_ppo.py` now scores periodic checkpoints on `validation_eval_score()`
— mean survival over 10 fixed validation seeds of *sampled* rollouts,
instead of one deterministic rollout. This no longer saturates instantly.
But a retroactive check (N=30, held out from that 10-seed set) found even
10 samples aren't powerful enough to reliably rank checkpoints once
survival is already in the 80–100% range — seed 0's "best" and "final"
checkpoints both scored a perfect 200.0/200 on the 10-sample validation
gate despite a real underlying difference (73% vs. 100% true survival).
This held for all 3 seeds: every "best"-labeled checkpoint measured 60–97%
true survival (N=30) despite passing its 10-sample gate; every **final**
checkpoint measured 100%, every time. **Resolution: the FINAL checkpoint
(end of the fixed 400k-step budget) is used for all 3 seeds in this
report — disclosed explicitly, not silently substituted for "best."** The
validation mechanism is a real improvement (it no longer locks in at
iteration 5) but its statistical power remains an open question, not
solved by this one fix.

### Three seeds, identical configuration, budget and perturbation range held fixed

`zero_mean_init=True`, `init_logstd=-3.0`, `w_action_rate=0.3`,
`desired_vx=0.0`, 400,000 steps, per instruction — seed 0 reused from U27,
seeds 1 and 2 newly trained. **Standing preservation is reproducible, not a
fluke**: all 3 seeds independently hit `fall_rate=0.0` within 8–10% of
budget (iterations 16, 19, and 17 of 195, respectively).

### Evaluation (fresh held-out perturbation seeds, never used in selection or in U27's own test)

**Untrained policy, sampled nominal (N=30)** — isolates what initialization
alone contributes: **63.3%** survival, mean 178.4/200 steps. This is well
below what any trained policy achieves (below), so training is doing real
work on top of the initialization choice, not just riding it.

| seed | det. nominal | sampled nominal (N=30) | perturbed policy (N=40, fresh) | perturbed zero-action (same resets) | survival flips (policy:zero) | mean paired step diff |
|---|---|---|---|---|---|---|
| 0 | 200/200 | 100.0% | 97.5% | 82.5% | 6 : 0 | +22.8 |
| 1 | 200/200 | 100.0% | 95.0% | 82.5% | 6 : 1 | +20.6 |
| 2 | 200/200 | 100.0% | 97.5% | 82.5% | 6 : 0 | +22.5 |

Zero-action's own survival (82.5%) is identical across all three rows by
construction — same fresh reset seeds (800–839), same passive baseline,
reused as the constant reference each time. The learned policy converts
6 of those zero-action failures into successes in every single seed, with
only 0–1 going the other way — this is now a **survival-outcome** result
(the strong form of evidence), not a time-to-fall artifact: in every seed
almost all of the head-to-head advantage is real survival flips (33/32/33
resets where both succeed anyway, then a clean +6 flip margin), with only
0–1 residual time-to-fall-only differences among what's left.

### Decision table readout

- *Standing reproducible; recovery small* — does not fire: standing is
  reproducible, but the recovery advantage is not small this time.
- ***Recovery advantage is reproducible* — fires.** All 3 seeds:
  substantially higher perturbed survival than zero action (95–97.5% vs.
  82.5%), 6 survival flips in the policy's favor per seed (vs. 0–1 the
  other way), consistent mean step advantage (+20.6 to +22.8). Markedly
  clearer than U27's original single-seed read (70.0% vs. 67.5%) — mostly
  because that comparison used the flawed early "best" checkpoint rather
  than the properly-selected final one, on a smaller, non-fresh reset set.
- *Results vary substantially across seeds* — does not fire: survival
  rates, flip counts, and step advantages are closely clustered across all
  three independent training runs.

**Decision: gradually broaden perturbations and retain held-out tests** —
not started this session, flagged for the next one. No perturbation
magnitude increase, no additional seeds, and no training-budget extension
were made this round, per instruction. All artifacts (both training runs,
the discrepancy investigation, the checkpoint-selection fix and its
disclosed limitation, and the full three-seed evaluation report) are under
`cara_description/runs/u27_standing_init/`, indexed by `manifest.json` (U27)
and `u28_manifest.json` (this workstream).

## U29 — one bounded expansion of the joint-velocity perturbation family

Scope, stated up front per instruction: these are **3 policies tested on
the same 40 reset cases**, not 3×40=120 independent disturbance cases.

**Frozen baselines**: U28's final checkpoints
(`cara_description/runs/u27_standing_init/u27_seed{0,1,2}.pt`) untouched —
used both as U29's resume source and as the "frozen U28" arm of the
3-way comparison below. Reward, action bounds, actuator settings, and
architecture all unchanged; the reset-perturbation *distribution* is the
only thing that changed.

**Mechanism** (still the same perturbation family, nothing new):
`cara_env.py`'s `reset_qvel_noise_bands` — a tuple of `(probability, std)`
pairs; `reset()` samples one per episode. `None` (default) falls back to
the single `reset_qvel_noise_std` from U27/U28, regression-checked to
reproduce every prior deterministic-reset result exactly. Initial mixture:
40% nominal, 30% "previously manageable" (3.0 rad/s, where zero action
itself is ~100% robust), 30% "modestly harder" (7.0 rad/s, one step past
U28's own 6.0 test point) — the easy/manageable 70% share never shrinks,
so recovery training can't erase standing. **Live progression**:
`train_ppo.py --perturb-curriculum` checks deterministic survival at the
hard band's current std every 10 iterations (after ≥50,000 steps there)
against 30 *fixed* validation seeds (9600–9629, disjoint from the final
held-out set) and advances the hard band by +1.0 rad/s once that check
hits 90% — propagated live to all 8 workers via `AsyncVectorEnv.call`, the
same mechanism the `desired_vx` curriculum used in U22–U24.

**Training**: resumed **exactly** (optimizer state present and restored)
from each seed's U28 final checkpoint, same fixed 400,000-step budget, no
automatic extension. Outcome: seed 0 advanced 7.0→8.0 rad/s at step
163,840, then plateaued at 73–83% survival at the new std for the rest of
budget; seeds 1 and 2 stayed at 7.0 rad/s, oscillating 70–86% and 76–83%
respectively, never reaching the 90% gate. The curriculum mechanism works;
consistent progress past 8.0 rad/s simply wasn't established within this
budget — reported as such, not extended.

### Comparison: zero action vs. frozen U28 vs. U29, by magnitude, on 40 fresh held-out resets

Fresh seeds (2000–2039) — never used in U28's own perturbation test
(100–139), U28's held-out eval (800–839), or U29's training-time gating set
(9600–9629).

| magnitude (rad/s) | zero / frozen-U28 / U29 — seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 0.0 | 100/100/100 | 100/100/100 | 100/100/100 |
| 6.0 | 87.5/95/95 | 87.5/95/97.5 | 87.5/97.5/100 |
| 8.0 | 67.5/77.5/**80** | 67.5/77.5/**82.5** | 67.5/72.5/**77.5** |
| 10.0 | 37.5/57.5/**65** | 37.5/65/65 | 37.5/60/**65** |

**Success criterion** (nominal preserved + the 6.0 rad/s range preserved +
harder band improved by >5pp somewhere): **seeds 0 and 2 meet it fully**
(deltas up to +7.5pp and consistently +5pp respectively in the 8–10 rad/s
band). **Seed 1 does not** — its 7–10 rad/s deltas are small and
mixed-sign (-2.5, +5.0, -5.0, 0.0), a wash rather than a regression.
Nominal standing and the established 6.0 rad/s range are preserved in
**all three seeds**, with no exceptions. Per instruction, this is reported
as where the improvement ends for seed 1, not smoothed into a uniform win.

### Settling check — a threshold problem caught before it became a false alarm

For every episode surviving the full 200 steps, mean `|tilt|` and mean
`|base angular velocity|` were computed over the last 20 steps (0.4s). An
initial pass/fail rule (tilt < 5°, angvel < 1.0 rad/s) produced a
confusing result — e.g. seed 1's frozen-U28 policy showed **0/40 "settled"**
at pure nominal reset, which reads as alarming on its own. Investigated
before reporting it that way: the actual tail tilt was 6.11° (just over the
cutoff) with tail angular velocity 0.056 rad/s — essentially motionless.
**The threshold was the problem, not the behavior.**

Reported instead as two separate continuous quantities. Across all trained
policies (frozen U28 and U29) and all 3 seeds, at every magnitude: tail
angular velocity stays consistently low (0.02–0.8 rad/s — genuinely
stopped, not oscillating, and often *lower* than zero action's own residual
velocity). But tail tilt sits consistently at **2.4–6.1°**, versus pure zero
action's **~0.4°** at the same magnitudes. **Conclusion: survivors are not
concealing ongoing instability — but training (both U28 and U29) has
shifted the deterministic policy's resting equilibrium to a measurably
non-upright pose.** A real, disclosed side effect, not a dramatic one, and
not something the survival numbers alone would have shown.

**Net assessment**: U29 preserved nominal standing and the U28-established
6.0 rad/s recovery range in all 3 seeds, and produced a real if
seed-dependent improvement in the 7–10 rad/s band (2 of 3 clearly better, 1
of 3 a wash with no regression). The curriculum mechanism itself works
(seed 0's clean 7.0→8.0 advance proves it); consistent gains past 8.0 rad/s
weren't established within budget, and none was taken automatically. This
remains recovery from initial-state disturbances only — not adaptation to
changed masses or weakened actuators. Per instruction, the next direction
is **low-speed walking**, not another round of standing-disturbance
escalation. All artifacts (3 resumed checkpoints, training logs, the
by-magnitude evaluation script and its full JSON output, and the settling
analysis) are under `cara_description/runs/u29_perturb_curriculum/` with
`manifest.json` as the index.

### Correction: "0.02–0.8 rad/s, genuinely stopped" overstated the low end of that range

`0.8 rad/s` is about 46°/s — the settling-check writeup above should not
have implied every survivor in that range had actually stopped moving.
Retracted as a blanket "stopped" claim. The corrected framing: tail angular
velocity varies across that range, and the higher end (~0.5–0.8 rad/s)
describes **surviving with residual motion**, not full quiescence — the
low tilt-vs-velocity comparison (velocity often lower than zero action's
own residual) still holds and still argues against *oscillatory*
instability, but "stopped" was too strong a word for the whole range. Does
not reopen standing training; noted for how future settling checks should
describe their own numbers.

## Reward audit: a real sign bug found and fixed before U30

Before planning U30, a specific inconsistency was flagged between the
reward code and its own comment:

```python
r_upright = float(self._projected_gravity()[2] + 1.0)  # 0 upright, -2 fully tipped
```

With this project's own stated convention (`_projected_gravity()` returns
world `-Z` in the pelvis frame — upright gives `z=-1`, inverted gives
`z=+1`), this expression gives **0 at upright and +2 at full inversion** —
the *opposite* of its comment, which describes `-2` at full inversion.
Checked directly at synthetic tilts before touching any code, not assumed
from the comment mismatch alone:

| tilt | `projected_gravity_z` | formula (as shipped) |
|---|---|---|
| 0° | -1.0000 | +0.0000 |
| 10° | -0.9848 | +0.0152 |
| 20° | -0.9397 | +00603 |
| 90° | -0.0000 | +1.0000 |
| 180° | +1.0000 | +2.0000 |

Confirmed: the shipped formula **rewards leaning and inversion**,
monotonically, growing toward +2 as the robot tips further over — a real
implementation bug, not a stale comment. **Fixed** by negating it
(`r_upright = -float(self._projected_gravity()[2] + 1.0)`), restoring
`0` at upright and `-2` at full inversion, matching the comment. `reward_version`
bumped to `"v4_upright_sign_fix"` — every U20–U29 checkpoint was trained
under the buggy `v3` sign, disclosed here, not retroactively relabeled.

**Re-ran `reward_audit.py` immediately** (not skipped): still **PASS**,
margin unchanged (100.8%) — expected, since the bug's magnitude at the
tilts checked there (nominal standing, ~0.4–0.8°) is tiny. At U29's own
measured tail tilts (2.4–6.1°) the bug was worth roughly +0.001 to +0.006
raw (weighted ≈0.0005–0.003/step) — real, wrong-signed, but small enough
that, per instruction, it should not be inferred to explain the whole
equilibrium-tilt shift on its own. It would have compounded materially
closer to an actual fall (where tilt is large), which is exactly the
regime where an upright incentive matters most — reason enough to fix
before any further training, independent of how much it explains U29's
specific finding.

## U30 — first locomotion attempt

**Command-normalization check, done before training, not assumed safe.**
The `desired_vx` observation (index 30 of 43) held fixed at `0.0` through
*every* U27–U29 run, so its adaptively-learned variance converges toward
the epsilon floor. Measured directly on a U29 checkpoint before training:
`mean=0.0, var=1.25e-14`. A raw `0.03` m/s command would normalize to
`(0.03-0)/1.12e-7 ≈ 300`, clipped to the ±10 ceiling — fully saturated,
indistinguishable from any other nonzero command. **Fixed**: `RunningNorm`
gained `fixed_scale_dims` (obs index → constant) — that one dimension now
normalizes as `raw/scale` regardless of the adaptive stats, `--vx-obs-scale
0.5` for U30, overriding whatever the resumed checkpoint's degenerate stats
said. Smoke-tested before the real run: round-trips correctly through
save/load and correctly overrides inherited stats.

**Setup**: warm-started (exact resumption, optimizer state restored) from
each seed's U29 final checkpoint. Nominal resets only — the perturbation
curriculum removed entirely, per instruction. `desired_vx_bands=(0.5:0.0,
0.5:0.03)`, one drawn per episode and held constant within it. Reward
(now `v4`), action bounds, actuator settings, and the exploration schedule
all otherwise untouched — `init_logstd` is ignored on resume, so the
*actual* learned std simply continues from wherever U29 left it. Fixed
400,000-step budget per seed, no automatic extension. Incentive check
(not proof): at `vx=0.03`, ideal tracking under `w_vel=10` gives a real
+0.3/step advantage over standing, before movement penalties.

**Training**: standing preserved throughout in all 3 seeds — `fall_rate`
stayed near 0.0 for the full budget, no destabilization signal, nothing to
stop early for.

### Evaluation — the milestone is not met, reported plainly

`evaluate_u30.py`, per the required table:

| seed | zero-command | commanded 0.03 m/s | swing steps (L/R) | foot clearance | verdict |
|---|---|---|---|---|---|
| 0 | 200/200, retained | achieved 0.0063 m/s, +0.022m | 0/0 | 1.9/0.9mm | **STANDING** |
| 1 | 200/200, retained | achieved 0.0138 m/s, +0.055m | 0/0 | 1.3/2.7mm | **SLIDING** |
| 2 | 200/200, retained | achieved -0.0004 m/s, -0.001m | 0/0 | 1.8/1.8mm | **STANDING** |

Zero genuine swing phases (≥60ms airborne) on either foot, in any of the 6
seed×condition combinations. Foot clearance is 1–3mm everywhere — noise-
level vibration, not a real lift. Seed 1's forward drift happens *without*
any genuine airborne phase — weight-shifting/sliding, correctly not
credited as stepping. Torque use (27.5–36.0% mean saturation) and joint
tracking (0.008–0.010 rad mean error) are unremarkable across every
condition — movement, what little occurred, was never actuator-limited.
**The first walking milestone is not met in any seed.** Standing was
retained cleanly in every case — per instruction, this is reported as its
own outcome, not re-filed as a balance failure, because it manifestly
isn't one.

### Exploration and cost-of-moving, checked as instructed

Final exploration std, measured directly: **0.0275–0.0291** (normalized
action-space units) across the 3 seeds — roughly *half* of U27's already-
conservative starting value (0.0498), having decayed further through
U28's and U29's recovery training, which specifically reinforced smooth,
quiet corrections (recall U26 raised `w_action_rate` precisely to suppress
jitter). At `hip_pitch`'s own scale (0.42 rad range), that's ≈0.012 rad —
**about 0.7°** of 1-sigma per-step swing. Far too small for PPO's
on-policy sampling to ever produce a clean, deliberate leg lift, regardless
of whether the reward would pay for one if sampled. **Leading explanation:
exploration magnitude, not reward incentive or actuator authority** — the
+0.3/step tracking advantage is real and untouched, and torque/tracking
stayed unremarkable throughout, but three successive generations of
standing-and-recovery training optimized specifically for quiet correction
appear to have suppressed exploration below the level stepping would ever
require to be *discovered*. Not yet checked: the actual action-rate/effort
cost of a stepping-scale motion against that +0.3/step benefit (would need
a scripted-gait probe, in the spirit of U25's actuator-response tests) —
flagged, not run this session.

**Net assessment**: the infrastructure fixes (upright sign, command
normalization) worked cleanly — standing stayed intact under both
commands in every seed, nothing destabilized. The walking task itself
wasn't discovered. This reads as an exploration-budget problem specific to
this policy lineage, not a reward or actuator problem — pointing toward
either a deliberate, bounded exploration boost targeted at the walking
task, or a fresh initialization strategy for locomotion analogous to U27's
for standing. Neither started here — flagged for the next decision. All
artifacts (3 resumed checkpoints, training logs, the per-seed evaluation
script and its full JSON, and the exploration-std measurements) are under
`cara_description/runs/u30_walk_command/` with `manifest.json` as the
index.