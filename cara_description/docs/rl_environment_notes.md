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

Artifacts from this run: `/tmp/ars_run1.npz` (w_vel=1.0, "safe standing"),
`/tmp/ars_run2.npz` (w_vel=10.0, unstable) — both in the session scratch
area, not committed, not moved into the repo (informative negative results,
not an accepted configuration).
