#!/usr/bin/env python3
"""U21 -- PPO baseline for CaraWalkEnv, run in a SEPARATE venv (`.venv-rl`,
Python 3.11 + torch + gymnasium) from the rest of the project. `cara_env.py`
itself has no torch/gymnasium import anywhere -- this script is the only
place those dependencies are used, per the instruction to keep CaraWalkEnv
independent of training dependencies.

Structure follows CleanRL's ppo_continuous_action.py (Huang et al.) -- an
established, widely-used single-file PPO reference for continuous control --
adapted to CaraWalkEnv: a small tanh MLP actor-critic with a state-
independent log-std, running observation normalization (reward is
deliberately NOT normalized, to keep returns comparable with
reward_audit.py's raw-reward analysis), GAE with correctly separated
termination/truncation bootstrapping, and vectorized rollout collection via
gymnasium.vector.AsyncVectorEnv (NEXT_STEP autoreset mode -- verified
empirically before relying on it: the terminal observation is returned on
the done step itself, not swallowed by an immediate reset, which is exactly
what correct timeout bootstrapping needs).

Usage (from the .venv-rl venv):
    .venv-rl/bin/python3 cara_description/scripts/train_ppo.py \
        --total-timesteps 300000 --num-envs 8 --seed 0 --out /path/to/ppo_run.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cara_env import CaraWalkEnv, CaraWalkEnvConfig  # noqa: E402

try:
    import gymnasium as gym
    import torch
    import torch.nn as nn
    from torch.distributions.normal import Normal
except ImportError as e:
    raise SystemExit(
        f"train_ppo.py needs torch+gymnasium, which live in a SEPARATE venv "
        f"(.venv-rl), not the project's main .venv (which stays dependency-"
        f"light per the project's own choice not to add an ML framework "
        f"there). Run this with .venv-rl/bin/python3. Original error: {e}"
    )


class CaraWalkGymEnv(gym.Env):
    """Thin Gymnasium adapter around CaraWalkEnv. All the actual environment
    logic (model, action mapping, reward, termination) lives in cara_env.py
    and is untouched by this wrapper -- it only translates dtypes/spaces."""

    def __init__(self, cfg: CaraWalkEnvConfig):
        super().__init__()
        self.env = CaraWalkEnv(cfg)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.env.n_act,), dtype=np.float32)
        obs_dim = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

    def set_desired_vx(self, vx):
        """Curriculum hook -- called remotely via AsyncVectorEnv.call() from
        the main training process so every worker's environment picks up a
        new target speed without tearing down and recreating the vector env
        (which would cost a full MJCF recompile per worker)."""
        self.env.cfg.desired_vx = vx
        return vx

    def set_harder_band_std(self, new_std):
        """U29 perturbation-curriculum hook: the reset_qvel_noise_bands
        mixture's LAST band (by convention, the hardest one) gets its std
        replaced in place, live, the same way set_desired_vx works. The
        easier bands (nominal, previously-manageable) and their
        probabilities are untouched -- only the hard band's magnitude moves."""
        bands = list(self.env.cfg.reset_qvel_noise_bands)
        prob, _old_std = bands[-1]
        bands[-1] = (prob, new_std)
        self.env.cfg.reset_qvel_noise_bands = tuple(bands)
        return new_std


def make_env(cfg: CaraWalkEnvConfig, seed: int):
    def thunk():
        e = CaraWalkGymEnv(cfg)
        e.reset(seed=seed)
        return e
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def quick_eval_steps(env, agent, obs_norm, max_steps, seed=0):
    """One deterministic-mean rollout from a nominal (unperturbed) reset.
    Kept for informational logging only -- NOT used for checkpoint
    selection since U28: a single deterministic rollout saturates at
    max_steps almost immediately (U27 hit it by iteration 5 of 195) and
    stops discriminating between checkpoints for the entire rest of
    training, silently freezing in an early, under-converged snapshot as
    "best." See validation_eval_score below."""
    obs, _ = env.reset(seed=seed)
    steps = 0
    for _ in range(max_steps):
        obs_n = obs_norm.normalize(obs.astype("float64"))
        with torch.no_grad():
            mean = agent.actor_mean(torch.as_tensor(obs_n, dtype=torch.float32))
        obs, reward, terminated, truncated, info = env.step(mean.numpy())
        steps += 1
        if terminated or truncated:
            break
    return steps


# Fixed validation seeds for periodic checkpoint SELECTION during training --
# frozen for the life of a run, and deliberately disjoint from any reset
# seeds used in a later held-out evaluation (U28 point 2: "use a separate
# validation set to select checkpoints; keep the final evaluation resets
# unseen during selection").
VALIDATION_TORCH_SEEDS = list(range(9000, 9010))  # 10 repeats


def validation_eval_score(env, agent, obs_norm, max_steps, reset_seed=0):
    """Mean survival steps over VALIDATION_TORCH_SEEDS sampled rollouts from
    a nominal reset -- a continuous score that does not saturate the moment
    the deterministic case reaches max_steps (sampling noise keeps giving it
    room to move: a policy that's merely adequate under sampling will still
    show a lower mean than one that's robustly good). Used for checkpoint
    SELECTION; quick_eval_steps (deterministic) stays as an informational
    log column only."""
    total = 0
    for torch_seed in VALIDATION_TORCH_SEEDS:
        gen = torch.Generator().manual_seed(torch_seed)
        obs, _ = env.reset(seed=reset_seed)
        steps = 0
        for _ in range(max_steps):
            obs_n = obs_norm.normalize(obs.astype("float64"))
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_n, dtype=torch.float32)
                mean = agent.actor_mean(obs_t)
                std = torch.exp(agent.actor_logstd.view(-1))
                noise = torch.randn(mean.shape, generator=gen)
                action = (mean + noise * std).numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        total += steps
    return total / len(VALIDATION_TORCH_SEEDS)


# U29: fixed, held-out reset seeds for gating perturbation-curriculum
# advancement -- distinct from VALIDATION_TORCH_SEEDS (different purpose:
# nominal-reset sampling robustness) and from any final-evaluation seed
# range, per "use a separate validation set to control progression... keep
# the final held-out set out of that decision." N=30, not 10 -- U28 found
# 10 samples cannot reliably resolve survival rates in the 80-100% range.
PERTURB_VALIDATION_RESET_SEEDS = list(range(9600, 9630))


def perturb_band_survival(env, agent, obs_norm, max_steps, band_std):
    """Deterministic-mean survival rate at a FIXED perturbation magnitude
    (band_std), over PERTURB_VALIDATION_RESET_SEEDS -- used only to gate
    curriculum advancement, never as the final reported result."""
    survived = 0
    for reset_seed in PERTURB_VALIDATION_RESET_SEEDS:
        env.cfg.reset_qvel_noise_bands = None
        env.cfg.reset_qvel_noise_std = band_std
        obs, _ = env.reset(seed=reset_seed)
        steps = 0
        for _ in range(max_steps):
            obs_n = obs_norm.normalize(obs.astype("float64"))
            with torch.no_grad():
                mean = agent.actor_mean(torch.as_tensor(obs_n, dtype=torch.float32))
            obs, reward, terminated, truncated, info = env.step(mean.numpy())
            steps += 1
            if terminated or truncated:
                break
        if not terminated and steps >= max_steps:
            survived += 1
    return survived / len(PERTURB_VALIDATION_RESET_SEEDS)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, init_logstd=-0.5, zero_mean_init=False):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        final_actor_layer = nn.Linear(64, act_dim)
        if zero_mean_init:
            # U27: exact zero, not just small (std=0.01 orthogonal still left
            # a nonzero, obs-dependent mean -- U23 measured its action
            # magnitude at 0.0025, close to but not identical to zero
            # action). Weight AND bias zeroed means actor_mean(x) == 0 for
            # EVERY input, so the deterministic policy reproduces the
            # zero-action standing baseline exactly, not approximately --
            # verified in U27's pipeline check, not assumed.
            nn.init.zeros_(final_actor_layer.weight)
            nn.init.zeros_(final_actor_layer.bias)
        else:
            layer_init(final_actor_layer, std=0.01)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            final_actor_layer,
        )
        # U23 found that init_logstd=-0.5 (std=0.6065) gives a ~72% chance
        # ANY of the 12 joints' sampled action exceeds the [-1,1] bound on a
        # given step, at initialization -- before any learning happens. A
        # freshly-initialized policy's deterministic MEAN already matches
        # the zero-action standing baseline (verified in U23); sampling
        # noise alone was shown to destroy that in ~21 steps, 100% of the
        # time. Lower init_logstd (e.g. -1.0 -> ~8% clip chance) keeps the
        # action bounds (action_range_frac) themselves unchanged -- this
        # reduces exploration MAGNITUDE, not exploration RANGE.
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim) + init_logstd)

    def value(self, x):
        return self.critic(x)

    def act(self, x, action=None):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, self.critic(x)


class RunningNorm:
    """Online mean/std normalizer applied to observations (not rewards --
    see module docstring). Batched Welford/Chan update, applied centrally on
    the vectorized rollout's observation batches rather than per-worker.

    U30: fixed_scale_dims (dict: obs_index -> scale) opts specific
    dimensions OUT of the adaptive mean/var normalization entirely, using
    raw/scale instead. Necessary, not cosmetic: every U27-U29 standing run
    held desired_vx fixed at 0.0, so that observation dimension's LEARNED
    variance converges toward the epsilon floor (measured: ~1.25e-14 in
    U29's checkpoints). Normalizing a newly-introduced 0.03 m/s command
    through that near-zero variance gives (0.03-0)/sqrt(1.25e-14) ~= 300,
    clipped to the +/-10 ceiling -- a saturated, uninformative input
    indistinguishable from any other nonzero command. A fixed, documented
    scale sidesteps this regardless of what the adaptive stats happen to
    hold for that dimension."""

    def __init__(self, dim, clip=10.0, eps=1e-8, fixed_scale_dims=None):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = eps
        self.clip = clip
        self.fixed_scale_dims = dict(fixed_scale_dims) if fixed_scale_dims else {}

    def update(self, x):
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot

    def normalize(self, x):
        out = np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)
        for dim_idx, scale in self.fixed_scale_dims.items():
            out[..., dim_idx] = np.clip(x[..., dim_idx] / scale, -self.clip, self.clip)
        return out

    def state_dict(self):
        return dict(mean=self.mean, var=self.var, count=self.count, fixed_scale_dims=self.fixed_scale_dims)

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]
        self.fixed_scale_dims = dict(d.get("fixed_scale_dims") or {})


def train(args):
    import gymnasium.vector as vector

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vx_stages = [float(v) for v in args.vx_stages.split(",")] if args.curriculum else None
    initial_vx = vx_stages[0] if args.curriculum else args.desired_vx
    perturb_bands = None
    if args.perturb_curriculum:
        perturb_bands = tuple((float(p), float(s)) for p, s in
                               (pair.split(":") for pair in args.perturb_bands.split(",")))
        assert abs(sum(p for p, _ in perturb_bands) - 1.0) < 1e-6, "perturb-bands probabilities must sum to 1"
    vx_bands = None
    if args.desired_vx_bands:
        vx_bands = tuple((float(p), float(v)) for p, v in
                          (pair.split(":") for pair in args.desired_vx_bands.split(",")))
        assert abs(sum(p for p, _ in vx_bands) - 1.0) < 1e-6, "desired-vx-bands probabilities must sum to 1"
    cfg = CaraWalkEnvConfig(episode_seconds=args.episode_seconds, desired_vx=initial_vx,
                             w_action_rate=args.w_action_rate, reset_qvel_noise_bands=perturb_bands,
                             desired_vx_bands=vx_bands)
    if vx_bands:
        print(f"desired_vx MIXTURE ON: {vx_bands} -- one drawn per episode, held constant within it.")
    env_fns = [make_env(cfg, args.seed + i) for i in range(args.num_envs)]
    envs = vector.AsyncVectorEnv(env_fns, autoreset_mode=vector.AutoresetMode.NEXT_STEP)
    episode_max_steps = int(round(args.episode_seconds * cfg.control_hz))

    # A deep copy, not the same cfg object used to build env_fns: some
    # periodic checks (perturb_band_survival) mutate cfg.reset_qvel_noise_*
    # fields on whatever object they're given, and that must never leak
    # into the checkpoint's saved env_config or (were it shared) the actual
    # training workers.
    import copy as _copy
    eval_env = CaraWalkEnv(_copy.deepcopy(cfg)) if (args.eval_every > 0 or args.perturb_curriculum) else None
    best_validation_score = -1.0

    stage_idx = 0
    stage_start_step = 0
    if args.curriculum:
        print(f"curriculum ON: stages={vx_stages}  advance when the last "
              f"{args.advance_window} episodes average >= {args.advance_len_frac:.0%} "
              f"of {episode_max_steps} steps AND fall_rate <= {args.advance_max_fall_rate:.0%}, "
              f"after at least {args.stage_min_steps} steps in the current stage.")

    perturb_hard_std = perturb_bands[-1][1] if args.perturb_curriculum else None
    perturb_stage_start_step = 0
    if args.perturb_curriculum:
        print(f"perturbation curriculum ON: bands={perturb_bands}  advance the hard band by "
              f"+{args.perturb_advance_step} rad/s once deterministic survival at that std, over "
              f"{len(PERTURB_VALIDATION_RESET_SEEDS)} held-out reset seeds, reaches "
              f"{args.perturb_advance_target:.0%}, after at least {args.perturb_min_steps} steps at "
              f"the current std.")

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    device = torch.device("cpu")

    agent = ActorCritic(obs_dim, act_dim, init_logstd=args.init_logstd,
                         zero_mean_init=args.zero_mean_init).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # desired_vx lives at this fixed index in the observation vector (see
    # CaraWalkEnv.desired_vx_obs_index) -- computed the same way here since
    # no single-env instance exists yet at this point in train().
    desired_vx_obs_idx = 2 * act_dim + 3 + 3
    fixed_scale_dims = {desired_vx_obs_idx: args.vx_obs_scale} if args.vx_obs_scale is not None else None
    obs_norm = RunningNorm(obs_dim, fixed_scale_dims=fixed_scale_dims)

    resume_is_exact = False
    if args.resume_from:
        # Load the FULL learned state -- including actor_logstd, which is a
        # trainable nn.Parameter, not fixed at init_logstd. Resuming must
        # continue the actual learned exploration distribution, not silently
        # reset it back to --init-logstd's starting value.
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        agent.load_state_dict(ckpt["model_state"])
        obs_norm.load_state_dict(ckpt["obs_norm"])
        if fixed_scale_dims:
            # This run's --vx-obs-scale intent takes precedence over
            # whatever the resumed checkpoint's obs_norm had (U27-U29
            # checkpoints predate this feature entirely and would silently
            # load fixed_scale_dims={}, reintroducing the saturation bug).
            obs_norm.fixed_scale_dims = dict(fixed_scale_dims)
            print(f"desired_vx observation (index {desired_vx_obs_idx}) given a FIXED scale of "
                  f"{args.vx_obs_scale} m/s, overriding whatever the resumed obs_norm had for that "
                  f"dimension (U27-U29 checkpoints saw only desired_vx=0.0, so its learned variance "
                  f"is near the epsilon floor -- adaptive normalization there would saturate any "
                  f"nonzero command).")
        with torch.no_grad():
            resumed_logstd = agent.actor_logstd.detach().numpy().ravel()
        opt_note = "optimizer state NOT restored (checkpoint predates optimizer-state saving) -- this is a WARM START, not an exact resumption"
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            resume_is_exact = True
            opt_note = "optimizer state (Adam moments) restored -- this is an EXACT resumption"
        print(f"resumed from {args.resume_from}: actor_logstd={resumed_logstd} "
              f"(std={np.exp(resumed_logstd)}) -- --init-logstd={args.init_logstd} was IGNORED "
              f"since a checkpoint was loaded. {opt_note}.")

    batch_size = args.num_envs * args.num_steps
    num_iterations = args.total_timesteps // batch_size

    obs_buf = np.zeros((args.num_steps, args.num_envs, obs_dim), dtype=np.float32)
    actions_buf = np.zeros((args.num_steps, args.num_envs, act_dim), dtype=np.float32)
    logprobs_buf = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    rewards_buf = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    terminated_buf = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    truncated_buf = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    values_buf = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    next_values_buf = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)

    ep_returns = np.zeros(args.num_envs)
    ep_lengths = np.zeros(args.num_envs, dtype=int)
    completed_returns, completed_lengths, completed_fell = [], [], []

    next_obs_raw, _ = envs.reset(seed=args.seed)
    history = []
    global_step = 0
    t_start = time.time()

    for it in range(1, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (it - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_norm.update(next_obs_raw)
            obs_n = obs_norm.normalize(next_obs_raw)
            obs_buf[step] = obs_n

            with torch.no_grad():
                obs_t = torch.as_tensor(obs_n, dtype=torch.float32)
                action, logprob, _, value = agent.act(obs_t)
            action_np = action.numpy()
            action_clipped = np.clip(action_np, -1.0, 1.0)

            next_obs_raw, reward, terminated, truncated, infos = envs.step(action_clipped)

            actions_buf[step] = action_np
            logprobs_buf[step] = logprob.numpy()
            rewards_buf[step] = reward
            terminated_buf[step] = terminated.astype(np.float32)
            truncated_buf[step] = truncated.astype(np.float32)
            values_buf[step] = value.squeeze(-1).numpy()

            # NEXT_STEP autoreset: next_obs_raw here IS the true terminal
            # observation for any env with terminated|truncated True at this
            # step (verified empirically -- see module docstring). Bootstrap
            # with the critic's value of that real terminal observation for
            # truncation; use 0 (implicitly, via terminated mask below) for
            # a real fall.
            with torch.no_grad():
                term_obs_n = obs_norm.normalize(next_obs_raw)
                next_values_buf[step] = agent.value(
                    torch.as_tensor(term_obs_n, dtype=torch.float32)).squeeze(-1).numpy()

            ep_returns += reward
            ep_lengths += 1
            done = terminated | truncated
            for i in np.nonzero(done)[0]:
                completed_returns.append(float(ep_returns[i]))
                completed_lengths.append(int(ep_lengths[i]))
                completed_fell.append(bool(terminated[i]))
                ep_returns[i] = 0.0
                ep_lengths[i] = 0

        # ---- GAE ----
        # next_values_buf[t] = critic(next_obs_raw) computed fresh every
        # step (above), which is exactly V(s_{t+1}) whether that next
        # observation continues the same episode or is the genuine terminal
        # observation of a truncated one (NEXT_STEP autoreset guarantees
        # next_obs_raw IS that real terminal observation, not a reset).
        # `terminated` forces the bootstrap to 0 (a real fall has no
        # continuation value); `truncated` uses the real critic bootstrap
        # instead of 0 -- this is the "correctly handled timeouts" the
        # review guidance asked for. `continues` breaks the lambda-return
        # recursion at ANY episode boundary (real fall or timeout), since
        # the next buffer row belongs to an unrelated, freshly-reset episode.
        advantages = np.zeros_like(rewards_buf)
        lastgaelam = np.zeros(args.num_envs)
        for t in reversed(range(args.num_steps)):
            next_value = np.where(terminated_buf[t] > 0, 0.0, next_values_buf[t])
            continues = 1.0 - np.maximum(terminated_buf[t], truncated_buf[t])
            delta = rewards_buf[t] + args.gamma * next_value - values_buf[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * continues * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values_buf

        b_obs = torch.as_tensor(obs_buf.reshape(-1, obs_dim))
        b_actions = torch.as_tensor(actions_buf.reshape(-1, act_dim))
        b_logprobs = torch.as_tensor(logprobs_buf.reshape(-1))
        b_advantages = torch.as_tensor(advantages.reshape(-1))
        b_returns = torch.as_tensor(returns.reshape(-1))
        b_values = torch.as_tensor(values_buf.reshape(-1))

        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        idxs = np.arange(batch_size)
        minibatch_size = batch_size // args.num_minibatches
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch_size):
                mb = idxs[start:start + minibatch_size]
                mb_obs = b_obs[mb].float()
                _, newlogprob, entropy, newvalue = agent.act(mb_obs, b_actions[mb].float())
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_clipped = b_values[mb] + torch.clamp(newvalue - b_values[mb], -args.clip_coef, args.clip_coef)
                    v_loss_unclipped = (newvalue - b_returns[mb]) ** 2
                    v_loss_clipped = (v_clipped - b_returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        elapsed = time.time() - t_start
        window = args.advance_window
        recent_returns = completed_returns[-window:] if completed_returns else [float("nan")]
        recent_lengths = completed_lengths[-window:] if completed_lengths else [0]
        recent_fell = completed_fell[-window:] if completed_fell else [None]
        mean_len_frac = float(np.mean(recent_lengths)) / episode_max_steps if recent_lengths else 0.0
        fall_rate = float(np.mean(recent_fell)) if recent_fell[0] is not None else None
        # actor_logstd is an nn.Parameter -- trainable, and NOT held fixed at
        # init_logstd. Record it every iteration: init_logstd only sets the
        # starting value, and "lower at init" vs "lower throughout training"
        # are different interventions that look identical if only the final
        # checkpoint is inspected.
        with torch.no_grad():
            logstd_now = agent.actor_logstd.detach().numpy().ravel()
        row = dict(iter=it, global_step=global_step,
                   mean_ep_return=float(np.mean(recent_returns)),
                   mean_ep_length=float(np.mean(recent_lengths)),
                   fall_rate=fall_rate,
                   logstd_mean=float(logstd_now.mean()), logstd_min=float(logstd_now.min()),
                   logstd_max=float(logstd_now.max()),
                   pg_loss=float(pg_loss.item()), v_loss=float(v_loss.item()),
                   clipfrac=float(np.mean(clipfracs)), elapsed_s=round(elapsed, 1))
        if args.curriculum:
            row["stage_idx"] = stage_idx
            row["desired_vx"] = vx_stages[stage_idx]
        history.append(row)
        stage_note = f"  stage={stage_idx}(vx={vx_stages[stage_idx]:.2f})" if args.curriculum else ""
        eval_note = ""
        if args.eval_every > 0 and it % args.eval_every == 0:
            # Deterministic figure kept for the log only (informational --
            # saturates almost immediately, see quick_eval_steps docstring).
            eval_steps = quick_eval_steps(eval_env, agent, obs_norm, episode_max_steps)
            row["eval_steps_nominal_deterministic"] = eval_steps
            # Validation score (mean steps over VALIDATION_TORCH_SEEDS sampled
            # rollouts) is what actually gates "best" -- continuous, does not
            # saturate the moment deterministic survival hits the ceiling.
            val_score = validation_eval_score(eval_env, agent, obs_norm, episode_max_steps)
            row["validation_score"] = val_score
            eval_note = f"  eval_steps={eval_steps}  val_score={val_score:.1f}"
            if val_score > best_validation_score:
                best_validation_score = val_score
                eval_note += " (NEW BEST)"
                if args.out:
                    best_path = args.out.rsplit(".", 1)[0] + "_best.pt"
                    best_cfg_dict = dict(cfg.__dict__)
                    if args.perturb_curriculum:
                        bands = list(best_cfg_dict["reset_qvel_noise_bands"])
                        bands[-1] = (bands[-1][0], perturb_hard_std)
                        best_cfg_dict["reset_qvel_noise_bands"] = tuple(bands)
                    torch.save(dict(model_state=agent.state_dict(), obs_norm=obs_norm.state_dict(),
                                     optimizer_state=optimizer.state_dict(),
                                     env_config=json.loads(json.dumps(best_cfg_dict)), args=vars(args),
                                     best_validation_score=best_validation_score,
                                     best_eval_steps_deterministic=eval_steps, best_at_iter=it,
                                     best_at_global_step=global_step,
                                     perturb_hard_std_at_save=perturb_hard_std if args.perturb_curriculum else None
                                     ), best_path)

        print(f"iter {it:4d}/{num_iterations}  step={global_step:8d}  "
              f"ep_return={row['mean_ep_return']:8.2f}  ep_len={row['mean_ep_length']:6.1f}  "
              f"fall_rate={row['fall_rate']}  logstd={row['logstd_mean']:+.3f}  "
              f"clipfrac={row['clipfrac']:.2f}  t={elapsed:6.1f}s{stage_note}{eval_note}")

        if (args.curriculum and stage_idx < len(vx_stages) - 1
                and (global_step - stage_start_step) >= args.stage_min_steps
                and len(completed_lengths) >= window
                and mean_len_frac >= args.advance_len_frac
                and fall_rate is not None and fall_rate <= args.advance_max_fall_rate):
            stage_idx += 1
            new_vx = vx_stages[stage_idx]
            cfg.desired_vx = new_vx
            envs.call("set_desired_vx", new_vx)
            stage_start_step = global_step
            # A stage transition changes the task the observation/reward is
            # describing (a different desired_vx observation input and a
            # different vel-tracking target) -- the completed-episode window
            # from the OLD stage is not a fair measurement of the new one.
            completed_returns.clear()
            completed_lengths.clear()
            completed_fell.clear()
            print(f"  >>> CURRICULUM ADVANCE at step {global_step}: stage {stage_idx-1} -> {stage_idx}, "
                  f"desired_vx {vx_stages[stage_idx-1]:.3f} -> {new_vx:.3f}")

        if (args.perturb_curriculum and it % args.perturb_check_every == 0
                and (global_step - perturb_stage_start_step) >= args.perturb_min_steps):
            surv = perturb_band_survival(eval_env, agent, obs_norm, episode_max_steps, perturb_hard_std)
            row["perturb_hard_std"] = perturb_hard_std
            row["perturb_hard_survival"] = surv
            print(f"  [perturb check] hard_std={perturb_hard_std:.2f} rad/s  "
                  f"survival={surv:.1%} (N={len(PERTURB_VALIDATION_RESET_SEEDS)}, target={args.perturb_advance_target:.0%})")
            if surv >= args.perturb_advance_target:
                new_hard_std = perturb_hard_std + args.perturb_advance_step
                envs.call("set_harder_band_std", new_hard_std)
                print(f"  >>> PERTURBATION ADVANCE at step {global_step}: hard band "
                      f"{perturb_hard_std:.2f} -> {new_hard_std:.2f} rad/s")
                perturb_hard_std = new_hard_std
                perturb_stage_start_step = global_step

    envs.close()

    if args.curriculum and stage_idx < len(vx_stages) - 1:
        print(f"\nCURRICULUM DID NOT COMPLETE: stuck at stage {stage_idx} "
              f"(desired_vx={vx_stages[stage_idx]:.3f}) of {len(vx_stages)} stages "
              f"after the full {args.total_timesteps} step budget. This is itself the "
              f"result -- report where it got stuck, not just whether it finished.")
    elif args.curriculum:
        print(f"\nCURRICULUM COMPLETED: reached the final stage "
              f"(desired_vx={vx_stages[-1]:.3f}) within budget.")

    if args.perturb_curriculum:
        print(f"\nPERTURBATION CURRICULUM ended at hard_std={perturb_hard_std:.2f} rad/s "
              f"after the full {args.total_timesteps} step budget (no automatic extension). "
              f"This is itself the result.")

    if args.out:
        if args.perturb_curriculum:
            # cfg itself (unlike each worker's own copy) was never mutated
            # by envs.call("set_harder_band_std", ...) -- update it here so
            # the saved env_config reflects the ACTUAL final bands used,
            # not the initial ones.
            bands = list(cfg.reset_qvel_noise_bands)
            bands[-1] = (bands[-1][0], perturb_hard_std)
            cfg.reset_qvel_noise_bands = tuple(bands)
        torch.save(dict(model_state=agent.state_dict(), obs_norm=obs_norm.state_dict(),
                         optimizer_state=optimizer.state_dict(),
                         env_config=json.loads(json.dumps(cfg.__dict__)), args=vars(args),
                         curriculum_final_stage_idx=stage_idx if args.curriculum else None,
                         curriculum_vx_stages=vx_stages,
                         perturb_final_hard_std=perturb_hard_std if args.perturb_curriculum else None,
                         resumed_from=args.resume_from, resume_was_exact=resume_is_exact), args.out)
        with open(args.out.rsplit(".", 1)[0] + "_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"saved checkpoint -> {args.out}")

    return agent, obs_norm, history


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total-timesteps", type=int, default=300_000)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=256, help="rollout length per env per iteration")
    ap.add_argument("--num-minibatches", type=int, default=32)
    ap.add_argument("--update-epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--anneal-lr", action="store_true", default=True)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-coef", type=float, default=0.2)
    ap.add_argument("--clip-vloss", action="store_true", default=True)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--norm-adv", action="store_true", default=True)
    ap.add_argument("--init-logstd", type=float, default=-0.5,
                     help="initial log-std of the action distribution (U23: -0.5 gives ~72%% per-step "
                          "chance ANY of 12 joints' sampled action clips at init; -1.0 gives ~8%%). "
                          "Ignored if --resume-from is given.")
    ap.add_argument("--resume-from", default=None,
                     help="path to a checkpoint .pt to resume from -- loads model weights (including "
                          "the LEARNED actor_logstd, not init_logstd), the observation normalizer, and "
                          "(if present in the checkpoint) optimizer state for an exact resumption; "
                          "older checkpoints without saved optimizer state produce a disclosed warm start.")
    ap.add_argument("--w-action-rate", type=float, default=0.01,
                     help="reward weight on -mean((action-prev_action)^2); the ONLY reward term this "
                          "script exposes for direct A/B comparison (U26)")
    ap.add_argument("--zero-mean-init", action="store_true",
                     help="U27: zero the final actor_mean layer's weight AND bias exactly, so the "
                          "deterministic policy starts EXACTLY at zero action (not just small) -- "
                          "hidden layers keep ordinary orthogonal init")
    ap.add_argument("--eval-every", type=int, default=0,
                     help="U27: every N iterations, run a quick deterministic-mean eval from a nominal "
                          "reset and save a *_best.pt checkpoint if it's the best survival seen so far "
                          "(0 disables)")
    ap.add_argument("--episode-seconds", type=float, default=4.0)
    ap.add_argument("--desired-vx", type=float, default=0.10,
                     help="fixed target speed when --curriculum is not set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)

    ap.add_argument("--curriculum", action="store_true",
                     help="start desired_vx near 0 and expand gradually instead of fixing it")
    ap.add_argument("--vx-stages", type=str, default="0.0,0.03,0.06,0.10",
                     help="comma-separated desired_vx stages, easiest first")
    ap.add_argument("--advance-window", type=int, default=20,
                     help="number of most-recent completed episodes used to judge stage readiness")
    ap.add_argument("--advance-len-frac", type=float, default=0.8,
                     help="advance once mean episode length >= this fraction of the episode horizon")
    ap.add_argument("--advance-max-fall-rate", type=float, default=0.3,
                     help="advance only if the recent fall rate is at or below this")
    ap.add_argument("--stage-min-steps", type=int, default=40_000,
                     help="minimum env-steps spent in a stage before advancement is even checked")

    ap.add_argument("--perturb-curriculum", action="store_true",
                     help="U29: broaden cara_env's reset_qvel_noise_bands mixture gradually instead of "
                          "using a single fixed reset_qvel_noise_std")
    ap.add_argument("--perturb-bands", type=str, default="0.4:0.0,0.3:3.0,0.3:7.0",
                     help="comma-separated prob:std pairs, easiest first; the LAST band is the one that "
                          "advances (probabilities stay fixed)")
    ap.add_argument("--perturb-advance-target", type=float, default=0.9,
                     help="advance the hard band's std once deterministic survival AT that std, over "
                          "PERTURB_VALIDATION_RESET_SEEDS, reaches this rate")
    ap.add_argument("--perturb-advance-step", type=float, default=1.0,
                     help="rad/s added to the hard band's std on each advance")
    ap.add_argument("--perturb-min-steps", type=int, default=50_000,
                     help="minimum env-steps at the current hard-band std before advancement is checked")
    ap.add_argument("--perturb-check-every", type=int, default=10,
                     help="iterations between perturbation-curriculum advancement checks")

    ap.add_argument("--desired-vx-bands", type=str, default=None,
                     help="U30: comma-separated prob:vx pairs (e.g. '0.5:0.0,0.5:0.03') -- one drawn per "
                          "EPISODE at reset() and held constant for that episode. None (default) uses "
                          "the single fixed --desired-vx value.")
    ap.add_argument("--vx-obs-scale", type=float, default=None,
                     help="U30: fixed (non-adaptive) scale for the desired_vx observation dimension, "
                          "in m/s -- required whenever introducing a nonzero command into a policy "
                          "whose obs_norm only ever saw desired_vx=0.0 (see RunningNorm docstring). "
                          "None disables (uses the ordinary adaptive normalization for that dimension too).")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
