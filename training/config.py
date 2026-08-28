from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
#python isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#    --task Isaac-Velocity-Rough-Cara-v0 \
#   --num_envs 4096 \
#    --headless
@configclass
class CaraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24       # Rollout length
    max_iterations = 1500        # Total training epochs
    save_interval = 50           # Save a "checkpoint" every 50 epochs
    experiment_name = "cara_locomotion"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std = 1.0,
        actor_hidden_dims = [512, 256, 128], # The "Brain" size
        critic_hidden_dims = [512, 256, 128],
        activation = "elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef = 1.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.01,     # Encourages Cara to try new moves (exploration)
        learning_rate = 1e-3,    # How fast she learns from mistakes
        num_learning_epochs = 5,
        num_mini_batches = 4,    # Split 4000 envs into 4 batches for the 5060
    )
