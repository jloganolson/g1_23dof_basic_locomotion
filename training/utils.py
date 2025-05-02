from ml_collections import config_dict

def get_ppo_params(env_cfg):
   return config_dict.create(
        num_timesteps=200_000_000,
        num_evals=20,
        reward_scaling=1.0,
        episode_length=env_cfg.episode_length,
        normalize_observations=True,
        action_repeat=1,
        clipping_epsilon = 0.2,
        num_resets_per_eval = 1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost= 0.005,
        num_envs=8192,
        batch_size=256,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
    )