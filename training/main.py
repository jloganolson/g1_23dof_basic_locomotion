import numpy as np
import os
from ml_collections import config_dict
import pickle
from etils import epath
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from functools import partial
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import wandb
import jax
# import mediapy as media # media import removed as it wasn't used after refactor
from randomize import domain_randomize
import locomotion_env # Import balance to access default_config
from datetime import datetime

from flax.training import orbax_utils
from orbax import checkpoint as ocp
from mujoco_playground import wrapper
from ml_collections import config_dict
import json

# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

def train_run(config=None, use_wandb=False):
    """Trains the G1 balance task with PPO, optionally using wandb."""
    run = None
    if use_wandb and wandb:
        run = wandb.init(project="23dof_oldjax_standalone", config=config)
        cfg = wandb.config # Use wandb config if available
    else:
        # Use provided config dict or create an empty one if none provided
        cfg = config if config is not None else config_dict.ConfigDict()

    env_cfg = locomotion_env.default_config()
    env_name = "g1_locomotion"

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{env_name}-{timestamp}"

    ckpt_path = epath.Path("checkpoints").resolve() / exp_name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"{ckpt_path}")

    with open(ckpt_path / "config.json", "w") as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

    env = locomotion_env.G1Locomotion()
    eval_env = locomotion_env.G1Locomotion()

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    run_name = (run.name if run and hasattr(run, 'name') else timestamp) if use_wandb and wandb else timestamp
    exp_name = f"{env_name}-{run_name}"

    print(f"Checkpoint path: {ckpt_path}")
    ppo_params = config_dict.create(
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

    # Log the actual PPO params being used if wandb is active
    if use_wandb and run:
        # Convert config_dict to a standard dict for wandb logging
        wandb.config.update(ppo_params.to_dict(), allow_val_change=True)

    # --- Progress Callback ---
    def progress_cli(num_steps, metrics):
        if use_wandb and run:
            wandb.log(metrics, step=num_steps)
        print(".", end="", flush=True)

    # --- Training Setup ---
    ppo_training_params = dict(ppo_params)
    network_factory_config = ppo_training_params.pop("network_factory")
    network_factory = partial(
        ppo_networks.make_ppo_networks,
        **network_factory_config
    )

    def policy_params_fn(current_step, make_policy, params):
        del make_policy  # Unused.
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f"{current_step}"
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)

    training_params = dict(ppo_params)
    del training_params["network_factory"]

    train_fn = partial(
        ppo.train, **ppo_training_params,
        network_factory=network_factory,
        progress_fn=progress_cli,
        randomization_fn=domain_randomize,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        policy_params_fn=policy_params_fn,
        # save_checkpoint_path=ckpt_path
    )

    # --- Run Training ---
    print("Starting training...")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env
    )
    print("\nTraining finished.")

    normalizer_params, policy_params, value_params = params
    with open(ckpt_path / "params.pkl", "wb") as f:
        data = {
            "normalizer_params": normalizer_params,
            "policy_params": policy_params,
            "value_params": value_params,
        }
        pickle.dump(data, f)
    print(f"Saved params.pkl to {ckpt_path / 'params.pkl'}")

    # --- Save Final Checkpoint as Wandb Artifact ---
    if use_wandb and run:
        print(f"Saving params.pkl to wandb...")
        artifact = wandb.Artifact(f'{exp_name}-params', type='model')
        artifact.add_file(ckpt_path / "params.pkl")
        run.log_artifact(artifact)
        print("Params saved to wandb.")

    # --- Evaluation & Logging ---
    print("Starting evaluation...")
    # Use a fresh eval env instance with the same overrides
    eval_env_2 = locomotion_env.G1Locomotion()

    jit_reset = jax.jit(eval_env_2.reset)
    jit_step = jax.jit(eval_env_2.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(cfg.get('eval_seed', 42))
    rollout = []
    n_episodes = 1

    for _ in range(n_episodes):
        state = jit_reset(rng)
        rollout.append(state)
        for i in range(env_cfg.episode_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state)


    frames = eval_env_2.render(rollout, camera="track")
    frames_np = np.array(frames)
    frames_np_rearranged = np.transpose(frames_np, (0, 3, 1, 2))
    if use_wandb and run:
        wandb.log({"video": wandb.Video(frames_np_rearranged, fps=1.0 / env.dt, format="gif")})
    else:
        # Consider saving the video locally if wandb is not used
        print("Video generated. Consider saving it locally.") # Placeholder for local saving

    print("Evaluation finished.")
    
    if use_wandb and run:
        print(f"Wandb run URL: {run.url}")
        run.finish()

if __name__ == "__main__":
    # Example: Train without wandb by default
    # To enable wandb, you could parse command-line args here
    # For now, it defaults to False
    train_run(use_wandb=True)
