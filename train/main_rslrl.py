import numpy as np
import os
from ml_collections import config_dict
import mujoco
import jax
import mediapy as media
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import wandb
import torch
from mujoco_playground import wrapper_torch
from rsl_rl.runners import OnPolicyRunner
from utils import get_rslrl_config
from randomize import domain_randomize
import locomotion_env
from datetime import datetime


from ml_collections import config_dict
import json

def train_run(config=None, use_wandb=False):
    device = "cuda:0"
    device_rank = int(device.split(":")[-1]) 
    run = None
    if use_wandb and wandb:
        run = wandb.init(project="23dof_oldjax_standalone", config=config)


    env_cfg = locomotion_env.default_config()
    env_name = "g1_locomotion"

      # Generate unique experiment name
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{env_name}-{timestamp}"
    print(f"Experiment name: {exp_name}")

    # Logging directory
    logdir = os.path.abspath(os.path.join("logs", exp_name))
    os.makedirs(logdir, exist_ok=True)
    print(f"Logs are being stored in: {logdir}")

    # Checkpoint directory
    ckpt_path = os.path.join(logdir, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    # Initialize Weights & Biases if required
    if use_wandb:
        wandb.tensorboard.patch(root_logdir=logdir)
        wandb.init(project="mjxrl", entity="dextrm", name=exp_name)
        wandb.config.update(env_cfg.to_dict())
        wandb.config.update({"env_name": env_name})

    # Save environment config to JSON
    with open(
        os.path.join(ckpt_path, "config.json"), "w", encoding="utf-8"
    ) as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

    # Domain randomization
    randomizer = domain_randomize

    # We'll store environment states during rendering
    render_trajectory = []

    # Callback to gather states for rendering
    def render_callback(_, state):
        render_trajectory.append(state)
    
    SEED = 42
    NUM_ENVS = 4096

    # Create the environment
    raw_env = locomotion_env.G1Locomotion(config=env_cfg)
    brax_env = wrapper_torch.RSLRLBraxWrapper(
        raw_env,
        NUM_ENVS,
        SEED,
        env_cfg.episode_length,
        1,
        render_callback=render_callback,
        randomization_fn=randomizer,
        device_rank=device_rank,
    )

    # Build RSL-RL config
    train_cfg = get_rslrl_config()

    # Overwrite default config with flags
    train_cfg.seed = SEED
    train_cfg.run_name = exp_name
    train_cfg.resume = False
    train_cfg.load_run =  "-1"
    train_cfg.checkpoint = 0

    train_cfg_dict = train_cfg.to_dict()
    runner = OnPolicyRunner(brax_env, train_cfg_dict, logdir, device=device)
    print("Training...")
    runner.learn(
        num_learning_iterations=train_cfg.max_iterations,
        init_at_random_ep_len=False,
    )
    print("Done training.")

    policy = runner.get_inference_policy(device=device)

    # Example: run a single rollout
    eval_env =  locomotion_env.G1Locomotion(config=env_cfg)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(SEED)
    state = jit_reset(rng)
    rollout = [state]

    # We’ll assume your environment’s observation is in state.obs["state"].
    obs_torch = wrapper_torch._jax_to_torch(state.obs["state"])

    for _ in range(env_cfg.episode_length):
        with torch.no_grad():
            actions = policy(obs_torch)
        # Step environment
        state = jit_step(state, wrapper_torch._torch_to_jax(actions.flatten()))
        rollout.append(state)
        obs_torch = wrapper_torch._jax_to_torch(state.obs["state"])
        if state.done:
         break

    # Render
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    render_every = 2
    # If your environment is wrapped multiple times, adjust as needed:
    base_env = eval_env  # or brax_env.env.env.env
    fps = 1.0 / base_env.dt / render_every
    traj = rollout[::render_every]
    frames = eval_env.render(
        traj,
        camera="track",
        height=480,
        width=640,
        scene_option=scene_option,
    )
    media.write_video("rollout.mp4", frames, fps=fps)
    if use_wandb and run:
        wandb.log({"video": wandb.Video(frames, fps=fps, format="gif")})


if __name__ == "__main__":
    train_run(use_wandb=True)
