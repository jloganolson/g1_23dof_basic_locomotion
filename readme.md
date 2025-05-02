This is my first working pass of a standalone repo to train and deploy a basic locomotion policy for the 23dof Unitree G1. 

It is adapted from the G1 29dof code distributed throughout the [Mujoco Playground repo](https://github.com/google-deepmind/mujoco_playground) along with the deployment code from the [Unitree RL repo](https://github.com/unitreerobotics/unitree_rl_gym). 

Note: The policy is quite jank (i.e. the robot stumbles drunkenly around), but it's a full train-test-deploy loop and the robot actually balances and moves around IRL!


![policy output](output.gif)


### Instructions
IMPORTANT: For this to work, be sure to use the old 4.3.8 version of jax as mentioned in this Mujoco Playground issue: (https://github.com/google-deepmind/mujoco_playground/issues/112)

Setup/pip install the following repos 
(https://github.com/unitreerobotics/unitree_sdk2_python)
(https://github.com/google-deepmind/mujoco_playground)
(https://github.com/google/brax)

To train, run `python train/main.py` 
To convert pkl to onnx, update the ckpt_path and run all the cells in `train/brax_network_to_onnx.ipynb`
To test using sim2sim with mujoco, run `python deploy/deploy_sim.py` with updated ONNX_PATH string
To deploy irl, setup your G1 and run `python deploy/deploy_real.py` with updated ONNX_PATH and NETWORK_CARD_NAME strings



