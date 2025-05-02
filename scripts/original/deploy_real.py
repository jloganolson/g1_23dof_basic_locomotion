import jax
from utils_real import (
    init_cmd_hg,
    create_damping_cmd,
    create_zero_cmd,
    MotorMode,
    NonBlockingInput,
    joint2motor_idx,
    Kp,
    Kd,
    G1_NUM_MOTOR,
    default_pos,
    get_gravity_orientation,
    dof_pos_scale,
    dof_vel_scale,
    action_scale,
    ang_vel_scale,
    mask_arms,
)
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
import time
from dotenv import load_dotenv
import os
import jax.numpy as jp
import numpy as np
load_dotenv()
NETWORK_CARD_NAME = os.getenv('NETWORK_CARD_NAME')
mask_arms = True
num_obs = 72

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
import jax.random  # Add global import for jax.random
from pathlib import Path


import time

CHECKPOINT_PATH = "./checkpoints/g1_balance-happy-mountain-1/000408944640"


class Policy:
    def __init__(self):
        relative_path = Path(CHECKPOINT_PATH)
        policy_fn = ppo_checkpoint.load_policy(relative_path.resolve())
        # Just JIT the function here
        self.policy = jax.jit(policy_fn)
        test_input = {
        'state': jp.ones(num_obs),
        'privileged_state': jp.zeros(153) # state (72) + gyro (3) + acc (3) + grav (3) + joint_angles (23) + joint_vel (23) + root_height (1) + actuator_force (23) + contact (2) = 153
    }
        jax_pred, _ = self.policy(test_input, jax.random.PRNGKey(0))
        print(jax_pred)

    def call_policy(self, obs: jax.Array, privileged_obs: jax.Array):
        input = {
            'state': obs,
            'privileged_state': privileged_obs
        }
        return self.policy(input, jax.random.PRNGKey(0))
 

class Controller:


    def __init__(self, policy: Policy) -> None:
        self.policy = policy

        # Initialize the policy network
        # self.policy = torch.jit.load(config.policy_path)
        # # Initializing process variables
        self.qj = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.dqj = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.action = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        # self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(num_obs, dtype=np.float32)
        # self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        self.control_dt = 0.02
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        self.default_pos_array = jp.array(default_pos)

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def setup_policy(self):
        relative_path = Path("./checkpoints/g1_balance-happy-mountain-1/000408944640")
        policy_fn = ppo_checkpoint.load_policy(relative_path.resolve())
        self.policy = jax.jit(policy_fn)

    
    def send_cmd(self, cmd:  LowCmd_):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Press Enter to continue...")
        with NonBlockingInput() as nbi:
            while not nbi.check_key('\n'):
                create_zero_cmd(self.low_cmd)
                self.send_cmd(self.low_cmd)
        print("Zero torque state confirmed. Proceeding...")

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.control_dt)

        init_dof_pos = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        for i in range(G1_NUM_MOTOR):
            init_dof_pos[i] = self.low_state.motor_state[joint2motor_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(G1_NUM_MOTOR):
                motor_idx = joint2motor_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * \
                    (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = Kp[j]
                self.low_cmd.motor_cmd[motor_idx].kd = Kd[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Press Enter to start the controller...")
        with NonBlockingInput() as nbi:
            while not nbi.check_key('\n'):  # Check for Enter key
                # Keep sending default position commands while waiting
                for i in range(len(joint2motor_idx)):
                    motor_idx = joint2motor_idx[i]
                    self.low_cmd.motor_cmd[motor_idx].q = default_pos[i]
                    self.low_cmd.motor_cmd[motor_idx].qd = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
                    self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0

                self.send_cmd(self.low_cmd)
                time.sleep(self.control_dt)
        print("Default position state confirmed. Starting controller...")

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(G1_NUM_MOTOR):
            self.qj[i] = self.low_state.motor_state[joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(
            [self.low_state.imu_state.gyroscope], dtype=np.float32)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - default_pos) * \
            dof_pos_scale
        dqj_obs = dqj_obs * dof_vel_scale
        ang_vel = ang_vel * ang_vel_scale
     
        # Construct the observation vector based on the training structure:
        # (gravity, joint_angles - default, joint_vel, last_action)
        # Note: Ensure self.obs is initialized with the correct size (3 + 23 + 23 + 23 = 72).
        # Note: The subsequent lines that assign values to self.obs should be removed
        #       as this block completely defines the observation vector according to training.
        current_obs_list = [
            gravity_orientation,  # 3 elements
            qj_obs,               # 23 elements (already relative to default and scaled)
            dqj_obs,              # 23 elements (already scaled)
            self.action           # 23 elements (action from the previous step)
        ]
        # Concatenate the list into a single numpy array for the observation
        self.obs = np.concatenate(current_obs_list).astype(np.float32)

        # Get the action from the policy network
        obs_tensor = jax.numpy.array(self.obs).reshape(1, -1)
        # Create a deterministic key for inference
        # inference_key = jax.random.PRNGKey(0)
        # input = {
        #     'state': obs_tensor,
        #     'privileged_state': jp.zeros(153)
        # }
        self.action= self.policy.call_policy(obs_tensor,  jp.zeros(153))[0].squeeze()
        print(self.action)
        action_effect = self.action * action_scale
        # Create a mask to zero out action for the last 10 joints (arms) if config is set.
        # Assuming mjx_model.nu is 23.
        arm_mask = jp.ones_like(action_effect).at[-10:].set(0.0)
        masked_action_effect = jp.where(
            mask_arms, action_effect * arm_mask, action_effect
        )
        # print(f"default_pos_array type: {type(self.default_pos_array)}, shape: {self.default_pos_array.shape}")
        # print(f"masked_action_effect type: {type(masked_action_effect)}, shape: {masked_action_effect.shape}")
        motor_targets = self.default_pos_array  + masked_action_effect

        # transform action to target_dof_pos
        # target_dof_pos = self.action

        # Build low cmd
        for i in range(G1_NUM_MOTOR):
            motor_idx = joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = motor_targets[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
            self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.control_dt)


if __name__ == "__main__":
    print("Setting up policy...")
    policy = Policy()
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    # Initial prompt doesn't need non-blocking
    input("Press Enter to acknowledge warning and proceed...")

    ChannelFactoryInitialize(0, NETWORK_CARD_NAME)

    controller = Controller(policy)

    # Enter the zero torque state, press Enter key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press Enter key to continue executing
    controller.default_pos_state()

    print("Controller running. Press 'q' to quit.")
    with NonBlockingInput() as nbi:  # Use context manager for the main loop
        while True:
            controller.run()
            # Check for 'q' key press to exit
            if nbi.check_key('q'):
                print("\n'q' pressed. Exiting loop...")
                break
            # Add a small sleep to prevent busy-waiting if controller.run() is very fast
            time.sleep(0.001)

    print("Entering damping state...")
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)

    print("Exit")
