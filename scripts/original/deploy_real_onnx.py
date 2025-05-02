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
    G1MjxJointIndex,
    RESTRICTED_JOINT_RANGE,
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


import numpy as np
load_dotenv()
NETWORK_CARD_NAME = 'enxc8a362b43bfd'
# mask_arms = True
# num_obs = 72

# from brax.training.agents.ppo import checkpoint as ppo_checkpoint
# import jax
# import jax.random  # Add global import for jax.random
# from pathlib import Path
import onnxruntime as rt

from keyboard_reader import KeyboardController

import time


class OnnxPolicy:
  """ONNX controller for the Go-1 robot."""

  def __init__(
      self,
      policy_path: str,

  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CUDAExecutionProvider"]
    )

  def get_control(self, obs: np.ndarray) -> None:

    onnx_input = {"obs": obs.reshape(1, -1)}
    onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]

    # Zero out arm control similar to training logic
    ZERO_ARM_CONTROL = True  # Set this flag as needed
    if ZERO_ARM_CONTROL:
        # Assuming arm indices are the last 10 (indices 13 to 22)
        # Order: L leg (6), R leg (6), Waist (1), L arm (5), R arm (5) -> Total 23
        arm_indices = slice(13, 23)  # Indices 13, 14, ..., 22
    onnx_pred[arm_indices] = 0.0
    return onnx_pred



 

class Controller:


    def __init__(self, policy: OnnxPolicy) -> None:
        self.policy = policy

        # Initialize the policy network
        # self.policy = torch.jit.load(config.policy_path)
        # # Initializing process variables
        self.qj = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.dqj = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.action = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        # self.target_dof_pos = config.default_angles.copy()
        # self.obs = np.zeros(num_obs, dtype=np.float32)
        # self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        # Convert joint range tuples to numpy arrays for efficient clamping
        joint_limits = np.array(RESTRICTED_JOINT_RANGE, dtype=np.float32)
        self._joint_lower_bounds = joint_limits[:, 0]
        self._joint_upper_bounds = joint_limits[:, 1]

        self._controller = KeyboardController(
            vel_scale_x=1.0,
            vel_scale_y=1.0,
            vel_scale_rot=1.0,
        )
        # print("Hello")

        self.control_dt = 0.02
        self._phase = np.array([0.0, np.pi])
        self._gait_freq = 1.5
        self._phase_dt = 2 * np.pi * self._gait_freq * self.control_dt

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        self.default_pos_array = np.array(default_pos)

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    # def setup_policy(self):
    #     relative_path = Path("./checkpoints/g1_balance-happy-mountain-1/000408944640")
    #     policy_fn = ppo_checkpoint.load_policy(relative_path.resolve())
    #     self.policy = jax.jit(policy_fn)

    
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
            self.qj[i] = self.low_state.motor_state[joint2motor_idx[i]].q - default_pos[i]
            self.dqj[i] = self.low_state.motor_state[joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        gyro = self.low_state.imu_state.gyroscope
        # print(gyro)
        # ang_vel = np.array(
        #     self.low_state.imu_state.gyroscope, dtype=np.float32)

        # create observation
        gravity = get_gravity_orientation(quat)
        joint_angles = self.qj.copy()
        joint_velocities = self.dqj.copy()
        phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
        command = self._controller.get_command() # Original line
        # command = np.array([0.0, 0.0, 0.0], dtype=np.float32) # Debug: Set command to zeros
        obs = np.hstack([
            gyro,
            gravity,
            command,
            joint_angles,
            joint_velocities,
            self.action,
            phase,
        ]).astype(np.float32)
        # return obs.astype(np.float32)

        self.action = self.policy.get_control(obs)
        # print("Action: ", self.action)
        action_effect = self.action * action_scale
        # Create a mask to zero out action for the last 10 joints (arms) if config is set.
        # Assuming mjx_model.nu is 23.
        arm_mask = np.ones_like(action_effect)
        arm_mask[-10:] = 0.0
        masked_action_effect = np.where(
            mask_arms, action_effect * arm_mask, action_effect
        )
        # print(masked_action_effect)
        # print(f"default_pos_array type: {type(self.default_pos_array)}, shape: {self.default_pos_array.shape}")
        # print(f"masked_action_effect type: {type(masked_action_effect)}, shape: {masked_action_effect.shape}")
        motor_targets_unclamped = self.default_pos_array  + masked_action_effect

        # Clamp motor targets to joint limits and check for clamping
        motor_targets = np.clip(
            motor_targets_unclamped, self._joint_lower_bounds, self._joint_upper_bounds
        )
        clamped_indices = np.where(motor_targets != motor_targets_unclamped)[0]
        if clamped_indices.size > 0:
            print("WARNING: Clamping motor targets for joints:")
            for idx in clamped_indices:
                print(f"  Joint {idx}: {motor_targets_unclamped[idx]:.3f} -> {motor_targets[idx]:.3f} (limits: [{self._joint_lower_bounds[idx]:.3f}, {self._joint_upper_bounds[idx]:.3f}])")

        # print("Ankle motor targets:")
        # print(f"Left ankle pitch: {motor_targets[G1MjxJointIndex.LeftAnklePitch]:.3f}")

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

    #   self._last_action = onnx_pred.copy()
        # data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
        phase_tp1 = self._phase + self._phase_dt
        self._phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi

        time.sleep(self.control_dt)


if __name__ == "__main__":
    print("Setting up policy...")
    policy = OnnxPolicy("./bh_policy.onnx")
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    # Initial prompt doesn't need non-blocking
    input("Press Enter to acknowledge warning and proceed...")

    ChannelFactoryInitialize(0, NETWORK_CARD_NAME)

    controller = Controller(policy)
    # Initial prompt doesn't need non-blocking
    # input("Press Enter to acknowledge warning and proceed...")

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
