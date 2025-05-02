from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ 
import numpy as np

G1_NUM_MOTOR = 23

Kp = [
    60, 60, 60, 100, 40, 40,      # legs
    60, 60, 60, 100, 40, 40,      # legs
    60,                   # waist
    40, 40, 40, 40,  40,   # arms
    40, 40, 40, 40,  40,   # arms
]

Kd = [ 
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1, 2, 1, 1,     # legs
    1,              # waist
    1, 1, 1, 1, 1,  # arms
    1, 1, 1, 1, 1,    # arms 
]

default_pos = [
    -0.1, 0, 0, 0.3, -0.2, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    0, 
    0.2, 0.2, 0, 1.28, 0, 
    0.2, -0.2, 0, 1.28, 0, 
]

dof_pos_scale = 1.0
dof_vel_scale = 1.0
action_scale = 0.5
ang_vel_scale = 1.0
mask_arms = True



class G1MjxJointIndex:
    """Joint indices based on the order in g1_mjx_alt.xml (23 DoF model)."""
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    WaistYaw = 12
    LeftShoulderPitch = 13
    LeftShoulderRoll = 14
    LeftShoulderYaw = 15
    LeftElbow = 16
    LeftWristRoll = 17
    RightShoulderPitch = 18
    RightShoulderRoll = 19
    RightShoulderYaw = 20
    RightElbow = 21
    RightWristRoll = 22

    # Note: This model has 23 degrees of freedom (indices 0-22).
    # It lacks WaistRoll, WaistPitch, LeftWristPitch, LeftWristYaw,
    # RightWristPitch, and RightWristYaw compared to the original G1JointIndex.


# Mapping from G1MjxJointIndex (0-22) to G1JointIndex (0-28)
joint2motor_idx = [
    0,  # LeftHipPitch
    1,  # LeftHipRoll
    2,  # LeftHipYaw
    3,  # LeftKnee
    4,  # LeftAnklePitch
    5,  # LeftAnkleRoll
    6,  # RightHipPitch
    7,  # RightHipRoll
    8,  # RightHipYaw
    9,  # RightKnee
    10, # RightAnklePitch
    11, # RightAnkleRoll
    12, # WaistYaw
    15, # LeftShoulderPitch (skips WaistRoll=13, WaistPitch=14)
    16, # LeftShoulderRoll
    17, # LeftShoulderYaw
    18, # LeftElbow
    19, # LeftWristRoll (skips LeftWristPitch=20, LeftWristYaw=21)
    22, # RightShoulderPitch
    23, # RightShoulderRoll
    24, # RightShoulderYaw
    25, # RightElbow
    26, # RightWristRoll (skips RightWristPitch=27, RightWristYaw=28)
]

class MotorMode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


def init_cmd_hg(cmd: LowCmd_, mode_machine: int, mode_pr: int):
    cmd.mode_machine = mode_machine
    cmd.mode_pr = mode_pr
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

def create_damping_cmd(cmd:  LowCmd_):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 8
        cmd.motor_cmd[i].tau = 0


def create_zero_cmd(cmd:LowCmd_):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

import select
import tty
import termios
import sys


# --- Non-blocking Keyboard Input Context Manager ---
class NonBlockingInput:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        try:
            tty.setraw(sys.stdin.fileno())
        except termios.error as e:
            # Fallback if not a tty (e.g., running in certain IDEs/environments)
            print(f"Warning: Could not set raw mode: {e}. Key detection might not work.", file=sys.stderr)
            self.old_settings = None # Indicate failure
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_settings:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        print("\nRestored terminal settings.") # Optional: provide feedback

    def check_key(self, key='\n'):
        """Check if a specific key is pressed without blocking."""
        if not self.old_settings: # If raw mode failed, don't check
            return False
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            ch = sys.stdin.read(1)
            # In raw mode, Enter is often '\r' (carriage return)
            return ch == (key if key != '\n' else '\r')
        return False
# -----------------------------------------------------

RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Right leg. 6
    (-2.5307, 2.8798),
    (-2.9671, 0.5236),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Waist.
    (-2.618, 2.618),
    # Left shoulder.
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    # Right shoulder.
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
)