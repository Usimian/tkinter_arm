import numpy as np
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, convert_degrees_to_steps
import threading
import tkinter as tk
from tkinter import ttk
import logging
from servo_utils import set_servo_position, read_servo_position, set_servo_zero_position, debug_servo_position, set_servo_offset

# Set up logger
logger = logging.getLogger("robot_arm")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Link lengths in mm
a1 = 80   # base to shoulder
a2 = 115  # shoulder to arm
a3 = 135  # arm to wrist
a4 = 63   # wrist to claw (end effector)
a5 = 110  # claw to gripper tip

# HARD LIMITS for each joint (in radians)
# Adjust these as needed for your robot's safe range
JOINT_LIMITS = [
    (-0.5, 0.5),   # base_joint
    ( 0.0, 1.5),   # shoulder_joint
    ( 0.0, 1.5),   # upper_arm_joint
    (-0.9, 0.8),   # forearm_joint
    (-1.5, 1.5),   # wrist_joint
]

CLAW_LIMITS = [
    (-0.1, 0.5),   # base_joint (e.g., -149 to +149 deg)
]

# Helper: Rotation matrix from roll, pitch, yaw (ZYX convention)
def rpy_to_matrix(roll, pitch, yaw):
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    return Rz @ Ry @ Rx

# Inverse kinematics for 5-DOF arm (elbow-down solution)
def inverse_kinematics(x, y, z, roll, pitch, yaw):
    """
    x, y, z: desired end effector position (mm)
    roll, pitch, yaw: desired end effector orientation (radians, ZYX order)
    Returns: [theta1, theta2, theta3, theta4, theta5] (radians)
    Order: base_joint, shoulder_joint, upper_arm_joint, forearm_joint, wrist_joint
    """
    # 1. Wrist center position
    R05 = rpy_to_matrix(roll, pitch, yaw)
    ax, ay, az = R05[:, 2]
    wx = x - a4 * ax
    wy = y - a4 * ay
    wz = z - a4 * az

    # 2. theta1 (base rotation)
    theta1 = np.arctan2(wy, wx)

    # 3. Planar distance from base to wrist center
    r = np.sqrt(wx**2 + wy**2) - a1
    s = wz

    # 4. theta2 and theta3 (shoulder and elbow)
    D = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    D = np.clip(D, -1.0, 1.0)  # Clamp for numerical safety
    theta3 = np.arctan2(-np.sqrt(1 - D**2), D)  # elbow-down
    phi1 = np.arctan2(s, r)
    phi2 = np.arctan2(a3 * np.sin(theta3), a2 + a3 * np.cos(theta3))
    theta2 = phi1 - phi2

    # 5. Compute wrist orientation (theta4, theta5)
    # Forward kinematics for first 3 joints
    def rot_z(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
    def rot_y(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    T01 = rot_z(theta1)
    T12 = rot_y(theta2)
    T23 = rot_y(theta3)
    R03 = T01 @ T12 @ T23
    R35 = R03.T @ R05
    # theta4, theta5 from R35 (ZY Euler)
    theta5 = np.arccos(R35[2,2])
    if np.sin(theta5) < 1e-6:
        theta4 = 0
    else:
        theta4 = np.arctan2(R35[1,2], R35[0,2])
    return [theta1, theta2, theta3, theta4, theta5]

def forward_kinematics(theta1, theta2, theta3, theta4, theta5):
    """
    Compute the end effector pose (x, y, z, roll, pitch, yaw) given 5 joint angles.
    Angles are in radians, order: base_joint, shoulder_joint, upper_arm_joint, forearm_joint, wrist_joint
    Returns: (x, y, z, roll, pitch, yaw)
    """
    # Helper rotation matrices
    def rot_z(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
    def rot_y(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    # Base to shoulder
    T01 = np.eye(4)
    T01[:3, :3] = rot_z(theta1)
    T01[:3, 3] = [a1, 0, 0]
    # Shoulder to upper arm
    T12 = np.eye(4)
    T12[:3, :3] = rot_y(theta2)
    T12[:3, 3] = [a2, 0, 0]
    # Upper arm to forearm
    T23 = np.eye(4)
    T23[:3, :3] = rot_y(theta3)
    T23[:3, 3] = [a3, 0, 0]
    # Forearm to wrist
    T34 = np.eye(4)
    T34[:3, :3] = rot_y(theta4)
    # Wrist to end effector
    T45 = np.eye(4)
    T45[:3, :3] = rot_y(theta5)
    T45[:3, 3] = [a4, 0, 0]
    # Chain the transforms
    T = T01 @ T12 @ T23 @ T34 @ T45
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    # Extract roll, pitch, yaw from rotation matrix (ZYX order)
    R = T[:3, :3]
    yaw = np.arctan2(R[1,0], R[0,0])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    roll = np.arctan2(R[2,1], R[2,2])
    return x, y, z, roll, pitch, yaw

def set_joint_angles(joint_angles, port='/dev/ttyACM0'):
    from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
    from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, convert_degrees_to_steps

    # Clamp joint angles to hard limits
    clamped_angles = [
        max(JOINT_LIMITS[i][0], min(JOINT_LIMITS[i][1], joint_angles[i]))
        for i in range(5)
    ]

    # Convert radians to degrees
    clamped_degrees = [np.degrees(a) for a in clamped_angles]
    models = ["sts3215"] * 5
    servo_steps = convert_degrees_to_steps(np.array(clamped_degrees), models)

    servo_ids = [1, 2, 3, 4, 5]  # Update as needed
    motors = {f"motor{i+1}": (servo_ids[i], "sts3215") for i in range(5)}
    config = FeetechMotorsBusConfig(port=port, motors=motors)
    bus = FeetechMotorsBus(config=config)
    bus.connect()
    # Use group write for all motors
    bus.write("Goal_Position", servo_steps, motor_names=list(motors.keys()))
    bus.disconnect()
    logger.info(f"Sent joint angles to servos (clamped): {servo_steps}")

def launch_servo_monitor_gui():
    class ServoMonitorApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Dual Arm Servo Position Monitor")
            self.geometry("800x300")
            self.resizable(False, False)
            
            # Create frames for each arm
            left_frame = ttk.LabelFrame(self, text="Arm 1 (/dev/ttyACM0)")
            left_frame.pack(side="left", padx=10, pady=5, fill="both", expand=True)
            
            right_frame = ttk.LabelFrame(self, text="Arm 2 (/dev/ttyACM1)")
            right_frame.pack(side="left", padx=10, pady=5, fill="both", expand=True)
            
            # Initialize lists for both arms
            self.labels = [[], []]  # [arm1_labels, arm2_labels]
            self.check_vars = [[], []]  # [arm1_vars, arm2_vars]
            self.checkboxes = [[], []]  # [arm1_checkboxes, arm2_checkboxes]
            self.servo_present = [[False] * 6, [False] * 6]  # Status for both arms
            
            # Create widgets for both arms
            for arm_idx, frame in enumerate([left_frame, right_frame]):
                for i in range(6):
                    row_frame = ttk.Frame(frame)
                    row_frame.pack(pady=5, anchor="w", padx=10)
                    var = tk.BooleanVar(value=True)
                    chk = ttk.Checkbutton(row_frame, variable=var)
                    chk.pack(side="left")
                    label = ttk.Label(row_frame, text=f"Servo {i+1}: ...", font=("Arial", 14))
                    label.pack(side="left", padx=10)
                    self.labels[arm_idx].append(label)
                    self.check_vars[arm_idx].append(var)
                    self.checkboxes[arm_idx].append(chk)
            
            self.scan_servos()
            self.update_servo_widgets()
            self.poll_servo_positions()

        def scan_servos(self):
            """Scan for servos 1-6 on both arms."""
            ports = ['/dev/ttyACM0', '/dev/ttyACM1']
            for arm_idx, port in enumerate(ports):
                for i in range(6):
                    pos = read_servo_position(i+1, port=port)
                    self.servo_present[arm_idx][i] = pos is not None

        def update_servo_widgets(self):
            """Update widget states for both arms."""
            for arm_idx in range(2):
                for i in range(6):
                    if not self.servo_present[arm_idx][i]:
                        self.labels[arm_idx][i].config(foreground="grey")
                        self.checkboxes[arm_idx][i].config(state="disabled")
                    else:
                        self.labels[arm_idx][i].config(foreground="black")
                        self.checkboxes[arm_idx][i].config(state="normal")

        def poll_servo_positions(self):
            """Poll positions from both arms."""
            ports = ['/dev/ttyACM0', '/dev/ttyACM1']
            for arm_idx, port in enumerate(ports):
                for i in range(6):
                    if not self.servo_present[arm_idx][i]:
                        self.labels[arm_idx][i].config(text=f"Servo {i+1}: (not present)")
                        continue
                    if self.check_vars[arm_idx][i].get():
                        try:
                            pos = read_servo_position(i+1, port=port)
                            if pos is not None:
                                # Map 0 -> -1.57, 2048 -> 0, 4095 -> 1.57
                                radians = (pos - 2048) * (1.57 / 2047)
                                self.labels[arm_idx][i].config(text=f"Servo {i+1}: {pos:5d} ({radians:7.3f} rad)")
                            else:
                                self.labels[arm_idx][i].config(text=f"Servo {i+1}: (no data)")
                        except Exception as e:
                            self.labels[arm_idx][i].config(text=f"Servo {i+1}: Error: {e}")
                    else:
                        self.labels[arm_idx][i].config(text=f"Servo {i+1}: (skipped)")
            self.after(200, self.poll_servo_positions)  # 0.2 seconds

    app = ServoMonitorApp()
    app.mainloop()

def read_joint_angles(port='/dev/ttyACM0', num_servos=5):
    """
    Reads the current position of all servos (default 5, or 6 if specified).
    Returns a list of positions in raw units (typically -2048 to +2048), or None for each servo if not found or error.
    """
    angles = []
    for i in range(1, num_servos+1):
        pos = read_servo_position(i, port=port)
        angles.append(pos)
    return angles

# Example usage:
if __name__ == "__main__":
    # Target pose (x, y, z, roll, pitch, yaw)
    x, y, z = 200, 0, -98
    roll, pitch, yaw = 0, 0, 0
    joint_angles = inverse_kinematics(x, y, z, roll, pitch, yaw)

    # Forward kinematics test
    x_fk, y_fk, z_fk, roll_fk, pitch_fk, yaw_fk = forward_kinematics(*joint_angles)

    # Send joint angles to hardware
    # set_joint_angles(joint_angles, port='/dev/ttyACM0')
    launch_servo_monitor_gui()
