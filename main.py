import numpy as np
import serial
import time

# Link lengths in mm
a1 = 80   # base to shoulder
a2 = 115  # shoulder to arm
a3 = 135  # arm to wrist
a4 = 63   # wrist to claw (end effector)

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
    """
    Set the joint angles on Feetech STS3215 servos via serial port.
    joint_angles: list of 5 angles in radians (base, shoulder, upper_arm, forearm, wrist)
    port: serial port (default: /dev/ttyACM0)
    This is a template; you may need to fine-tune the protocol and mapping for your hardware.
    """
    # Example: Map radians to servo units (0-1023 for 0-300 degrees)
    def rad_to_servo(angle_rad):
        angle_deg = angle_rad * 180.0 / np.pi
        # Clamp to servo range (0-300 degrees)
        angle_deg = max(0, min(300, angle_deg + 150))  # shift -150~+150 to 0~300
        servo_val = int(angle_deg / 300.0 * 1023)
        return servo_val

    servo_ids = [1, 2, 3, 4, 5]  # Update with your servo IDs
    servo_vals = [rad_to_servo(a) for a in joint_angles]

    # Open serial port
    with serial.Serial(port, baudrate=115200, timeout=1) as ser:
        for sid, sval in zip(servo_ids, servo_vals):
            # Example command: [0xFF, 0xFF, ID, ...data...]
            # You must replace this with the correct protocol for STS3215
            # This is a placeholder for Feetech serial protocol
            cmd = bytearray([0xFF, 0xFF, sid, (sval & 0xFF), ((sval >> 8) & 0xFF)])
            ser.write(cmd)
            time.sleep(0.05)  # Small delay between commands
    print("Sent joint angles to servos:", servo_vals)

# Example usage:
if __name__ == "__main__":
    # Target pose (x, y, z, roll, pitch, yaw)
    x, y, z = 258, 0, -98
    roll, pitch, yaw = 0, 0, 0
    joint_angles = inverse_kinematics(x, y, z, roll, pitch, yaw)

    # Forward kinematics test
    x_fk, y_fk, z_fk, roll_fk, pitch_fk, yaw_fk = forward_kinematics(*joint_angles)
    print("FK pose:", x_fk, y_fk, z_fk, roll_fk, pitch_fk, yaw_fk)

    # Send joint angles to hardware
    set_joint_angles(joint_angles, port='/dev/ttyACM0')
