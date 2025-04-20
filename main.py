import numpy as np
import serial
import time
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, convert_degrees_to_steps

# Link lengths in mm
a1 = 80   # base to shoulder
a2 = 115  # shoulder to arm
a3 = 135  # arm to wrist
a4 = 63   # wrist to claw (end effector)

# HARD LIMITS for each joint (in radians)
# Adjust these as needed for your robot's safe range
JOINT_LIMITS = [
    (-2.6, 2.6),   # base_joint (e.g., -149 to +149 deg)
    (-1.5, 1.5),   # shoulder_joint (e.g., -86 to +86 deg)
    (-2.0, 2.0),   # upper_arm_joint (e.g., -114 to +114 deg)
    (-2.0, 2.0),   # forearm_joint (e.g., -114 to +114 deg)
    (-2.6, 2.6),   # wrist_joint (e.g., -149 to +149 deg)
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
    print("Sent joint angles to servos (clamped):", servo_steps)

def set_servo_position(servo_id, value, port='/dev/ttyACM0'):
    """
    Set the specified servo to the given value (servo units, -2047 to +2047).
    Example: set_servo(1, 1024) will set servo 1 to about +90 degrees.
    """
    from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
    from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

    # Clamp value to servo range
    value = int(max(-2047, min(2047, value)))

    motors = {f"motor{servo_id}": (servo_id, "sts3215")}
    config = FeetechMotorsBusConfig(port=port, motors=motors)
    bus = FeetechMotorsBus(config=config)
    bus.connect()
    bus.write("Goal_Position", value, motor_names=f"motor{servo_id}")
    bus.disconnect()
    print(f"Set servo {servo_id} to value: {value}")

def read_servo_position(servo_id=None, port='/dev/ttyACM0'):
    """
    Reads the current position of the specified servo.
    Returns the position in raw units (typically -2048 to +2048), or None if not found or error.
    """
    from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
    from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

    if servo_id is not None:
        motors = {f"motor{servo_id}": (servo_id, "sts3215")}
        config = FeetechMotorsBusConfig(port=port, motors=motors)
        bus = FeetechMotorsBus(config=config)
        try:
            bus.connect()
            position = bus.read("Present_Position", motor_names=f"motor{servo_id}")
            bus.disconnect()
            return position[0]
        except Exception as e:
            print(f"Error: Servo {servo_id} not found. Reason: {e}")
            try:
                bus.disconnect()
            except Exception:
                pass
            return None

def set_servo_zero_position(servo_id, port='/dev/ttyACM0'):
    """
    Sets the current position of the specified servo as the new zero (centered at 2048) in the servo's memory, without moving the servo.
    """
    from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
    from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

    motors = {f"motor{servo_id}": (servo_id, "sts3215")}
    config = FeetechMotorsBusConfig(port=port, motors=motors)
    bus = FeetechMotorsBus(config=config)
    try:
        bus.connect()
        # Unlock the servo to allow writing to Offset
        bus.write("Lock", 0, motor_names=f"motor{servo_id}")
        # Set offset to zero first
        set_servo_offset(servo_id, 0)
        time.sleep(0.1)
        # Read the current position
        current_pos = read_servo_position(servo_id)
        # Calculate the required offset so that current_pos becomes 2048
        offset = 2048 - int(current_pos)
        # Write the new offset
        set_servo_offset(servo_id, offset)
        bus.disconnect()
        print(f"Set current position {current_pos} of servo {servo_id} as new zero (centered at 2048, offset={offset})")
    except Exception as e:
        print(f"Error: Could not set servo {servo_id} zero offset. Reason: {e}")
        try:
            bus.disconnect()
        except Exception:
            pass

def debug_servo_position(servo_id, port='/dev/ttyACM0'):
    from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
    from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

    motors = {f"motor{servo_id}": (servo_id, "sts3215")}
    config = FeetechMotorsBusConfig(port=port, motors=motors)
    bus = FeetechMotorsBus(config=config)
    try:
        bus.connect()
        bus.write("Lock", 0, motor_names=f"motor{servo_id}")
        set_servo_offset(servo_id, 0)
        time.sleep(0.1)
        pos1 = read_servo_position(servo_id)
        time.sleep(0.1)
        set_servo_offset(servo_id, 500)
        time.sleep(0.1)
        pos2 = read_servo_position(servo_id)
        set_servo_offset(servo_id, 0)
        time.sleep(0.1)
        pos3 = read_servo_position(servo_id)
        time.sleep(0.1)
        pos4 = read_servo_position(servo_id)
        print(f"First read: {pos1}, Second read: {pos2}, Third read: {pos3}, Fourth read: {pos4}")
        bus.disconnect()
    except Exception as e:
        print(f"Error: {e}")
        try:
            bus.disconnect()
        except Exception:
            pass

def set_servo_offset(servo_id, offset, port='/dev/ttyACM0'):
    """
    Writes the specified offset value to the servo's memory.
    """
    from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
    from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

    motors = {f"motor{servo_id}": (servo_id, "sts3215")}
    config = FeetechMotorsBusConfig(port=port, motors=motors)
    bus = FeetechMotorsBus(config=config)
    try:
        bus.connect()
        bus.write("Lock", 0, motor_names=f"motor{servo_id}")
        bus.write("Offset", offset, motor_names=f"motor{servo_id}")
        bus.disconnect()
        print(f"Set offset {offset} for servo {servo_id}.")
    except Exception as e:
        print(f"Error: Could not set offset {offset} for servo {servo_id}. Reason: {e}")
        try:
            bus.disconnect()
        except Exception:
            pass

# Example usage:
if __name__ == "__main__":
    # Target pose (x, y, z, roll, pitch, yaw)
    x, y, z = 200, 0, -98
    roll, pitch, yaw = 0, 0, 0
    joint_angles = inverse_kinematics(x, y, z, roll, pitch, yaw)

    # Forward kinematics test
    x_fk, y_fk, z_fk, roll_fk, pitch_fk, yaw_fk = forward_kinematics(*joint_angles)
    # print("FK pose:", x_fk, y_fk, z_fk, roll_fk, pitch_fk, yaw_fk)

    # Send joint angles to hardware
    # set_joint_angles(joint_angles, port='/dev/ttyACM0')
 
    # debug_servo_position(5)

    # for i in range(6):
    #     # time.sleep(1)

    print(read_servo_position(1))
    print(read_servo_position(2))
    print(read_servo_position(3))
    print(read_servo_position(4))
    print(read_servo_position(5))
    print(read_servo_position(6))
    # set_servo_zero_position(5)  # Set servo 2 to zero position
    # set_servo_position(5, 2048)
    
    # for i in range(10):
    #     set_servo_position(5, 100*i)
    #     read_servo_position(servo_id=5)
    #     time.sleep(1)
