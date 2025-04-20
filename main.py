import numpy as np

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

# Inverse kinematics for 6-DOF arm (elbow-down solution)
def inverse_kinematics(x, y, z, roll, pitch, yaw):
    """
    x, y, z: desired end effector position (mm)
    roll, pitch, yaw: desired end effector orientation (radians, ZYX order)
    Returns: [theta1, theta2, theta3, theta4, theta5, theta6] (radians)
    """
    # 1. Wrist center position
    R06 = rpy_to_matrix(roll, pitch, yaw)
    nx, ny, nz = R06[:, 0]
    ox, oy, oz = R06[:, 1]
    ax, ay, az = R06[:, 2]
    # End effector offset along approach vector (z)
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

    # 5. Compute wrist orientation (theta4, theta5, theta6)
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
    T01 = rot_z(theta1) @ np.array([[1,0,0],[0,1,0],[0,0,1]])
    T12 = rot_y(theta2) @ np.array([[1,0,0],[0,1,0],[0,0,1]])
    T23 = rot_y(theta3) @ np.array([[1,0,0],[0,1,0],[0,0,1]])
    R03 = T01 @ T12 @ T23
    R36 = R03.T @ R06
    # theta4, theta5, theta6 from R36 (ZYZ Euler)
    theta5 = np.arccos(R36[2,2])
    if np.sin(theta5) < 1e-6:
        theta4 = 0
        theta6 = np.arctan2(-R36[0,1], R36[0,0])
    else:
        theta4 = np.arctan2(R36[1,2], R36[0,2])
        theta6 = np.arctan2(R36[2,1], -R36[2,0])
    return [theta1, theta2, theta3, theta4, theta5, theta6]

# Example usage:
if __name__ == "__main__":
    # Target pose (x, y, z, roll, pitch, yaw)
    x, y, z = 200, 0, 100
    roll, pitch, yaw = 0, 0, 0
    joint_angles = inverse_kinematics(x, y, z, roll, pitch, yaw)
    print("Joint angles (radians):", joint_angles)
