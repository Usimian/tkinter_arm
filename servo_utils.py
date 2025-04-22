import time
import logging

from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

# Set up logger (same as in main.py)
logger = logging.getLogger("robot_arm")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)

def set_servo_position(servo_id, value, port='/dev/ttyACM0'):
    """
    Set the specified servo to the given value (servo units, -2047 to +2047).
    Example: set_servo(1, 1024) will set servo 1 to about +90 degrees.
    """
    value = int(max(-2047, min(2047, value)))
    motors = {f"motor{servo_id}": (servo_id, "sts3215")}
    config = FeetechMotorsBusConfig(port=port, motors=motors)
    bus = FeetechMotorsBus(config=config)
    bus.connect()
    bus.write("Goal_Position", value, motor_names=f"motor{servo_id}")
    bus.disconnect()
    logger.info(f"Set servo {servo_id} to value: {value}")

def read_servo_position(servo_id=None, port='/dev/ttyACM0'):
    """
    Reads the current position of the specified servo.
    Returns the position in raw units (typically -2048 to +2048), or None if not found or error.
    """
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
            logger.error(f"Error: Servo {servo_id} not found. Reason: {e}")
            try:
                bus.disconnect()
            except Exception:
                pass
            return None

def set_servo_zero_position(servo_id, port='/dev/ttyACM0'):
    """
    Sets the current position of the specified servo as the new zero (centered at 2048) in the servo's memory, without moving the servo.
    """
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
        logger.info(f"Set current position {current_pos} of servo {servo_id} as new zero (centered at 2048, offset={offset})")
    except Exception as e:
        logger.error(f"Error: Could not set servo {servo_id} zero offset. Reason: {e}")
        try:
            bus.disconnect()
        except Exception:
            pass

def debug_servo_position(servo_id, port='/dev/ttyACM0'):
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
        logger.info(f"First read: {pos1}, Second read: {pos2}, Third read: {pos3}, Fourth read: {pos4}")
        bus.disconnect()
    except Exception as e:
        logger.error(f"Error: {e}")
        try:
            bus.disconnect()
        except Exception:
            pass

def set_servo_offset(servo_id, offset, port='/dev/ttyACM0'):
    """
    Writes the specified offset value to the servo's memory.
    """
    motors = {f"motor{servo_id}": (servo_id, "sts3215")}
    config = FeetechMotorsBusConfig(port=port, motors=motors)
    bus = FeetechMotorsBus(config=config)
    try:
        bus.connect()
        bus.write("Lock", 0, motor_names=f"motor{servo_id}")
        bus.write("Offset", offset, motor_names=f"motor{servo_id}")
        bus.disconnect()
        logger.info(f"Set offset {offset} for servo {servo_id}.")
    except Exception as e:
        logger.error(f"Error: Could not set offset {offset} for servo {servo_id}. Reason: {e}")
        try:
            bus.disconnect()
        except Exception:
            pass 