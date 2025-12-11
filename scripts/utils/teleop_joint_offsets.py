from pathlib import Path
from typing import Dict, Any, List
import yaml
import numpy as np
import logging
from lerobot_robot_franka import FrankaInterfaceClient
from lerobot_teleoperator_franka.dynamixel import DynamixelDriver

# ------------------------ Logging Setup ------------------------ #
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ------------------------ Robot Functions ------------------------ #
def get_start_joints(cfg) -> List[float]:
    """Connects to the Franka robot and retrieves current joint positions."""
    try:
        logger.info("\n===== [ROBOT] Connecting to Franka robot =====")
        robot = FrankaInterfaceClient(cfg.robot_ip)
        robot.robot_start_joint_impedance_control()
        joint_positions = (robot.robot_get_joint_positions()).tolist()
        logger.info(f"[ROBOT] Current joint positions: {joint_positions}")
        logger.info("===== [ROBOT] Franka connected successfully =====\n")
        return joint_positions
    except Exception as e:
        logger.error("===== [ERROR] Failed to connect to Franka robot =====")
        logger.error(f"Exception: {e}\n")
        return []

# ------------------------ Offset Calculation ------------------------ #
def compute_joint_offsets(cfg, start_joints: List[float]):
    """Compute offsets for Dynamixel joints to match the Franka joint positions."""
    
    driver = DynamixelDriver(cfg.joint_ids, port=cfg.port, baudrate=57600)

    # Warmup reads
    for _ in range(10):
        driver.get_joints()

    def joint_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        """Calculate error between adjusted joint state and start joint."""
        joint_sign = cfg.joint_signs[index]
        joint_val = joint_sign * (joint_state[index] - offset)
        return np.abs(joint_val - start_joints[index])

    # Compute best offsets
    curr_joints = driver.get_joints_deg()
    curr_modified_joints = driver.get_positions(cfg.hardware_offsets)
    logger.info("Dynamixel current joint positions: %s", curr_joints)
    logger.info("Dynamixel current modified joint positions: %s", curr_modified_joints)

    # Close driver
    driver.close()

    best_offsets = []
    
    for i in range(len(cfg.joint_ids)):
        best_offset = 0
        best_error = float('inf')
        for offset in np.linspace(-8 * np.pi, 8 * np.pi, 33):  # intervals of pi/2
            error = joint_error(offset, i, curr_modified_joints)
            if error < best_error:
                best_error = error
                best_offset = offset
        best_offsets.append(float(best_offset))

    logger.info("Joint offsets: %s", [round(x, 3) for x in best_offsets])

    return [round(x, 3) for x in best_offsets]

# ------------------------ Config Loader ------------------------ #
class RecordConfig:
    """Configuration for teleoperation and robot."""
    def __init__(self, cfg: Dict[str, Any]):
        teleop = cfg["teleop"]
        robot = cfg["robot"]
        dxl_cfg = teleop["dynamixel_config"]

        # Teleop config
        self.port = dxl_cfg["port"]
        self.joint_ids = dxl_cfg["joint_ids"]
        self.joint_signs = dxl_cfg["joint_signs"]
        self.hardware_offsets = dxl_cfg["hardware_offsets"]

        # Robot config
        self.robot_ip: str = robot["ip"]

def run(record_cfg):
    start_joints = get_start_joints(record_cfg)
    print(start_joints)
    if start_joints:
        return compute_joint_offsets(record_cfg, start_joints)
    else:
        raise RuntimeError("Failed to retrieve start joints from Franka robot.")

# ------------------------ Main ------------------------ #
def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    record_cfg = RecordConfig(cfg["record"])
    run(record_cfg)
