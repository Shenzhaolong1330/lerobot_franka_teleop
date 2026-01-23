import yaml
from pathlib import Path
from typing import Dict, Any
from scripts.utils.dataset_utils import generate_dataset_name, update_dataset_info
from lerobot_robot_franka import FrankaConfig, Franka
from lerobot_teleoperator_franka import FrankaTeleopConfig, FrankaTeleop
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.control_utils import init_keyboard_listener
import shutil
import termios, sys
from lerobot.utils.constants import HF_LEROBOT_HOME
from scripts.utils.teleop_joint_offsets import get_start_joints, compute_joint_offsets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

class RecordConfig:
    def __init__(self, cfg: Dict[str, Any]):
        storage = cfg["storage"]
        task = cfg["task"]
        time = cfg["time"]
        cam = cfg["cameras"]
        robot = cfg["robot"]
        teleop = cfg["teleop"]
        dxl_cfg = teleop["dynamixel_config"]
        sm_cfg = teleop["spacemouse_config"]

        # global config
        self.repo_id: str = cfg["repo_id"]
        self.debug: bool = cfg.get("debug", True)
        self.fps: str = cfg.get("fps", 15)
        self.dataset_path: str = HF_LEROBOT_HOME / self.repo_id
        self.user_info: str = cfg.get("user_notes", None)

        if teleop["control_mode"] == "isoteleop":
            # teleop config
            self.port = dxl_cfg["port"]
            self.use_gripper = dxl_cfg["use_gripper"]  
            self.joint_ids = dxl_cfg["joint_ids"]
            self.joint_offsets = dxl_cfg["joint_offsets"]
            self.joint_signs = dxl_cfg["joint_signs"]
            self.gripper_config = dxl_cfg["gripper_config"]
            self.hardware_offsets = dxl_cfg["hardware_offsets"]
            self.control_mode = teleop.get("control_mode", "isoteleop")
        elif teleop["control_mode"] == "spacemouse":
            self.use_gripper = sm_cfg["use_gripper"]
            self.pose_scaler = sm_cfg["pose_scaler"]
            self.channel_signs = sm_cfg["channel_signs"]
            self.control_mode = teleop.get("control_mode", "spacemouse")

        # robot config
        self.robot_ip: str = robot["ip"]
        # self.gripper_port: str = robot["gripper_port"]
        self.use_gripper: str = robot["use_gripper"]
        self.close_threshold = robot["close_threshold"]
        self.gripper_reverse: str = robot["gripper_reverse"]
        self.gripper_bin_threshold: float = robot["gripper_bin_threshold"]

        # task config
        self.num_episodes: int = task.get("num_episodes", 1)
        self.display: bool = task.get("display", True)
        self.task_description: str = task.get("description", "default task")
        self.resume: bool = task.get("resume", "False")
        self.resume_dataset: str = task["resume_dataset"]

        # time config
        self.episode_time_sec: int = time.get("episode_time_sec", 60)
        self.reset_time_sec: int = time.get("reset_time_sec", 10)
        self.save_mera_period: int = time.get("save_mera_period", 1)

        # cameras config
        self.wrist_cam_serial: str = cam["wrist_cam_serial"]
        self.exterior_cam_serial: str = cam["exterior_cam_serial"]
        self.width: int = cam["width"]
        self.height: int = cam["height"]

        # storage config
        self.push_to_hub: bool = storage.get("push_to_hub", False)


def run_reset(record_cfg: RecordConfig):

    wrist_image_cfg = RealSenseCameraConfig(serial_number_or_name=record_cfg.wrist_cam_serial,
                                    fps=record_cfg.fps,
                                    width=record_cfg.width,
                                    height=record_cfg.height,
                                    color_mode=ColorMode.RGB,
                                    use_depth=False,
                                    rotation=Cv2Rotation.NO_ROTATION)

    exterior_image_cfg = RealSenseCameraConfig(serial_number_or_name=record_cfg.exterior_cam_serial,
                                    fps=record_cfg.fps,
                                    width=record_cfg.width,
                                    height=record_cfg.height,
                                    color_mode=ColorMode.RGB,
                                    use_depth=False,
                                    rotation=Cv2Rotation.NO_ROTATION)

    # Create the robot and teleoperator configurations
    camera_config = {"wrist_image": wrist_image_cfg, "exterior_image": exterior_image_cfg}
    if record_cfg.control_mode == "isoteleop":
        teleop_config = FrankaTeleopConfig(        
            port=record_cfg.port,
            use_gripper=record_cfg.use_gripper,
            hardware_offsets=record_cfg.hardware_offsets,
            joint_ids=record_cfg.joint_ids,
            joint_offsets=record_cfg.joint_offsets,
            joint_signs=record_cfg.joint_signs,
            gripper_config=record_cfg.gripper_config,
            control_mode=record_cfg.control_mode)
    elif record_cfg.control_mode == "spacemouse":
        teleop_config = FrankaTeleopConfig(
            use_gripper=record_cfg.use_gripper,
            pose_scaler=record_cfg.pose_scaler,
            channel_signs=record_cfg.channel_signs,
            control_mode=record_cfg.control_mode,       
        )
    
    robot_config = FrankaConfig(
        robot_ip=record_cfg.robot_ip,
        cameras = camera_config,
        debug = record_cfg.debug,
        close_threshold = record_cfg.close_threshold,
        use_gripper = record_cfg.use_gripper,
        gripper_reverse = record_cfg.gripper_reverse,
        gripper_bin_threshold = record_cfg.gripper_bin_threshold,
        control_mode = record_cfg.control_mode,
    )
    # Initialize the robot and 
    robot = Franka(robot_config)
    robot.connect()

    robot.reset()

    # Clean up
    logging.info("Reset Done.")
    robot.disconnect()


def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    record_cfg = RecordConfig(cfg["record"])
    run_reset(record_cfg)
