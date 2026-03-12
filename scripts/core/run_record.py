import yaml
from pathlib import Path
from typing import Dict, Any
from scripts.utils.dataset_utils import generate_dataset_name, update_dataset_info
from lerobot_robot_franka import FrankaConfig, Franka
from lerobot_teleoperator_franka import (
    DynamixelTeleopConfig,
    SpacemouseTeleopConfig,
    OculusTeleopConfig,
    create_teleop,
    create_teleop_config,
)
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.control_utils import init_keyboard_listener
from send2trash import send2trash
import termios, sys
from lerobot.utils.constants import HF_LEROBOT_HOME
from scripts.utils.teleop_joint_offsets import get_start_joints, compute_joint_offsets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from dataclasses import field

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


class RecordConfig:
    """Configuration class for recording sessions."""
    
    def __init__(self, cfg: Dict[str, Any]):
        storage = cfg["storage"]
        task = cfg["task"]
        time = cfg["time"]
        cam = cfg["cameras"]
        robot = cfg["robot"]
        policy = cfg["policy"]
        teleop = cfg["teleop"]
        
        # Global config
        self.repo_id: str = cfg["repo_id"]
        self.debug: bool = cfg.get("debug", True)
        self.fps: str = cfg.get("fps", 15)
        self.dataset_path: str = HF_LEROBOT_HOME / self.repo_id
        self.user_info: str = cfg.get("user_notes", None)
        self.run_mode: str = cfg.get("run_mode", "run_record")
        self.rename_map: dict[str, str] = field(default_factory=dict)
        
        # Teleop config - parse based on control mode
        self.control_mode = teleop.get("control_mode", "isoteleop")
        self._parse_teleop_config(teleop)
        
        # Policy config
        self._parse_policy_config(policy)
        
        # Robot config
        self.robot_ip: str = robot["ip"]
        self.use_gripper: bool = robot["use_gripper"]
        self.close_threshold = robot["close_threshold"]
        self.gripper_reverse: bool = robot["gripper_reverse"]
        self.gripper_bin_threshold: float = robot["gripper_bin_threshold"]
        self.gripper_max_open: float = robot.get("gripper_max_open", 0.08)
        self.execute_mode: str = robot.get("execute_mode", "ee_pose")  # "ee_pose" or "joint"
        
        # Task config
        self.num_episodes: int = task.get("num_episodes", 1)
        self.display: bool = task.get("display", True)
        self.task_description: str = task.get("description", "default task")
        self.resume: bool = task.get("resume", False)
        self.resume_dataset: str = task.get("resume_dataset", "")
        
        # Time config
        self.episode_time_sec: int = time.get("episode_time_sec", 60)
        self.reset_time_sec: int = time.get("reset_time_sec", 10)
        self.save_mera_period: int = time.get("save_mera_period", 1)
        
        # Cameras config
        self.wrist_cam_serial: str = cam["wrist_cam_serial"]
        self.exterior_cam_serial: str = cam["exterior_cam_serial"]
        self.width: int = cam["width"]
        self.height: int = cam["height"]
        
        # Storage config
        self.push_to_hub: bool = storage.get("push_to_hub", False)
    
    def _parse_teleop_config(self, teleop: Dict[str, Any]) -> None:
        """Parse teleoperation configuration based on control mode."""
        if self.control_mode == "isoteleop":
            dxl_cfg = teleop["dynamixel_config"]
            self.port = dxl_cfg["port"]
            self.use_gripper = dxl_cfg["use_gripper"]
            self.joint_ids = dxl_cfg["joint_ids"]
            self.joint_offsets = dxl_cfg["joint_offsets"]
            self.joint_signs = dxl_cfg["joint_signs"]
            self.gripper_config = dxl_cfg["gripper_config"]
            self.hardware_offsets = dxl_cfg["hardware_offsets"]
        
        elif self.control_mode == "spacemouse":
            sm_cfg = teleop["spacemouse_config"]
            self.use_gripper = sm_cfg["use_gripper"]
            self.pose_scaler = sm_cfg["pose_scaler"]
            self.channel_signs = sm_cfg["channel_signs"]
        
        elif self.control_mode == "oculus":
            oculus_cfg = teleop.get("oculus_config", {})
            self.use_gripper = oculus_cfg.get("use_gripper", True)
            self.oculus_ip = oculus_cfg.get("ip", "192.168.110.62")
            self.pose_scaler = oculus_cfg.get("pose_scaler", [1.0, 1.0])
            self.channel_signs = oculus_cfg.get("channel_signs", [1, 1, 1, 1, 1, 1])
            # Placo IK settings (now read from placo section)
            placo_cfg = teleop.get("placo", {})
            self.oculus_robot_ip = placo_cfg.get("robot_ip", "192.168.110.15")
            self.oculus_robot_port = placo_cfg.get("robot_port", 4242)
            urdf_path = placo_cfg.get("ik_urdf_path", "")
            # Resolve relative urdf_path to project root (lerobot_franka_isoteleop/)
            if urdf_path and not Path(urdf_path).is_absolute():
                project_root = Path(__file__).resolve().parent.parent.parent  # scripts/core/ -> scripts/ -> project root
                urdf_path = str((project_root / urdf_path).resolve())
            self.oculus_urdf_path = urdf_path
            self.oculus_enable_ik = placo_cfg.get("enable_ik", True)
            self.oculus_ik_iterations = placo_cfg.get("ik_iterations", 3)
            self.oculus_ik_pos_weight = placo_cfg.get("ik_pos_weight", 8.0)
            self.oculus_ik_ori_weight = placo_cfg.get("ik_ori_weight", 0.5)
            self.oculus_ik_joints_weight = placo_cfg.get("ik_joints_weight", 0.2)
            self.oculus_ik_regularization = placo_cfg.get("ik_regularization", 1e-4)
        
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")
    
    def _parse_policy_config(self, policy: Dict[str, Any]) -> None:
        """Parse policy configuration."""
        policy_type = policy["type"]
        if policy_type == "act":
            from lerobot.policies import ACTConfig
            self.policy = ACTConfig(
                device=policy["device"],
                push_to_hub=policy["push_to_hub"],
            )
        elif policy_type == "diffusion":
            from lerobot.policies import DiffusionConfig
            self.policy = DiffusionConfig(
                device=policy["device"],
                push_to_hub=policy["push_to_hub"],
            )
        else:
            raise ValueError(f"No config for policy type: {policy_type}")
        
        if policy.get("pretrained_path"):
            self.policy.pretrained_path = policy["pretrained_path"]
    
    def create_teleop_config(self):
        """Create teleoperation configuration object."""
        if self.control_mode == "isoteleop":
            return DynamixelTeleopConfig(
                port=self.port,
                use_gripper=self.use_gripper,
                hardware_offsets=self.hardware_offsets,
                joint_ids=self.joint_ids,
                joint_offsets=self.joint_offsets,
                joint_signs=self.joint_signs,
                gripper_config=self.gripper_config,
            )
        elif self.control_mode == "spacemouse":
            return SpacemouseTeleopConfig(
                use_gripper=self.use_gripper,
                pose_scaler=self.pose_scaler,
                channel_signs=self.channel_signs,
            )
        elif self.control_mode == "oculus":
            return OculusTeleopConfig(
                use_gripper=self.use_gripper,
                ip=self.oculus_ip,
                pose_scaler=self.pose_scaler,
                channel_signs=self.channel_signs,
                enable_ik=self.oculus_enable_ik,
                robot_ip=self.oculus_robot_ip,
                robot_port=self.oculus_robot_port,
                urdf_path=self.oculus_urdf_path,
                ik_iterations=self.oculus_ik_iterations,
                ik_pos_weight=self.oculus_ik_pos_weight,
                ik_ori_weight=self.oculus_ik_ori_weight,
                ik_joints_weight=self.oculus_ik_joints_weight,
                ik_regularization=self.oculus_ik_regularization,
            )
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")


def check_joint_offsets(record_cfg: RecordConfig):
    """Check the joint_offsets is set and correct."""

    if record_cfg.joint_offsets is None:
        raise ValueError("joint_offsets is None. Please check teleop_joint_offsets.py output.")

    start_joints = get_start_joints(record_cfg)
    if start_joints is None:
        raise RuntimeError("Failed to retrieve start joints from Franka robot.")

    joint_offsets = compute_joint_offsets(record_cfg, start_joints)

    if joint_offsets != record_cfg.joint_offsets:
        raise ValueError(
            f"Computed joint_offsets {joint_offsets} != provided joint_offsets {record_cfg.joint_offsets}. "
            "Please check teleop_joint_offsets.py output."
        )
    logging.info("Joint offsets verified successfully.")

def handle_incomplete_dataset(dataset_path):
    if dataset_path.exists():
        print(f"====== [WARNING] Detected an incomplete dataset folder: {dataset_path} ======")
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        ans = input("Do you want to delete it? (y/n): ").strip().lower()
        if ans == "y":
            print(f"====== [DELETE] Removing folder: {dataset_path} ======")
            # Send to trash
            send2trash(dataset_path)
            print("====== [DONE] Incomplete dataset folder deleted successfully. ======")
        else:
            print("====== [KEEP] Incomplete dataset folder retained, please check manually. ======")

def run_record(record_cfg: RecordConfig):
    print("====== [START] Starting recording ======")
    try:
        dataset_name, data_version = generate_dataset_name(record_cfg)

        # Check joint offsets
        # if not record_cfg.debug:
        #     check_joint_offsets(record_cfg)        
        
        # Create RealSenseCamera configurations
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
        
        # Create teleop config using the new method
        teleop_config = record_cfg.create_teleop_config()
        
        robot_config = FrankaConfig(
            robot_ip=record_cfg.robot_ip,
            cameras = camera_config,
            debug = record_cfg.debug,
            close_threshold = record_cfg.close_threshold,
            use_gripper = record_cfg.use_gripper,
            gripper_reverse = record_cfg.gripper_reverse,
            gripper_bin_threshold = record_cfg.gripper_bin_threshold,
            gripper_max_open = record_cfg.gripper_max_open,
            control_mode = record_cfg.control_mode,
            execute_mode = record_cfg.execute_mode,
        )
        # Initialize the robot
        robot = Franka(robot_config)

        # Configure the dataset features
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
        dataset_features = {**action_features, **obs_features}

        if record_cfg.resume:
            dataset = LeRobotDataset(
                dataset_name,
            )

            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer()
            sanity_check_dataset_robot_compatibility(dataset, robot, record_cfg.fps, dataset_features)
        else:
            # # Create the dataset
            dataset = LeRobotDataset.create(
                repo_id=dataset_name,
                fps=record_cfg.fps,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=True,
                image_writer_threads=4,
            )
        # Set the episode metadata buffer size to 1, so that each episode is saved immediately
        dataset.meta.metadata_buffer_size = record_cfg.save_mera_period

        # Initialize the keyboard listener and rerun visualization
        _, events = init_keyboard_listener()
        init_rerun(session_name="recording")

        # Create processor
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
        preprocessor = None
        postprocessor = None

        # configure the teleop and policy
        if record_cfg.run_mode == "run_record":
            logging.info("====== [INFO] Running in teleoperation mode ======")
            teleop = create_teleop(teleop_config)
            policy = None
        elif record_cfg.run_mode == "run_policy":
            logging.info("====== [INFO] Running in policy mode ======")
            policy = make_policy(record_cfg.policy, ds_meta=dataset.meta)
            teleop = None
        elif record_cfg.run_mode == "run_mix":
            logging.info("====== [INFO] Running in mixed mode ======")
            policy = make_policy(record_cfg.policy, ds_meta=dataset.meta)
            teleop = create_teleop(teleop_config)
        
        if policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=record_cfg.policy,
                pretrained_path=record_cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, {}),  # 使用空字典作为rename_map
                preprocessor_overrides={
                    "device_processor": {"device": record_cfg.policy.device},
                    "rename_observations_processor": {"rename_map": {}},  # 使用空字典作为rename_map
                },
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        episode_idx = 0

        while episode_idx < record_cfg.num_episodes and not events["stop_recording"]:
            logging.info(f"====== [RECORD] Recording episode {episode_idx + 1} of {record_cfg.num_episodes} ======")
            record_loop(
                robot=robot,
                events=events,
                fps=record_cfg.fps,
                teleop=teleop,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                control_time_s=record_cfg.episode_time_sec,
                single_task=record_cfg.task_description,
                display_data=record_cfg.display,
            )

            if events["rerecord_episode"]:
                logging.info("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (episode_idx < record_cfg.num_episodes - 1 or events["rerecord_episode"]):
                while True:
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                    user_input = input("====== [WAIT] Press Enter to reset the environment ======")
                    if user_input == "":
                        break  
                    else:
                        logging.info("====== [WARNING] Please press only Enter to continue ======")

                logging.info("====== [RESET] Resetting the environment ======")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=record_cfg.fps,
                    teleop=teleop,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    control_time_s=record_cfg.reset_time_sec,
                    single_task=record_cfg.task_description,
                    display_data=record_cfg.display,
                )

            episode_idx += 1

        # Clean up
        logging.info("Stop recording")
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        dataset.finalize()

        update_dataset_info(record_cfg, dataset_name, data_version)
        if record_cfg.push_to_hub:
            dataset.push_to_hub()

    except Exception as e:
        logging.info(f"====== [ERROR] {e} ======")
        dataset_path = Path(HF_LEROBOT_HOME) / dataset_name
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)

    except KeyboardInterrupt:
        logging.info("\n====== [INFO] Ctrl+C detected, cleaning up incomplete dataset... ======")
        dataset_path = Path(HF_LEROBOT_HOME) / dataset_name
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)


def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    record_cfg = RecordConfig(cfg["record"])
    run_record(record_cfg)

if __name__ == "__main__":
    main()