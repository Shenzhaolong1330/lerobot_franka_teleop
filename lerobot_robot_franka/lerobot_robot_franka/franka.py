import logging
import time
import threading
from pathlib import Path
from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot
from .config_franka import FrankaConfig
from typing import Any, Dict
import yaml
from .franka_interface_client import FrankaInterfaceClient
from scipy.spatial.transform import Rotation as R
import numpy as np
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class Franka(Robot):
    config_class = FrankaConfig
    name = "franka"

    def __init__(self, config: FrankaConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)

        self.config = config
        self._is_connected = False
        self._robot = None
        self._initial_pose = None
        self._prev_observation = None
        self._num_joints = 7
        self._gripper_force = 70
        self._gripper_speed = 0.2
        self._gripper_epsilon = 1.0
        self._gripper_position = 1
        self._dt = 0.002
        self._last_gripper_position = 1

        self.iteration = 0
        
    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")

        # Connect to robot
        self._robot = self._check_franka_connection(self.config.robot_ip)
        
        # Initialize gripper
        if self.config.use_gripper:
            self._gripper = self._check_gripper_connection(self.config.robot_ip)


        # Connect cameras
        logger.info("\n===== [CAM] Initializing Cameras =====")
        for cam_name, cam in self.cameras.items():
            cam.connect()
            logger.info(f"[CAM] {cam_name} connected successfully.")
        logger.info("===== [CAM] Cameras Initialized Successfully =====\n")

        self.is_connected = True
        logger.info(f"[INFO] {self.name} env initialization completed successfully.\n")


    def _check_gripper_connection(self, robot_ip: str):
        logger.info("\n===== [GRIPPER] Initializing gripper...")
        self._robot.gripper_initialize()
        print("Homing gripper")
        self._robot.gripper_goto(width=self.config.gripper_max_open, speed=self._gripper_speed, force=self._gripper_force, blocking=True)
        logger.info("===== [GRIPPER] Gripper initialized successfully.\n")
        return None


    def _check_franka_connection(self, robot_ip: str):
        try:
            logger.info("\n===== [ROBOT] Connecting to Franka robot =====")
            
            franka = FrankaInterfaceClient(ip=robot_ip, port=4242)
            franka.robot_start_joint_impedance_control()

            joint_positions = franka.robot_get_joint_positions()
            if joint_positions is not None and len(joint_positions) == 7:
                formatted_joints = [round(j, 4) for j in joint_positions]
                logger.info(f"[ROBOT] Current joint positions: {formatted_joints}")
                logger.info("===== [ROBOT] Franka connected successfully =====\n")
            else:
                logger.info("===== [ERROR] Failed to read joint positions. Check connection or remote control mode =====")

        except Exception as e:
            logger.info("===== [ERROR] Failed to connect to Franka robot =====")
            logger.info(f"Exception: {e}\n")

        return franka


    def reset(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        # Reset robot
        self._robot.robot_go_home()
        self._robot.gripper_goto(width=self.config.gripper_max_open, speed=self._gripper_speed, force=self._gripper_force, blocking=True)
        logger.info("===== [ROBOT] Robot reset successfully =====\n")


    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            # joint positions
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,
            "joint_7.pos": float,
            # gripper state
            "gripper_state_norm": float, # raw position in [0,1]
            # "gripper_raw_bin": float, # raw position bin (0 or 1)
            "gripper_cmd_bin": float, # action command bin (0 or 1)
            # joint velocities
            "joint_1.vel": float,
            "joint_2.vel": float,
            "joint_3.vel": float,
            "joint_4.vel": float,
            "joint_5.vel": float,
            "joint_6.vel": float,
            "joint_7.vel": float,
            # # joint accelerations
            # "joint_1.acc": float,
            # "joint_2.acc": float,
            # "joint_3.acc": float,       
            # "joint_4.acc": float,
            # "joint_5.acc": float,
            # "joint_6.acc": float,
            # "joint_7.acc": float,
            # # joint forces
            # "joint_1.force": float,
            # "joint_2.force": float,
            # "joint_3.force": float,
            # "joint_4.force": float,
            # "joint_5.force": float,
            # "joint_6.force": float,
            # "joint_7.force": float,
            # end effector pose
            "ee_pose.x": float,
            "ee_pose.y": float,
            "ee_pose.z": float,
            "ee_pose.rx": float,
            "ee_pose.ry": float,
            "ee_pose.rz": float,
            # # end effector velocity
            # "ee_vel.x": float,
            # "ee_vel.y": float,
            # "ee_vel.z": float,
            # "ee_vel.rx": float,
            # "ee_vel.ry": float,
            # "ee_vel.rz": float,
            # # end effector acceleration
            # "ee_acc.x": float,
            # "ee_acc.y": float,
            # "ee_acc.z": float,
            # # end effector force and torque
            # "ee_force.x": float,
            # "ee_force.y": float,
            # "ee_force.z": float,
            # "ee_force.rx": float,
            # "ee_force.ry": float,
            # "ee_force.rz": float,
        }

    @property
    def action_features(self) -> dict[str, type]:
        if self.config.control_mode == "isoteleop":
            # print("using control mode: ", self.config.control_mode)
            return {
                "joint_1.pos": float,
                "joint_2.pos": float,
                "joint_3.pos": float,
                "joint_4.pos": float,
                "joint_5.pos": float,
                "joint_6.pos": float,
                "joint_7.pos": float,
                "gripper_position": float,
            }
        elif self.config.control_mode == "spacemouse":
            # print("using control mode: ", self.config.control_mode)
            return {
                "delta_ee_pose.x": float,
                "delta_ee_pose.y": float,
                "delta_ee_pose.z": float,
                "delta_ee_pose.rx": float,
                "delta_ee_pose.ry": float,
                "delta_ee_pose.rz": float,
                "gripper_cmd_bin": float,
            }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.iteration += 1
        if self.config.control_mode == "isoteleop":
            # print("using control mode: ", self.config.control_mode)
            target_joints = np.array([action[f"joint_{i+1}.pos"] for i in range(self._num_joints)])
            noise_amplitude = 0.008 if self.iteration % 2 == 0 else 0.0
            if noise_amplitude > 0:
            # 生成均匀分布的随机噪声 [-amplitude, +amplitude]
                joint_noise = np.random.uniform(-noise_amplitude, noise_amplitude, self._num_joints)
                target_joints += joint_noise
            if not self.config.debug:
                # 获取当前关节位置
                joint_positions = self._robot.robot_get_joint_positions()
                
                # 计算最大关节位置差
                max_delta = (np.abs(joint_positions - target_joints)).max()
                
                # 如果最大差值超过阈值，则进行插值移动
                if max_delta > 0.3:  # 设置一个合理的阈值
                    print("MOVING TOO FAST! SLWO DOWN!")
                    steps = min(int(max_delta / 0.02), 100)
                    
                    for i, jnt in enumerate(np.linspace(joint_positions, target_joints, steps)):
                        self._robot.robot_update_desired_joint_positions(jnt)
                        time.sleep(0.05)
                        
                else:
                    # 直接发送目标位置
                    self._robot.robot_update_desired_joint_positions(target_joints)
                
                
            if "gripper_position" in action:
                gripper_position = 0.0 if action["gripper_position"]  < self.config.close_threshold else 1.0
                if self.config.gripper_reverse:
                    gripper_position = 1 - gripper_position

                if gripper_position != self._last_gripper_position:
                    self._robot.gripper_goto(
                        width = gripper_position*self.config.gripper_max_open, 
                        speed=self._gripper_speed, 
                        force = self._gripper_force, 
                        # epsilon_outer=self._gripper_epsilon
                    )
                    self._last_gripper_position = gripper_position
                
                gripper_state = self._robot.gripper_get_state()
                gripper_state_norm = max(0.0, min(1.0, gripper_state["width"]/self.config.gripper_max_open))
                if self.config.gripper_reverse:
                    gripper_state_norm = 1 - gripper_state_norm

                self._gripper_position = gripper_state_norm

        elif self.config.control_mode == "spacemouse":
            # print("using control mode: ", self.config.control_mode)
            delta_ee_pose = np.array([action[f"delta_ee_pose.{axis}"] for axis in ["x", "y", "z", "rx", "ry", "rz"]])

            if not self.config.debug:
                
                import scipy.spatial.transform as st

                ee_pose = self._robot.robot_get_ee_pose()

                if np.linalg.norm(delta_ee_pose) >= 0.01:
                    target_position = ee_pose[:3] + delta_ee_pose[:3]
                    current_rot = st.Rotation.from_rotvec(ee_pose[3:])
                    delta_rot = st.Rotation.from_rotvec(delta_ee_pose[3:])
                    target_rotation = delta_rot * current_rot  # 注意顺序：增量旋转 * 当前旋转
                    target_rotvec = target_rotation.as_rotvec()
                    target_ee_pose = np.concatenate([target_position, target_rotvec])
                    # print("target_ee_pose:", target_ee_pose[3])
                    self._robot.robot_update_desired_ee_pose(target_ee_pose)
                else:
                    pass
            
            if "gripper_cmd_bin" in action:
                gripper_position = action["gripper_cmd_bin"]
                if self.config.gripper_reverse:
                    gripper_position = 1 - gripper_position

                if gripper_position != self._last_gripper_position:
                    self._robot.gripper_goto(
                        width = gripper_position*self.config.gripper_max_open, 
                        speed=self._gripper_speed, 
                        force = self._gripper_force, 
                        # epsilon_outer=self._gripper_epsilon
                    )
                    self._last_gripper_position = gripper_position
                
                gripper_state = self._robot.gripper_get_state()
                gripper_state_norm = max(0.0, min(1.0, gripper_state["width"]/self.config.gripper_max_open))
                if self.config.gripper_reverse:
                    gripper_state_norm = 1 - gripper_state_norm

                self._gripper_position = gripper_state_norm
            
        return action

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # Read joint positions
        joint_position = self._robot.robot_get_joint_positions()
        # print("joint_position:", joint_position)
        # Read joint velocities
        joint_velocity = self._robot.robot_get_joint_velocities()
        # print("joint_velocity:", joint_velocity)
        # Read end effector pose
        ee_pose = self._robot.robot_get_ee_pose()
        # print("ee_pose:", ee_pose)

        # Read ee speed
        # ee_speed = robot_state.O_dP_EE_d
        # print("ee_speed:", ee_speed)
        
        # Prepare observation dictionary
        obs_dict = {}
        for i in range(len(joint_position)):
            obs_dict[f"joint_{i+1}.pos"] = float(joint_position[i])
            obs_dict[f"joint_{i+1}.vel"] = float(joint_velocity[i])

        for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            obs_dict[f"ee_pose.{axis}"] = float(ee_pose[i])
  
        # for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
        #     obs_dict[f"ee_vel.{axis}"] = float(ee_speed[i])

        if self.config.use_gripper:

            obs_dict["gripper_state_norm"] = self._gripper_position
            obs_dict["gripper_cmd_bin"] = self._last_gripper_position
        else:
            obs_dict["gripper_state_norm"] = None
            obs_dict["gripper_cmd_bin"] = None

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        self._prev_observation = obs_dict

        return obs_dict

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        for cam in self.cameras.values():
            cam.disconnect()

        self.is_connected = False
        logger.info(f"[INFO] ===== All {self.name} connections have been closed =====")

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return self.is_connected
    
    def configure(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
           cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def cameras(self):
        return self._cameras

    @cameras.setter
    def cameras(self, value):
        self._cameras = value

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    class RecordConfig:
        def __init__(self, cfg: Dict[str, Any]):
            robot = cfg["robot"]
            cam = cfg["cameras"]
            self.fps: str = cfg.get("fps", 15)

            # robot config
            self.robot_ip = robot["ip"]
            self.use_gripper = robot["use_gripper"]
            self.close_threshold = robot["close_threshold"]
            self.gripper_bin_threshold = robot["gripper_bin_threshold"]
            self.gripper_reverse = robot["gripper_reverse"]
            self.control_mode = robot["control_mode"]


            # cameras config
            self.wrist_cam_serial: str = cam["wrist_cam_serial"]
            self.exterior_cam_serial: str = cam["exterior_cam_serial"]
            self.width: int = cam["width"]
            self.height: int = cam["height"]


    with open(Path(__file__).parent / "config" / "cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)


    record_cfg = RecordConfig(cfg["record"])

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

    robot_config = FrankaConfig(
            robot_ip=record_cfg.robot_ip,
            cameras = camera_config,
            debug = False,
            close_threshold = record_cfg.close_threshold,
            use_gripper = record_cfg.use_gripper,
            gripper_reverse = record_cfg.gripper_reverse,
            gripper_bin_threshold = record_cfg.gripper_bin_threshold,
            control_mode = record_cfg.control_mode
        )
    franka = Franka(robot_config)
    franka.connect()