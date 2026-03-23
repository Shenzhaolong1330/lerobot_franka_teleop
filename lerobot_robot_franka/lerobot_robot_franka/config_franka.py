from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("franka_robot")
@dataclass
class FrankaConfig(RobotConfig):
    use_gripper: bool = True
    gripper_reverse: bool = True
    robot_ip: str = "192.168.1.104"
    gripper_bin_threshold: float = 0.98
    gripper_max_open: float = 0.0801  # gripper max open width in meters
    debug: bool = True
    close_threshold: float = 0.7
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    control_mode: str = "isoteleop"
    # Execute mode for oculus: "ee_pose" (cartesian impedance) or "joint" (joint impedance via IK)
    execute_mode: str = "ee_pose"

