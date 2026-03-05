from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("lerobot_teleoperator_franka")
@dataclass
class BaseTeleopConfig(TeleoperatorConfig):
    """Base configuration for all teleoperation modes."""
    control_mode: str = "isoteleop"
    use_gripper: bool = True


@TeleoperatorConfig.register_subclass("dynamixel_teleop")
@dataclass
class DynamixelTeleopConfig(BaseTeleopConfig):
    """Configuration for Dynamixel-based isomorphic teleoperation."""
    control_mode: str = "isoteleop"
    port: str = "/dev/ttyUSB0"
    hardware_offsets: List[float] = field(default_factory=lambda: [0.0] * 7)
    joint_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    joint_offsets: List[float] = field(default_factory=lambda: [0.0] * 7)
    joint_signs: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 1])
    gripper_config: Optional[Tuple[int, float, float]] = None  # (id, min, max)


@TeleoperatorConfig.register_subclass("spacemouse_teleop")
@dataclass
class SpacemouseTeleopConfig(BaseTeleopConfig):
    """Configuration for SpaceMouse teleoperation."""
    control_mode: str = "spacemouse"
    pose_scaler: List[float] = field(default_factory=lambda: [1.0, 1.0])  # [position_scale, orientation_scale]
    channel_signs: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])  # [x, y, z, rx, ry, rz]


@TeleoperatorConfig.register_subclass("oculus_teleop")
@dataclass
class OculusTeleopConfig(BaseTeleopConfig):
    """Configuration for Oculus Quest teleoperation."""
    control_mode: str = "oculus"
    ip: str = "192.168.110.62"
    pose_scaler: List[float] = field(default_factory=lambda: [1.0, 1.0])  # [position_scale, orientation_scale]
    channel_signs: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])  # [x, y, z, rx, ry, rz]
    # Placo IK settings
    robot_ip: str = "192.168.110.15"    # Franka robot IP for reading joint states
    robot_port: int = 4242              # Franka zerorpc port
    urdf_path: str = ""                 # Path to URDF file (no mesh version)
    ik_iterations: int = 3              # Number of IK solver iterations
    ik_pos_weight: float = 8.0          # IK position task weight
    ik_ori_weight: float = 0.5          # IK orientation task weight
    ik_joints_weight: float = 0.2       # IK joints anchoring task weight (7-DOF redundancy)
    ik_regularization: float = 1e-4     # IK regularization weight


# Legacy compatibility: FrankaTeleopConfig maps to DynamixelTeleopConfig
FrankaTeleopConfig = DynamixelTeleopConfig