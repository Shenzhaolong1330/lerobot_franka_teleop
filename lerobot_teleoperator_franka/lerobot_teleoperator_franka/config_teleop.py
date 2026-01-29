from dataclasses import dataclass, field
from typing import List

from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("lerobot_teleoperator_franka")
@dataclass
class FrankaTeleopConfig(TeleoperatorConfig):
    port: str = None
    use_gripper: bool = False
    hardware_offsets: List[float] = field(default_factory=list)
    joint_ids: List[int] = field(default_factory=list)
    joint_offsets: List[float] = field(default_factory=list)
    joint_signs: List[int] = field(default_factory=list)
    gripper_config: tuple[int, float, float] = None
    pose_scaler: List[float] = field(default_factory=lambda: [1.0, 1.0])
    channel_signs: List[bool] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    control_mode: str = "isoteleop"
    # placo
    # visualize_placo: bool = False
    # placo_dt: float = 0.01