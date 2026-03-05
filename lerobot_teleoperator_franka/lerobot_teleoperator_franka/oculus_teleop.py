#!/usr/bin/env python

"""
Oculus Quest teleoperation implementation.
"""

import logging
from typing import Any, Dict

from .base_teleop import BaseTeleop
from .config_teleop import OculusTeleopConfig
from .oculus.oculus_robot import OculusRobot

logger = logging.getLogger(__name__)


class OculusTeleop(BaseTeleop):
    """
    Teleoperation using Oculus Quest controller.
    
    This teleoperation mode uses an Oculus Quest controller to control the robot's
    end-effector in Cartesian space. The output is delta pose (position and orientation changes).
    
    Controls:
    - RG (Right Grip): Must be pressed to enable action recording
    - RTr (Right Trigger): Controls gripper (0.0 = open, 1.0 = closed)
    - Right controller pose: Controls end-effector delta pose
    """
    
    config_class = OculusTeleopConfig
    name = "OculusTeleop"
    
    def __init__(self, config: OculusTeleopConfig):
        super().__init__(config)
        self.oculus_robot: OculusRobot = None
    
    def _get_teleop_name(self) -> str:
        return "OculusTeleop"
    
    @property
    def action_features(self) -> dict:
        """Return action features for oculus mode (delta ee pose + joint positions)."""
        features = {}
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"delta_ee_pose.{axis}"] = float
        # Joint positions from Placo IK (always included for oculus mode)
        for i in range(7):
            features[f"joint_{i+1}.pos"] = float
        features["gripper_cmd_bin"] = float
        return features
    
    def _connect_impl(self) -> None:
        """Connect to Oculus Quest."""
        self.oculus_robot = OculusRobot(
            ip=self.cfg.ip,
            use_gripper=self.cfg.use_gripper,
            pose_scaler=self.cfg.pose_scaler,
            channel_signs=self.cfg.channel_signs,
            robot_ip=self.cfg.robot_ip,
            robot_port=self.cfg.robot_port,
            urdf_path=self.cfg.urdf_path,
            ik_iterations=self.cfg.ik_iterations,
            ik_pos_weight=self.cfg.ik_pos_weight,
            ik_ori_weight=self.cfg.ik_ori_weight,
            ik_joints_weight=self.cfg.ik_joints_weight,
            ik_regularization=self.cfg.ik_regularization,
        )
        ik_status = "enabled" if self.oculus_robot._ik_enabled else "disabled"
        logger.info(f"[TELEOP] Oculus connected at IP: {self.cfg.ip}, IK: {ik_status}")
    
    def _disconnect_impl(self) -> None:
        """Disconnect from Oculus Quest."""
        # OculusRobot doesn't have explicit disconnect, just let it be garbage collected
        pass
    
    def _get_action_impl(self) -> Dict[str, Any]:
        """Get delta pose from Oculus controller."""
        return self.oculus_robot.get_observations()
