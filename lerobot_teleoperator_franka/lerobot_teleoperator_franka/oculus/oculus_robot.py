from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R

from .oculus_reader import OculusReader
from .robot import Robot


class OculusRobot(Robot):
    """
    A class representing a Oculus Quest 3/3s robot controller.
    
    Controls:
    - RG (Right Grip): Must be pressed to enable action recording
    - RTr (Right Trigger): Controls gripper (0.0 = open, 1.0 = closed)
    - Right controller pose: Controls end-effector delta pose
    
    Coordinate Systems:
        Oculus: X(right), Y(up), Z(backward/towards user)
        Robot:  X(forward), Y(left), Z(up)
    
    Transformation matrix from Oculus to Robot:
        robot_x =  -oculus_z   (oculus backward -> robot forward)
        robot_y =  -oculus_x   (oculus right    -> robot left)
        robot_z =   oculus_y   (oculus up       -> robot up)
    
    As a rotation matrix (applied to both position and orientation):
        T = [[ 0,  0, -1],
             [-1,  0,  0],
             [ 0,  1,  0]]
    """

    # Oculus -> Robot coordinate transform matrix (for position only)
    T_OCULUS_TO_ROBOT = np.array([
        [ 0.,  0., -1.],
        [-1.,  0.,  0.],
        [ 0.,  1.,  0.],
    ])

    def __init__(
        self,
        ip: str = '192.168.110.62',
        use_gripper: bool = True,
        pose_scaler: Sequence[float] = [1.0, 1.0],
        channel_signs: Sequence[int] = [1, 1, 1, 1, 1, 1],
    ):  
        self._oculus_reader = OculusReader(ip_address=ip)
        self._use_gripper = use_gripper
        self._pose_scaler = pose_scaler
        self._channel_signs = channel_signs
        self._last_gripper_position = 1.0  # 默认夹爪张开状态
        self._last_valid_action = np.zeros(7 if use_gripper else 6)
        self._prev_transform = None  # 上一帧的 4x4 变换矩阵
        self._reset_requested = False  # A 按钮重置请求

        
    def num_dofs(self) -> int:
        if self._use_gripper:
            return 7
        else:
            return 6

    def _compute_delta_pose(self, current_transform: np.ndarray) -> np.ndarray:
        """
        Compute delta pose and map to robot coordinate system.
        
        Position: use matrix T_OCULUS_TO_ROBOT for transformation.
        Rotation: compute delta rotvec in Oculus frame, then explicitly
                  map each component to the correct robot axis.
        
        Oculus rotvec components (right-hand rule):
            oculus_rx = rotation around X (right)    -> tilting hand left/right
            oculus_ry = rotation around Y (up)       -> yaw hand left/right 
            oculus_rz = rotation around Z (backward) -> rolling hand
            
        Robot axes:
            robot_rx = roll  (around X = forward)
            robot_ry = pitch (around Y = left)
            robot_rz = yaw   (around Z = up)
        
        Mapping (follows same logic as position: oz->-rx, ox->-ry, oy->rz):
            robot_rx (roll)  = -oculus_rz  (oculus Z-axis rotation -> robot X-axis rotation)
            robot_ry (pitch) =  oculus_rx  (oculus X-axis rotation -> robot Y-axis rotation)
            robot_rz (yaw)   = -oculus_ry  (oculus Y-axis rotation -> robot Z-axis rotation)
        
        Returns: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz] in robot frame
        """
        if self._prev_transform is None:
            return np.zeros(6)
        
        # --- Position delta (in Oculus frame -> Robot frame via matrix) ---
        oculus_delta_pos = current_transform[:3, 3] - self._prev_transform[:3, 3]
        robot_delta_pos = self.T_OCULUS_TO_ROBOT @ oculus_delta_pos
        
        # --- Rotation delta (in Oculus frame) ---
        current_rot = current_transform[:3, :3]
        prev_rot = self._prev_transform[:3, :3]
        delta_rot_oculus = current_rot @ prev_rot.T
        oculus_delta_rotvec = R.from_matrix(delta_rot_oculus).as_rotvec()
        
        # --- Explicit axis mapping for rotation ---
        # oculus_rotvec = [rx, ry, rz] in Oculus frame
        oculus_rx = oculus_delta_rotvec[0]  # around Oculus X (right)
        oculus_ry = oculus_delta_rotvec[1]  # around Oculus Y (up)
        oculus_rz = oculus_delta_rotvec[2]  # around Oculus Z (backward)
        
        robot_delta_rotvec = np.array([
            oculus_rz,  # robot roll  (around robot_x=forward) from oculus_rz (around oculus_z=backward, negate)
            oculus_rx,  # robot pitch (around robot_y=left)    from oculus_rx (around oculus_x=right, negate for axis flip but rotation direction same)
            oculus_ry,  # robot yaw   (around robot_z=up)      from oculus_ry (around oculus_y=up, negate for convention)
        ])
        
        return np.concatenate([robot_delta_pos, robot_delta_rotvec])

    def get_action(self) -> np.ndarray:
        """
        Return the current robot actions including gripper control.
        
        The delta pose is computed directly in robot coordinate system
        using rotation matrix transformation, avoiding axis-swapping issues.
        
        Output: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, (gripper)]
        All values are in robot frame.
        """
        transforms, buttons = self._oculus_reader.get_transformations_and_buttons()
        
        # Check buttons
        rg_pressed = buttons.get('RG', False)
        a_pressed = buttons.get('A', False)
        
        # Initialize action (always start with zeros)
        action = np.zeros(7 if self._use_gripper else 6)
        
        # Check for reset request (A button)
        self._reset_requested = a_pressed
        
        if 'r' in transforms:
            current_transform = transforms['r']  # 4x4 transformation matrix
            
            if rg_pressed:
                # Compute delta pose already in robot frame
                delta_robot = self._compute_delta_pose(current_transform)
                
                # Apply scaling and channel signs
                if len(self._pose_scaler) >= 2:
                    position_scale = self._pose_scaler[0]  
                    orientation_scale = self._pose_scaler[1]
                    
                    # Position
                    action[0] = delta_robot[0] * position_scale * self._channel_signs[0]
                    action[1] = delta_robot[1] * position_scale * self._channel_signs[1]
                    action[2] = delta_robot[2] * position_scale * self._channel_signs[2]
                    
                    # Orientation
                    action[3] = delta_robot[3] * orientation_scale * self._channel_signs[3]
                    action[4] = delta_robot[4] * orientation_scale * self._channel_signs[4]
                    action[5] = delta_robot[5] * orientation_scale * self._channel_signs[5]
                else:
                    action[:6] = delta_robot
                
                self._last_valid_action[:6] = action[:6]
                
                # Update previous transform
                self._prev_transform = current_transform.copy()
            else:
                # RG not pressed, return zero action and reset previous transform
                self._prev_transform = None
        else:
            # No right controller detected
            self._prev_transform = None

        # Handle gripper control with RTr (Right Trigger)
        if self._use_gripper:
            right_trigger = buttons.get('rightTrig', (0.0,))
            if isinstance(right_trigger, tuple) and len(right_trigger) > 0:
                trigger_value = right_trigger[0]
            else:
                trigger_value = 0.0
            
            # Map trigger value to gripper position: 0.0 (not pressed) = open, 1.0 (pressed) = closed
            gripper_position = 1.0 - trigger_value  # Invert: trigger pressed = closed (0.0)
            
            self._last_gripper_position = gripper_position
            action[6] = gripper_position
            self._last_valid_action[6] = gripper_position
        
        return action
    
    def is_reset_requested(self) -> bool:
        """Check if reset was requested (A button pressed)."""
        return getattr(self, '_reset_requested', False)

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Return the current robot observations by formatting the action data.
        """
        action_data = self.get_action()
        
        obs_dict = {}
        axes = ["x", "y", "z", "rx", "ry", "rz"]
        
        if len(action_data) >= 6:
            for i, axis in enumerate(axes):
                obs_dict[f"delta_ee_pose.{axis}"] = float(action_data[i])
        else:
            for axis in axes:
                obs_dict[f"delta_ee_pose.{axis}"] = float(0.0)
        
        if self._use_gripper and len(action_data) >= 7:
            obs_dict["gripper_cmd_bin"] = float(action_data[6])
        else:
            obs_dict["gripper_cmd_bin"] = None
        
        # Add reset request flag
        obs_dict["reset_requested"] = self._reset_requested
        
        return obs_dict


if __name__ == "__main__":
    import time
    
    # 创建 OculusRobot 实例
    oculus = OculusRobot(
        ip='192.168.110.62',  # 修改为你的 Oculus IP
        use_gripper=True,
        pose_scaler=[0.5, 0.5],  # 缩放因子
        channel_signs=[1, 1, 1, 1, 1, 1]
    )
    
    print("===== Oculus Robot Test =====")
    print("Controls:")
    print("  - RG (Right Grip): Press to enable action recording")
    print("  - RTr (Right Trigger): Control gripper (press = close)")
    print("  - A button: Request robot reset")
    print("  - Right controller: Move to control end-effector")
    print("\nCoordinate Mapping:")
    print("  Oculus X(right) -> Robot -Y(left)")
    print("  Oculus Y(up)    -> Robot  Z(up)")
    print("  Oculus Z(back)  -> Robot  X(forward)")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            action = oculus.get_action()
            obs = oculus.get_observations()
            
            # 打印 action 和状态
            reset_flag = " [RESET]" if obs.get("reset_requested", False) else ""
            rg_status = "RG:ON " if action[:6].any() else "RG:OFF"
            
            print(f"\r{rg_status} Robot: X={action[0]:+.4f} Y={action[1]:+.4f} Z={action[2]:+.4f} "
                  f"Rx={action[3]:+.4f} Ry={action[4]:+.4f} Rz={action[5]:+.4f} "
                  f"Gripper={action[6]:.2f}{reset_flag}    ", end="")
            
            time.sleep(0.05)  # 20 Hz
            
    except KeyboardInterrupt:
        print("\n\n===== Test Ended =====")


