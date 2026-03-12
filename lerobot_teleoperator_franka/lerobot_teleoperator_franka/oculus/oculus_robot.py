from typing import Dict, Optional, Sequence, Tuple
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

from .oculus_reader import OculusReader
from .robot import Robot

logger = logging.getLogger(__name__)


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
    
    Features:
    - Delta EE pose output (for Cartesian impedance control)
    - Joint position output via Placo IK (for joint impedance control)
    """

    # Oculus -> Robot coordinate transform matrix (for position only)
    T_OCULUS_TO_ROBOT = np.array([
        [ 0.,  0., -1.],
        [-1.,  0.,  0.],
        [ 0.,  1.,  0.],
    ])

    # Placo joint names for FR3v2
    JOINT_NAMES = [f"fr3v2_joint{i+1}" for i in range(7)]
    EE_FRAME = "fr3v2_hand_tcp"

    def __init__(
        self,
        ip: str = '192.168.110.62',
        use_gripper: bool = True,
        pose_scaler: Sequence[float] = [1.0, 1.0],
        channel_signs: Sequence[int] = [1, 1, 1, 1, 1, 1],
        enable_ik: bool = True,
        robot_ip: str = '192.168.110.15',
        robot_port: int = 4242,
        urdf_path: str = '',
        ik_iterations: int = 3,
        ik_pos_weight: float = 8.0,
        ik_ori_weight: float = 0.5,
        ik_joints_weight: float = 0.2,
        ik_regularization: float = 1e-4,
    ):  
        self._oculus_reader = OculusReader(ip_address=ip)
        self._use_gripper = use_gripper
        self._pose_scaler = pose_scaler
        self._channel_signs = channel_signs
        self._last_gripper_position = 1.0
        self._last_valid_action = np.zeros(7 if use_gripper else 6)
        self._prev_transform = None
        self._reset_requested = False

        # --- Placo IK ---
        self._ik_enabled = False
        self._robot_client = None
        self._placo_robot = None
        self._placo_solver = None
        self._pos_task = None
        self._ori_task = None
        self._joints_task = None  # 关节参考任务，锚定真实关节
        self._ik_iterations = ik_iterations
        self._ik_pos_weight = ik_pos_weight
        self._ik_ori_weight = ik_ori_weight
        self._ik_joints_weight = ik_joints_weight
        self._ik_regularization = ik_regularization
        self._last_joint_positions = None  # 最近一次 IK 解的关节位置

        # 仅在 enable_ik=True 且 urdf_path 和 robot_ip 都有效时初始化 IK
        if enable_ik and urdf_path and robot_ip:
            self._init_placo_ik(robot_ip, robot_port, urdf_path)

    def _init_placo_ik(self, robot_ip: str, robot_port: int, urdf_path: str):
        """Initialize Placo IK solver and FrankaInterfaceClient for joint reading."""
        try:
            import placo
            from lerobot_robot_franka.franka_interface_client import FrankaInterfaceClient

            self._robot_client = FrankaInterfaceClient(ip=robot_ip, port=robot_port)
            logger.info(f"[PLACO] Connected to Franka at {robot_ip}:{robot_port} for joint reading")

            # 加载 URDF
            self._placo_robot = placo.RobotWrapper(urdf_path, int(placo.Flags.ignore_collisions))
            
            # 创建 IK solver
            self._placo_solver = placo.KinematicsSolver(self._placo_robot)
            self._placo_solver.mask_fbase(True)
            self._placo_solver.enable_joint_limits(True)

            # 锁定手指关节
            self._placo_robot.set_joint("fr3v2_finger_joint1", 0.0)
            self._placo_robot.set_joint("fr3v2_finger_joint2", 0.0)

            # 读取当前关节位置初始化
            current_joints = self._robot_client.robot_get_joint_positions()
            for name, val in zip(self.JOINT_NAMES, current_joints):
                self._placo_robot.set_joint(name, val)
            self._placo_robot.update_kinematics()
            self._last_joint_positions = np.array(current_joints)

            # 获取当前末端位姿
            T_current = self._placo_robot.get_T_world_frame(self.EE_FRAME)
            current_pos = T_current[:3, 3].copy()
            current_rot = T_current[:3, :3].copy()

            # 创建位置和姿态任务
            self._pos_task = self._placo_solver.add_position_task(self.EE_FRAME, current_pos)
            self._pos_task.configure("pos", "soft", self._ik_pos_weight)

            self._ori_task = self._placo_solver.add_orientation_task(self.EE_FRAME, current_rot)
            self._ori_task.configure("ori", "soft", self._ik_ori_weight)

            # 关节参考任务：锚定当前真实关节构型，解决 7-DOF 冗余问题
            joints_dict = {name: float(val) for name, val in zip(self.JOINT_NAMES, current_joints)}
            self._joints_task = self._placo_solver.add_joints_task()
            self._joints_task.set_joints(joints_dict)
            self._joints_task.configure("joints_ref", "soft", self._ik_joints_weight)

            # 正则化
            self._placo_solver.add_regularization_task(self._ik_regularization)

            self._ik_enabled = True
            logger.info(f"[PLACO] IK solver initialized with URDF: {urdf_path}")
            logger.info(f"[PLACO] Initial EE pos: {current_pos}")
            logger.info(f"[PLACO] Weights: pos={self._ik_pos_weight}, ori={self._ik_ori_weight}, "
                         f"joints={self._ik_joints_weight}, reg={self._ik_regularization}, "
                         f"iterations={self._ik_iterations}")

        except Exception as e:
            logger.warning(f"[PLACO] Failed to initialize IK solver: {e}")
            logger.warning("[PLACO] Joint output will not be available")
            self._ik_enabled = False

    def num_dofs(self) -> int:
        if self._use_gripper:
            return 7
        else:
            return 6

    def _compute_delta_pose(self, current_transform: np.ndarray) -> np.ndarray:
        """
        Compute delta pose and map to robot coordinate system.
        
        Returns: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz] in robot frame
        """
        if self._prev_transform is None:
            return np.zeros(6)
        
        # Position delta
        oculus_delta_pos = current_transform[:3, 3] - self._prev_transform[:3, 3]
        robot_delta_pos = self.T_OCULUS_TO_ROBOT @ oculus_delta_pos
        
        # Rotation delta
        current_rot = current_transform[:3, :3]
        prev_rot = self._prev_transform[:3, :3]
        delta_rot_oculus = current_rot @ prev_rot.T
        oculus_delta_rotvec = R.from_matrix(delta_rot_oculus).as_rotvec()
        
        robot_delta_rotvec = np.array([
            oculus_delta_rotvec[2],   # robot roll  = oculus rz
            oculus_delta_rotvec[0],   # robot pitch = oculus rx
            oculus_delta_rotvec[1],   # robot yaw   = oculus ry
        ])
        
        return np.concatenate([robot_delta_pos, robot_delta_rotvec])

    def _sync_placo_from_real_joints(self):
        """
        从真实机械臂读取当前关节位置并同步到 Placo 模型。
        
        在复位（A 键）后调用，确保 Placo FK 计算的 EE pose 与实际一致，
        防止下次操作时 IK 目标基于旧位姿导致机械臂跳回。
        """
        if not self._ik_enabled or self._robot_client is None:
            return
        try:
            current_joints = self._robot_client.robot_get_joint_positions()
            for name, val in zip(self.JOINT_NAMES, current_joints):
                self._placo_robot.set_joint(name, val)
            self._placo_robot.update_kinematics()
            self._last_joint_positions = np.array(current_joints)

            # 同步 joints_task 目标
            if self._joints_task is not None:
                joints_dict = {name: float(val) for name, val in zip(self.JOINT_NAMES, current_joints)}
                self._joints_task.set_joints(joints_dict)

            # 同步 EE 任务目标到当前位姿，避免残留的旧目标
            T_current = self._placo_robot.get_T_world_frame(self.EE_FRAME)
            self._pos_task.target_world = T_current[:3, 3].copy()
            self._ori_task.R_world_frame = T_current[:3, :3].copy()

            logger.info(f"[PLACO] Synced model to real joints after reset, EE: {T_current[:3, 3]}")
        except Exception as e:
            logger.warning(f"[PLACO] Failed to sync after reset: {e}")

    def _solve_ik(self, target_ee_pose: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve IK given target EE pose [x, y, z, rx, ry, rz] (position + rotvec).
        
        每次从真实关节初始化 placo model，并将 joints_task 目标设为当前真实关节。
        这样 solver 在满足 EE 位姿约束的同时，会偏好接近当前关节构型的解，
        从而解决 7-DOF 冗余导致的关节跳变问题。
        
        Returns: np.ndarray of shape (7,) or None if failed
        """
        if not self._ik_enabled:
            return None

        try:
            # 每次都从真实关节初始化 placo model
            current_joints = self._robot_client.robot_get_joint_positions()
            for name, val in zip(self.JOINT_NAMES, current_joints):
                self._placo_robot.set_joint(name, val)
            self._placo_robot.update_kinematics()

            # 更新 joints_task 目标为当前真实关节（锚定冗余空间）
            if self._joints_task is not None:
                joints_dict = {name: float(val) for name, val in zip(self.JOINT_NAMES, current_joints)}
                self._joints_task.set_joints(joints_dict)

            # 设置 EE 目标位姿
            target_pos = target_ee_pose[:3]
            target_rot = R.from_rotvec(target_ee_pose[3:]).as_matrix()

            self._pos_task.target_world = target_pos
            self._ori_task.R_world_frame = target_rot

            # 迭代求解
            for _ in range(self._ik_iterations):
                self._placo_solver.solve(True)
                self._placo_robot.update_kinematics()

            # 读取结果
            result_joints = np.array([self._placo_robot.get_joint(name) for name in self.JOINT_NAMES])
            
            self._last_joint_positions = result_joints
            return result_joints

        except Exception as e:
            logger.warning(f"[PLACO] IK solve failed: {e}")
            return self._last_joint_positions

    def get_action(self) -> dict:
        """
        Return action dict containing both delta_ee_pose and joint_positions.
        
        Output dict keys:
        - delta_ee_pose: np.ndarray [dx, dy, dz, drx, dry, drz] (6,)
        - joint_positions: np.ndarray [j1, ..., j7] (7,) or None
        - gripper: float (gripper position)
        - reset_requested: bool
        """
        transforms, buttons = self._oculus_reader.get_transformations_and_buttons()
        
        rg_pressed = buttons.get('RG', False)
        a_pressed = buttons.get('A', False)
        
        delta_ee_pose = np.zeros(6)
        joint_positions = None

        # 检测复位完成：A 键从按下→释放，说明 franka 已经执行了 reset
        # 此时从真实关节同步 Placo 模型，防止下次操作跳回旧位置
        was_reset = self._reset_requested
        self._reset_requested = a_pressed
        if was_reset and not a_pressed:
            logger.info("[PLACO] Reset released, syncing Placo model to real joints...")
            self._sync_placo_from_real_joints()
            self._prev_transform = None  # 清空上一帧 transform，防止巨大 delta
        
        if 'r' in transforms:
            current_transform = transforms['r']
            
            if rg_pressed:
                # 计算 delta pose (robot frame)
                delta_robot = self._compute_delta_pose(current_transform)
                
                # 应用缩放和符号
                if len(self._pose_scaler) >= 2:
                    position_scale = self._pose_scaler[0]
                    orientation_scale = self._pose_scaler[1]
                    
                    delta_ee_pose[0] = delta_robot[0] * position_scale * self._channel_signs[0]
                    delta_ee_pose[1] = delta_robot[1] * position_scale * self._channel_signs[1]
                    delta_ee_pose[2] = delta_robot[2] * position_scale * self._channel_signs[2]
                    delta_ee_pose[3] = delta_robot[3] * orientation_scale * self._channel_signs[3]
                    delta_ee_pose[4] = delta_robot[4] * orientation_scale * self._channel_signs[4]
                    delta_ee_pose[5] = delta_robot[5] * orientation_scale * self._channel_signs[5]
                else:
                    delta_ee_pose = delta_robot
                
                # Placo IK: 计算目标绝对位姿 -> 求解关节角
                if self._ik_enabled and np.linalg.norm(delta_ee_pose) > 1e-6:
                    try:
                        # 用 Placo FK 获取当前 EE pose（不用 SDK，避免坐标系不一致）
                        T_placo = self._placo_robot.get_T_world_frame(self.EE_FRAME)
                        current_ee_pos = T_placo[:3, 3].copy()
                        current_ee_rot = R.from_matrix(T_placo[:3, :3].copy())
                        
                        # 目标位置 = 当前位置 + delta
                        target_pos = current_ee_pos + delta_ee_pose[:3]
                        # 目标姿态 = delta_rot * current_rot
                        delta_rot = R.from_rotvec(delta_ee_pose[3:])
                        target_rot = delta_rot * current_ee_rot
                        target_ee = np.concatenate([target_pos, target_rot.as_rotvec()])
                        
                        joint_positions = self._solve_ik(target_ee)
                    except Exception as e:
                        logger.warning(f"[PLACO] IK computation error: {e}")
                        joint_positions = self._last_joint_positions

                self._last_valid_action[:6] = delta_ee_pose
                self._prev_transform = current_transform.copy()
            else:
                self._prev_transform = None
        else:
            self._prev_transform = None

        # 夹爪
        gripper_position = self._last_gripper_position
        if self._use_gripper:
            right_trigger = buttons.get('rightTrig', (0.0,))
            if isinstance(right_trigger, tuple) and len(right_trigger) > 0:
                trigger_value = right_trigger[0]
            else:
                trigger_value = 0.0
            gripper_position = 1.0 - trigger_value
            self._last_gripper_position = gripper_position
            self._last_valid_action[6] = gripper_position

        return {
            "delta_ee_pose": delta_ee_pose,
            "joint_positions": joint_positions,
            "gripper": gripper_position,
            "reset_requested": self._reset_requested,
        }
    
    def is_reset_requested(self) -> bool:
        """Check if reset was requested (A button pressed)."""
        return getattr(self, '_reset_requested', False)

    def get_observations(self) -> Dict[str, any]:
        """
        Return observations dict containing both delta_ee_pose and joint actions.
        
        Output keys:
        - delta_ee_pose.{x,y,z,rx,ry,rz}: float
        - joint_{1..7}.pos: float (from IK, if available)
        - gripper_cmd_bin: float
        - gripper_position: float (for joint mode)
        - reset_requested: bool
        """
        action = self.get_action()
        
        obs_dict = {}
        
        # Delta EE pose
        axes = ["x", "y", "z", "rx", "ry", "rz"]
        for i, axis in enumerate(axes):
            obs_dict[f"delta_ee_pose.{axis}"] = float(action["delta_ee_pose"][i])
        
        # Joint positions (always output for dataset format consistency)
        if action["joint_positions"] is not None:
            for i in range(7):
                obs_dict[f"joint_{i+1}.pos"] = float(action["joint_positions"][i])
        else:
            for i in range(7):
                obs_dict[f"joint_{i+1}.pos"] = 0.0
        
        # Gripper (both formats for compatibility)
        obs_dict["gripper_cmd_bin"] = float(action["gripper"])
        obs_dict["gripper_position"] = float(action["gripper"])
        
        # Reset
        obs_dict["reset_requested"] = action["reset_requested"]
        
        return obs_dict


if __name__ == "__main__":
    import time
    import os

    URDF_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                              'assets', 'franka_urdf', 'fr3v2_no_mesh.urdf')
    URDF_PATH = os.path.abspath(URDF_PATH)
    
    # 创建 OculusRobot 实例 (带 IK)
    oculus = OculusRobot(
        ip='192.168.110.62',
        use_gripper=True,
        pose_scaler=[0.5, 0.5],
        channel_signs=[1, 1, 1, 1, 1, 1],
        robot_ip='192.168.110.15',
        robot_port=4242,
        urdf_path=URDF_PATH,
        ik_iterations=3,
    )
    
    print("===== Oculus Robot Test (with Placo IK) =====")
    print("Controls:")
    print("  - RG (Right Grip): Press to enable action recording")
    print("  - RTr (Right Trigger): Control gripper (press = close)")
    print("  - A button: Request robot reset")
    print("  - Right controller: Move to control end-effector")
    print(f"\nPlaco IK: {'ENABLED' if oculus._ik_enabled else 'DISABLED'}")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            action = oculus.get_action()
            delta = action["delta_ee_pose"]
            joints = action["joint_positions"]
            gripper = action["gripper"]
            reset = action["reset_requested"]
            
            reset_flag = " [RESET]" if reset else ""
            rg_status = "RG:ON " if np.any(delta != 0) else "RG:OFF"
            
            # Delta EE pose
            line1 = (f"{rg_status} Delta: X={delta[0]:+.4f} Y={delta[1]:+.4f} Z={delta[2]:+.4f} "
                      f"Rx={delta[3]:+.4f} Ry={delta[4]:+.4f} Rz={delta[5]:+.4f}")
            
            # Joint positions from IK
            if joints is not None:
                joint_str = " ".join([f"J{i+1}={j:+.3f}" for i, j in enumerate(joints)])
                line2 = f"  IK: {joint_str}"
            else:
                line2 = "  IK: N/A"
            
            print(f"\r{line1} G={gripper:.2f}{reset_flag}    ", end="")
            # Uncomment below to also see IK joints:
            # print(f"\n{line2}", end="")
            
            time.sleep(0.05)  # 20 Hz
            
    except KeyboardInterrupt:
        print("\n\n===== Test Ended =====")


