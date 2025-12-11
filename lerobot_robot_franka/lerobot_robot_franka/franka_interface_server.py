'''
run on the NUC connected to the franka robot
to provide zerorpc server interface
'''

import zerorpc
from polymetis import RobotInterface
from polymetis import GripperInterface
import scipy.spatial.transform as st
import numpy as np
import torch


class FrankaInterfaceServer:
    def __init__(self):
        # self.robot = RobotInterface(ip_address="172.16.0.2")
        self.robot = RobotInterface(enforce_version=False)
        
    def gripper_initialize(self):
        self.gripper = GripperInterface()

    def gripper_goto(
        self, 
        width: float, 
        speed: float, 
        force: float, 
        blocking: bool = True
    ):
        self.gripper.goto(
            width=width,
            speed=speed,
            force=force,
            blocking=blocking,
        )

    def gripper_grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        self.gripper.grasp(
            speed=speed,
            force=force,
            grasp_width=grasp_width,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
            blocking=blocking,
        )

    def gripper_get_state(self)-> dict:
        state = self.gripper.get_state()
        return {
            # "timestamp": state.timestamp,
            "width": state.width,
            "is_moving": state.is_moving,
            "is_grasped": state.is_grasped,
            "prev_command_successful": state.prev_command_successful,
            "error_code": state.error_code,
        }

    def robot_get_joint_positions(self)-> list:
        return self.robot.get_joint_positions().numpy().tolist()

    def robot_get_joint_velocities(self)-> list:
        return self.robot.get_joint_velocities().numpy().tolist()

    def robot_get_ee_pose(self)-> list:
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()
    
    def robot_move_to_joint_positions(
        self,
        positions: list,
        time_to_go: float = None,
        delta: bool = False,
        Kq: list = None,
        Kqd: list = None,
    ):
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(positions),
            time_to_go=time_to_go,
            delta=delta,
            Kq=torch.Tensor(Kq) if Kq is not None else None,
            Kqd=torch.Tensor(Kqd) if Kqd is not None else None,
        )

    def robot_go_home(self):
        self.robot.go_home()

    def robot_move_to_ee_pose(
        self,
        position: list = None,
        orientation: list = None,
        time_to_go: float = None,
        delta: bool = False,
        Kx: list = None,
        Kxd: list = None,
        op_space_interp: bool = True,
    ):
        self.robot.move_to_ee_pose(
            position=torch.Tensor(position),
            orientation=torch.Tensor(orientation),
            time_to_go=time_to_go,
            delta=delta,
            Kx=torch.Tensor(Kx) if Kx is not None else None,
            Kxd=torch.Tensor(Kxd) if Kxd is not None else None,
            op_space_interp=op_space_interp,
        )

    def robot_start_joint_impedance_control(self, Kq: list = None, Kqd: list = None, adaptive=True,):
        self.robot.start_joint_impedance(
            Kq=torch.Tensor(Kq) if Kq is not None else None,
            Kqd=torch.Tensor(Kqd) if Kqd is not None else None,
            adaptive=adaptive,
        )

    def robot_start_cartesian_impedance_control(self, Kx: list = None, Kxd: list = None):
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx) if Kx is not None else None,
            Kxd=torch.Tensor(Kxd) if Kxd is not None else None,
        )

    def robot_update_desired_joint_positions(self, positions: np.ndarray):
        self.robot.update_desired_joint_positions(
            positions=torch.Tensor(positions)
        )

    def robot_update_desired_ee_pose(self, pose: list):
        pose = torch.Tensor(pose)
        self.robot.update_desired_ee_pose(
            position=pose[:3],
            orientation=st.Rotation.from_rotvec(pose[3:]).as_quat()
        )

    def robot_terminate_current_policy(self):
        self.robot.terminate_current_policy()

if __name__ == "__main__":
    server = FrankaInterfaceServer()
    s = zerorpc.Server(server)
    s.bind("tcp://0.0.0.0:4242")
    s.run()