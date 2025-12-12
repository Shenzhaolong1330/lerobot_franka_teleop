'''
running on the user machine 
to connect to the franka_interface_server 
'''

import numpy as np
import zerorpc


class FrankaInterfaceClient:
    def __init__(self, ip='192.168.100.63', port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def gripper_initialize(self):
        self.server.gripper_initialize()

    def gripper_goto(
        self, 
        width: float, 
        speed: float, 
        force: float, 
        blocking: bool = True
    ):
        # self.server.gripper_goto(
        #     width=width,
        #     speed=speed,
        #     force=force,
        #     blocking=blocking,
        # )
        self.server.gripper_goto(width, speed, force, blocking)

    def gripper_grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        # self.server.gripper_grasp(
        #     speed=speed,
        #     force=force,
        #     grasp_width=grasp_width,
        #     epsilon_inner=epsilon_inner,
        #     epsilon_outer=epsilon_outer,
        #     blocking=blocking,
        # )
        self.server.gripper_grasp(
            speed,
            force,
            grasp_width,
            epsilon_inner,
            epsilon_outer,
            blocking,
        )

    def gripper_get_state(self)-> dict:
        return self.server.gripper_get_state()

    
    def robot_get_joint_positions(self):
        '''
        list -> np.ndarray
        '''
        joint_positions = np.array(self.server.robot_get_joint_positions())
        return joint_positions

    def robot_get_joint_velocities(self):
        '''
        list -> np.ndarray
        '''
        joint_velocities = np.array(self.server.robot_get_joint_velocities())
        return joint_velocities
    
    def robot_get_ee_pose(self):
        '''
        list -> np.ndarray
        '''
        pose = np.array(self.server.robot_get_ee_pose())
        return pose

    def robot_move_to_joint_positions(
        self,
        positions: np.ndarray,
        time_to_go: float = None,
        delta: bool = False,
        Kq: np.ndarray = None,
        Kqd: np.ndarray = None,
    ):
        self.server.robot_move_to_joint_positions(
            positions.tolist(), 
            time_to_go, 
            delta, 
            Kq.tolist() if Kq is not None else None, 
            Kqd.tolist() if Kqd is not None else None
        )

    def robot_go_home(self):
        self.server.robot_go_home()

    def robot_move_to_ee_pose(
        self,
        position: np.ndarray = None,
        orientation: np.ndarray = None,
        time_to_go: float = None,
        delta: bool = False,
        Kx: np.ndarray = None,
        Kxd: np.ndarray = None,
        op_space_interp: bool = True,
    ):
        self.server.robot_move_to_ee_pose(
            position.tolist(),
            orientation.tolist(),
            time_to_go,
            delta,
            Kx.tolist() if Kx is not None else None,
            Kxd.tolist() if Kxd is not None else None,
            op_space_interp,
        )

    def robot_start_joint_impedance_control(
        self, 
        Kq: np.ndarray = None, 
        Kqd: np.ndarray = None, 
        adaptive: bool = True,
    ):
        self.server.robot_start_joint_impedance_control(
            Kq.tolist() if Kq is not None else None,
            Kqd.tolist() if Kqd is not None else None,
            adaptive,
        )
        print(f"[ROBOT] Joint impedance control started")

    def robot_start_cartesian_impedance_control(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.robot_start_cartesian_impedance_control(
            Kx.tolist() if Kx is not None else None,
            Kxd.tolist() if Kxd is not None else None,
        )
        print(f"[ROBOT] Cartesian impedance control started")


    def robot_update_desired_joint_positions(self, positions: np.ndarray):
        self.server.robot_update_desired_joint_positions(positions.tolist())

    def robot_update_desired_ee_pose(self, pose: np.ndarray):
        self.server.robot_update_desired_ee_pose(pose.tolist())

    def robot_terminate_current_policy(self):
        self.server.robot_terminate_current_policy()

    def close(self):
        self.server.close()

if __name__ == "__main__":
    
    Franka = FrankaInterfaceClient()
    
    Franka.gripper_goto(width=0.08, speed=0.1, force=10.0)
    
    # Franka.gripper_goto(width=0.08, speed=0.1, force=10.0)
    # Reset
    Franka.robot_go_home()

    # Get joint positions
    joint_positions = Franka.robot_get_joint_positions()
    print(f"Current joint positions: {joint_positions}")

    # Command robot to pose (move 4th and 6th joint)
    joint_positions_desired = np.array(
        [-0.14, -0.02, -0.05, -1.57, 0.05, 1.50, -0.91]
    )
    print(f"\nMoving joints to: {joint_positions_desired} ...\n")
    state_log = Franka.robot_move_to_joint_positions(joint_positions_desired, time_to_go=2.0)

    # Get updated joint positions
    joint_positions = Franka.robot_get_joint_positions()
    print(f"New joint positions: {joint_positions}")