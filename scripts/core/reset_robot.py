import yaml
from pathlib import Path
from typing import Dict, Any
from lerobot_robot_franka import FrankaConfig, Franka
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 创建机器人配置
    robot_config = FrankaConfig(
        robot_ip=cfg["record"]["robot"]["ip"],
        use_gripper=cfg["record"]["robot"]["use_gripper"],
        close_threshold=cfg["record"]["robot"]["close_threshold"],
        gripper_bin_threshold=cfg["record"]["robot"]["gripper_bin_threshold"],
        gripper_reverse=cfg["record"]["robot"]["gripper_reverse"],
        gripper_max_open=cfg["record"]["robot"]["gripper_max_open"],
        debug=False
    )
    
    # 创建机器人实例并连接
    robot = Franka(robot_config)
    robot.connect()
    
    # 重置机器人到初始位置
    logging.info("Resetting robot to home position...")
    robot.reset()
    
    # # 打开夹爪
    # logging.info("Opening gripper...")
    # robot.gripper_goto(
    #     width=robot_config.gripper_max_open, 
    #     speed=robot._gripper_speed, 
    #     force=robot._gripper_force, 
    #     blocking=True
    # )
    
    # 断开连接
    robot.disconnect()
    logging.info("Robot reset completed successfully.")

if __name__ == "__main__":
    main()