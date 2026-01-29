from setuptools import setup, find_packages
from pathlib import Path

# ====== Project root ======
ROOT = Path(__file__).parent.resolve()

setup(
    name="lerobot_franka_teleop",
    version="0.1.0",
    description="Franka teleoperation and dataset collection utilities",
    python_requires=">=3.10",
    packages=find_packages(where=".", include=["scripts*", "scripts.*"]),
    include_package_data=True,
    install_requires=[
        "send2trash",
        f"lerobot_robot_franka @ file:///{ROOT}/lerobot_robot_franka",
        f"lerobot_teleoperator_franka @ file:///{ROOT}/lerobot_teleoperator_franka"
    ],
    scripts=[
        "scripts/tools/map_gripper.sh",
        "scripts/tools/check_master_port.sh",
    ],
    entry_points={
        "console_scripts": [
            # core commands
            "franka-record = scripts.core.run_record:main",
            "franka-replay = scripts.core.run_replay:main",
            "franka-visualize = scripts.core.run_visualize:main",
            "franka-reset = scripts.core.reset_robot:main",
            # utils commands (data utilities)
            "utils-joint-offsets = scripts.utils.teleop_joint_offsets:main",

            # tools commands (helper tools)
            "tools-check-dataset = scripts.tools.check_dataset_info:main",
            "tools-check-rs = scripts.tools.rs_devices:main",

            # test commands (testing scripts)
            "test-gripper-ctrl = scripts.test.gripper_ctrl:main",
            # unified help command
            "franka-help = scripts.help.help_info:main",
        ]
    },
)
