from setuptools import setup, find_packages

setup(
    name="lerobot_robot_franka",
    version="0.0.1",
    description="LeRobot Franka integration",
    author="Zhaolong Shen",
    author_email="shenzhaolong@buaa.edu.cn",
    packages=find_packages(),
    install_requires=[
        "pyrealsense2",
        "scipy",
        "zerorpc",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
