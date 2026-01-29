from setuptools import setup, find_packages

setup(
    name="lerobot_teleoperator_franka",
    version="0.0.1",
    description="LeRobot teleoperator integration",
    author="Zhaolong Shen",
    author_email="shenzhaolong@buaa.edu.cn",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "dynamixel_sdk",
        "easyhid",
        "placo"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
