def main():
    print("""
==================================================
 Franka Teleoperation Utilities - Command Reference
==================================================

Core Commands:
  franka-record           Record teleoperation dataset
  franka-replay           Replay a recorded dataset
  franka-visualize        Visualize recorded dataset
  franka-reset           Reset the robot to initial state
  franka-train          Train a policy on the recorded dataset

Utility Commands:
  utils-joint-offsets   Compute joint offsets for teleoperation

Tool Commands:
  tools-check-dataset   Check local dataset information
  tools-check-rs        Retrieve connected RealSense camera serial numbers

Shell Tools:
  map_gripper.sh        Map Gripper Serial Port
  check_master_port.sh  Get the Master Arm's Persistent Serial Identifier

Test Commands:
  test-gripper-ctrl     Run gripper control command (operate the gripper)

--------------------------------------------------
 Tip: Use 'franka-help' anytime to see this summary.
==================================================
""")
