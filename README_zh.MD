# Franka 同构遥操作数据采集与回放指南
<p align="center">
  <img src="assets/robot.png" alt="Record" width="600">
  <br>
</p>

## 0. Franka control说明
在这个版本中，使用[Polymetis](https://polymetis-docs.github.io/)对Franka 进行控制，由于Polymetis不支持python3.10，并且推荐单独在NUC上运行，因此在这里用Zerorpc进行远程通信实现，具体流程如下：
### 1. 在NUC上安装[Polymetis](https://polymetis-docs.github.io/)
我们开源了针对franka优化后的[Polymetis](https://github.com/Shenzhaolong1330/fairo-franka/tree/main)源码，详细参考[安装教程](https://github.com/Shenzhaolong1330/fairo-franka/blob/main/polymetis/FRANKA.md)进行安装。
### 2. 在NUC上启动Server
分别启动[robot interface](https://github.com/Shenzhaolong1330/fairo-franka/blob/main/polymetis/polymetis/python/scripts/launch_robot.py)，[gripper interface](https://github.com/Shenzhaolong1330/fairo-franka/blob/main/polymetis/polymetis/python/scripts/launch_gripper.py)和[interface server](https://github.com/Shenzhaolong1330/fairo-franka/blob/main/polymetis/polymetis/python/polymetis/franka_interface_server.py)
```
# On the NUC
python launch_robot.py robot_client=franka_hardware
python launch_gripper.py gripper=franka_hand
python franka_interface_server.py
```
### 3. 在本地启动[interface client](lerobot_robot_franka\lerobot_robot_franka\franka_interface_client.py)
在尝试链接Franka前，启动[interface client](lerobot_robot_franka\lerobot_robot_franka\franka_interface_client.py)
```
# On the local machine
python franka_interface_client.py
```
[interface client](lerobot_robot_franka\lerobot_robot_franka\franka_interface_client.py)中的函数功能与Polymetis中的函数功能基本一致，使用方法可以参考[Polymetis文档](https://polymetis-docs.github.io/)。


## 1. 环境配置

### 1.1 创建并激活 conda 环境
```bash
conda create -n franka_data python=3.10
conda activate franka_data
```

### 1.2 安装 lerobot
```bash
# 安装指定版本 0.3.4
# git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
pip install -e .
```

### 1.3 克隆并安装 Franka 遥操作控制源码
```bash
mkdir franka_data_collection && cd franka_data_collection
git clone https://github.com/Shenzhaolong1330/lerobot_franka_isoteleop.git
cd lerobot_franka_isoteleop
pip install -e .
```


### 1.4 查看可运行的命令
运行以下命令以查看所有可用的脚本：
```bash
franka-help
# ==================================================
#  Franka Teleoperation Utilities - Command Reference
# ==================================================

# Core Commands:
#   franka-record           Record teleoperation dataset
#   franka-replay           Replay a recorded dataset
#   franka-visualize        Visualize recorded dataset

# Utility Commands:
#   utils-joint-offsets   Compute joint offsets for teleoperation

# Tool Commands:
#   tools-check-dataset   Check local dataset information
#   tools-check-rs        Retrieve connected RealSense camera serial numbers

# Shell Tools:
#   map_gripper.sh        Map Gripper Serial Port
#   check_master_port.sh  Get the Master Arm's Persistent Serial Identifier

# Test Commands:
#   test-gripper-ctrl     Run gripper control command (operate the gripper)

# --------------------------------------------------
#  Tip: Use 'franka-help' anytime to see this summary.
# ==================================================
```
## 2. 获取和配置必要参数

### 2.1 获取 RealSense 相机序列号
请确保每次仅连接一个相机:
```bash
tools-check-rs
```

### 2.2 固定夹爪串口映射
例如，将夹爪映射到 `/dev/franka_left_gripper`, 执行此操作前，请确保仅连接一个夹爪的usb设备。
```bash
map_gripper.sh franka_left_gripper
```
随后，将设定的映射值填入`cfg.yaml`中的`gripper_port`字段
### 2.3 获取主臂的固定串口标识符
执行此操作前，请确保仅连接一个主臂的usb设备。
```bash
check_master_port.sh
```
随后，将获取的串口标识符填入`cfg.yaml`中的`port`字段，并修改设备访问权限:
```bash
sudo chmod 666 <your_master_port>
```
### 2.4 获取主臂-从臂关节角的误差周期
> **⚠️ 注意：在记录数据前，务必完成此步骤以设置正确的误差周期。否则，从臂可能发生意外的动作。**

在执行此操作前，请先手动拖动主臂，使其关节角与从臂的当前关节角尽量保持一致。然后运行以下命令来计算主臂与从臂关节角的误差周期：
```bash
utils-joint-offsets
```
随后，请将脚本输出的`joint_offsets`值填入`cfg.yaml`中的对应配置项

## 3. 数据集记录

### 3.1 上传数据到 Hugging Face（可选）
1. 在 `cfg.yaml` 中设置：
```yaml
push_to_hub: True
```
2. 获取 Hugging Face 账号的 token 并登录：
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} 
huggingface-cli whoami  # 检查是否登录成功
```

### 3.2 开始记录
1. 打开 Franka 示教器上的 **Remote Control**。
2. 确认 `cfg.yaml` 中参数配置正确。
3. 为了确保数据收集过程规范，请先阅读 `9. 数据集命名与记录详解` 以了解数据记录细节，然后再运行以下命令：
```bash
franka-record
```
<p align="center">
  <img src="assets/record.png" alt="Record" width="600">
  <br>
  <b>Figure 1: Record</b>
</p>

## 4. 数据集回放
```bash
franka-replay #注意cfg配置
```

## 5. 数据集可视化
```bash
franka-visualize #注意cfg配置
```
<p align="center">
  <img src="assets/visualize.png" alt="Visualization" width="600">
  <br>
  <b>Figure 2: Visualization</b>
</p>

## 6. 数据集追加与恢复
如果你已经在指定的 `repo_id` 下录制过数据集，可以在 `cfg.yaml` 中将 `resume` 设置为 `True`，并在 `resume_dataset` 中填写要追加的数据集名称，以便在现有数据集的基础上继续录制。然后运行以下命令：
```bash
franka-record
```

## 7. 数据集合并
如果你在不同阶段分别录制了数据集，请确保各阶段的数据集具有不同的 `repo_id`, 完成录制后，可通过以下命令将它们合并为一个数据集: 
```bash
lerobot-edit-dataset 
    --repo_id <merged_repo_id> 
    --operation.type merge 
    --operation.repo_ids "['<repo_id_1>', '<repo_id2>']"
```
- 更多数据集处理命令，请参考 [LeRobot](https://huggingface.co/docs/lerobot/using_dataset_tools)

## 8. 录制控制按键说明
1. **右方向键**  
   1. 按下右方向键: 结束当前 episode 的录制并保存数据，程序将暂停等待。
   2. 进入**复位**阶段: 按回车键以继续（此时从臂会直接跟随主臂运动），请通过主臂将从臂复位至初始位姿。
   3. 再次按下右方向键，开始录制下一个episode
2. **左方向键**  
   - 按下后：重新开始当前 episode 的录制

3. **Esc 键**  
   - 按下后：退出整个录制任务，并保存当前已录制的数据

4. **Ctrl+c或抛出异常**
   - 按下或发生异常时：自动进入异常处理阶段，提示用户是否删除当前未完成的录制数据
  
## 9. 数据集命名与记录详解
### 1. 数据集命名
<p align="center">
  <img src="assets/dataset.png" alt="dataset" width="600">
  <br>
  <b>Figure 3: Dataset</b>
</p>

<p align="center">
  <img src="assets/dataset_info.png" alt="dataset_info" width="600">
  <br>
  <b>Figure 4: Dataset Info</b>
</p>

1. 数据集默认保存在 `~/.cache/huggingface/lerobot` 目录下，包含三类内容：

   - `dataset_info.txt`：自动记录本地数据集信息，包括以下字段：`record_id`、`name`、`task`、`date`、`version`、`user_info` 和 `type`。其中，`user_info` 可以通过 `cfg.yaml` 中的 `user_notes` 进行注解。

   - `dataset_info_backup`：当通过 `tools-check-dataset` 手动更新 `dataset_info.txt` 时，保存旧的记录备份。

   - 数据集文件夹：存放实际的数据集内容。

  
2. 数据集命名格式为 `[description]_[date]_[version]`。其中：
   - `description` 来源于 `cfg.yaml` 中的 `repo_id=<user_name>/<description>`；
   - `date` 会自动生成；
   - `version` 会根据是否存在同名数据集( `repo_id` 相同)自动生成新版本号。
3. description 命名规则：`task.description -> Verb_SourceObject_prep_TargetObject`。  
   即将 `cfg.yaml` 中的 `task.description` 按照上述格式映射生成。例如：  
   `Pick up the green cube and put it into the trash bin -> pick_greencube_into_trashbin`。

### 2. 数据记录说明
1. 完成步骤 `2. 获取和配置必要参数`。
2. 根据任务内容，在 `cfg.yaml` 的 `task.description` 中填写指令，并根据数据集命名规则设置 `repo_id`。
3. 检查并调整 `cfg.yaml` 中的其他参数，确保配置正确。
4. 运行命令 `franka-record`，为确保安全， 请保证主臂与从臂关节角近似，`franka-record` 会自动执行 `2.4 获取主臂-从臂关节角误差周期` 检查，然后按照 `8. 录制控制按键说明` 完成录制操作。
5. 数据记录结束后，数据集同级目录会生成 `dataset_info.txt` 文件，用于保存本地数据集信息。如果手动删除过数据集，可通过以下命令更新记录信息：
```bash
tools-check-dataset
```
