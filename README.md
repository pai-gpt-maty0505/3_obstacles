# SO-101 障害物回避 AI

## 1. 本リポジトリの目的

本リポジトリは、SO-101 ロボットアームを対象として、
AIを用いた障害物回避動作をGazebo上で実現することを目的としています。

MoveIt 2 / OMPLを教師として安全な障害物回避軌道を生成し、
その軌道を教師データとしてMLP（Multi-Layer Perceptron）を学習します。

学習後は、障害物位置と動作進行度を入力として、
MLPがSO-101の5つの関節角度を推論します。

最終的には、オンラインでOMPLによる再計画を行わず、
AIが推論した軌道をMoveIt Planning Sceneで安全性確認したうえで、
Gazebo上のSO-101に実行させます。

システム構成：

障害物位置
↓
MLP
↓
関節軌道
↓
MoveIt 2による安全性確認
↓
ros2_control
↓
Gazebo上のSO-101


## 2. 目的のための手順

### Step 1. SO-101 / ROS 2 / Gazebo / MoveIt 2の動作確認

以下を確認します。

- SO-101がGazeboに表示される
- Joint Stateが取得できる
- arm_controllerがactive
- MoveIt 2からSO-101を操作できる
- /check_state_validityが利用できる
- /apply_planning_sceneが利用できる


### Step 2. IKデータ生成

ik_dataset_generator.py を使用し、
手先位置からSO-101の関節角度をIKで求めます。


### Step 3. 障害物回避教師データ生成

obstacle_dataset_generator_v4_2.py を使用します。

障害物を以下の3クラスに分類します。

- left
- center
- right

処理手順：

1. 障害物なしの基準軌道を生成
2. Planning Sceneへ障害物を配置
3. 基準軌道が障害物によって遮られることを確認
4. OMPL / RRTConnectで回避軌道を生成
5. 軌道全体を衝突判定
6. 基準軌道との差を確認
7. 条件を満たした軌道のみ教師データとして保存

最終的に、

left   : 100 episodes
center : 100 episodes
right  : 100 episodes
Total  : 300 episodes

を生成しました。


### Step 4. 学習用データセット作成

prepare_dataset_v5.py を使用します。

可変長軌道を各Episode 30 stepに補間します。

Train : 240 episodes
Val   : 30 episodes
Test  : 30 episodes


### Step 5. MLP学習

train_mlp.py を使用します。

入力：

- obs_x
- obs_y
- obs_z
- progress

出力：

- shoulder_pan
- shoulder_lift
- elbow_flex
- wrist_flex
- wrist_roll

MLP構成：

4
↓
128 ReLU
↓
128 ReLU
↓
64 ReLU
↓
5


### Step 6. MLP推論

infer_mlp.py を使用します。

Testデータでの結果：

Overall MAE  : 約4.83 deg
Overall RMSE : 約7.61 deg


### Step 7. 推論軌道の安全性確認

validate_mlp_trajectories.py を使用します。

30 step間も補間してMoveIt 2で衝突判定します。

結果：

Safe        : 28 / 30
Collision   : 2 / 30
Safety rate : 93.3 %

デモ候補：

left   : Episode 4
center : Episode 173
right  : Episode 203


### Step 8. Gazeboデモ

run_mlp_gazebo_demo.py を使用します。

本番実行時にはOMPLによるオンライン再計画を行わず、
MLP推論軌道をそのまま使用します。

PRE-FLIGHT：

python3 -u run_mlp_gazebo_demo.py --episode 4

Episode 4を実行：

python3 -u run_mlp_gazebo_demo.py --execute --episode 4

3条件を連続実行：

python3 -u run_mlp_gazebo_demo.py --execute


## 3. 環境設定

### OS

Ubuntu 24.04

### ROS 2

ROS 2 Jazzy

環境設定：

source /opt/ros/jazzy/setup.bash


### SO-101 ROS 2 Workspace

ROS 2パッケージ：

~/so101_ws

環境設定：

source ~/so101_ws/install/setup.bash


### 本リポジトリ

作業ディレクトリ：

~/pai_obstacle

通常は以下を実行します。

cd ~/pai_obstacle
source /opt/ros/jazzy/setup.bash
source ~/so101_ws/install/setup.bash


### Gazebo

Gazebo Harmonic

SO-101 + Gazebo + MoveIt 2：

ros2 launch so101_bringup so101_gazebo_moveit.launch.py \
  world_file:=empty.world \
  use_camera:=false \
  rviz_config_file:=move_group.rviz


### MoveIt 2

主に以下を使用します。

- Planning Scene
- IK
- OMPL
- RRTConnect
- State Validity Check
- CollisionObject


### ros2_control

主に以下を使用します。

- arm_controller
- gripper_controller
- joint_state_broadcaster


### Python

ROS 2関連はシステムPythonを使用します。

python3 -c "import rclpy; print('rclpy OK')"

AI学習時はPython仮想環境を使用します。

source ~/so101_ws/.venv/bin/activate


### GPU / PyTorch

- NVIDIA GeForce RTX 3060 12 GB
- PyTorch 2.13.0 + CUDA 13.0


## 主なコード

- ik_dataset_generator.py
- obstacle_dataset_generator_v4_2.py
- prepare_dataset_v5.py
- train_mlp.py
- infer_mlp.py
- validate_mlp_trajectories.py
- run_mlp_gazebo_demo.py
- demo_obstacle.sdf

資料：

doc/SO101_障害物回避AI_全体整理_1-21.docx
