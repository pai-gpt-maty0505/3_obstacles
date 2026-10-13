# SO-101 AI障害物回避・剛体搬送

## 1. 最終課題の目的

SO-101ロボットアームが、Gazebo内で**動的な剛体の荷物を摩擦だけで把持**し、
3つの障害物を所定の側から回避して皿へ置く、Physical AIの搬送デモを作成します。

この課題では、Gazebo、RViz、MoveIt 2、OMPL、LeRobotをそれぞれ次の目的で活用します。

- **Gazebo**：重力、接触、摩擦を含む物理シミュレーションとデモ実行
- **RViz / MoveIt 2**：ロボット、障害物、把持物を含む経路の可視化と衝突判定
- **OMPL**：安全な教師軌道の生成
- **LeRobot**：カメラ画像・関節状態・操作データから搬送方針を学習・推論するAI
- **ros2_control**：アームとグリッパの制御

提出締切は **2026年11月2日（月）10:00 JST** です。ただし、YouTube処理、ネットワーク障害、PC障害に備え、
**2026年10月29日（木）までに動画・レポート・GitHubリポジトリを完成させる**ことを内部締切とします。

> 注記：以下の最終ミッションは実装計画です。既存のMLP回避デモは、この計画のベースラインとして保持します。

### 最終ミッション

1. SO-101が右側の開始位置にある荷物へ接近する。
2. 手首を90°回転し、両開きグリッパの左右の指で荷物を横から挟む。
3. 三角錐の**外側**、高い直方体の**手前側**、円柱の**外側**を順に通る。
4. 左側の皿の中央へ荷物を静かに置き、グリッパを開く。
5. 1回のミッションを3分以内に完了する。

最終的な経路方針は、次のとおりです。障害物の上を越えるのではなく、すべて水平面内の指定側を通って回避します。

```text
荷物 → 三角錐の外側 → 直方体の手前側 → 円柱の外側 → 皿
```

### 物体仕様

| 対象 | 仕様 |
| --- | --- |
| 運搬物 | プラスチック製直方体、3 × 3 × 4 cm、36 g |
| 円柱障害物 | 直径 8 cm、高さ 10 cm |
| 直方体障害物 | 6 × 6 cm、高さ 20 cm |
| 三角錐障害物 | 底面：1辺 10 cmの正三角形、高さ 12 cm |
| 置き場所 | 皿：直径 6 cm、深さ 1 cm |
| グリッパ | 両開きの2指グリッパ。内側に摩擦係数 0.5 のゴムを装着 |

荷物は姿勢を保って搬送し、30 mm四方の底面を皿の中へ収めます。皿中心からの目標位置誤差は ±5 mm以内とします。

### 物理・安全上の必須条件

- 荷物はGazebo上で常に**動的な剛体**として扱う。
- 保持は、指先のゴムと荷物の接触摩擦だけで実現する。
- 固定ジョイント、強制attach、物体追従、テレポートは使用しない。
- 衝突無効化、障害物や荷物のすり抜け、めり込み、瞬間移動は使用しない。
- 把持、搬送、配置の前に、MoveIt 2でロボット・グリッパ・荷物を含めた衝突判定を行う。
- Gazebo実行中も、荷物の落下、障害物との接触、皿への配置成否を記録する。

### 配置ゆらぎ（ロバスト性の確認）

配置を固定して暗記させないため、各試行の開始時に水平面内の微小なゆらぎを加えます。

- 円柱・直方体・三角錐：基準位置から各軸 ±3 mm
- 荷物・皿：基準位置から各軸 ±2 mm
- 障害物のヨー角：必要に応じて ±2°
- 高さ、寸法、質量、摩擦係数は変えない。
- GazeboとMoveIt Planning Sceneには、同一の乱数シードから得た同じ物体姿勢を設定する。
- 安全余裕や到達可能性を満たさない配置が生成された場合は、実行せずに再抽選する。
- 各試行で乱数シード、実際の物体姿勢、衝突判定結果、成功・失敗、所要時間を保存する。

## 2. 実装・検証計画

### Phase 0：経路と到達可能性の確定

- RViz / MoveIt 2上に皿、荷物、3障害物の基準配置を作成する。
- `PICK`、`PYRAMID_OUTER`、`RECTANGLE_FRONT`、`CYLINDER_OUTER`、`DISH_PLACE` の5つの通過条件を定義する。
- 荷物の衝突形状を含めて、所定の側を通る経路に十分な安全余裕があることを確認する。

### Phase 1：Gazeboの物理的な把持・配置

- 荷物、皿、障害物をGazeboの衝突形状と慣性を持つモデルとして配置する。
- グリッパ単体で「接近 → 閉じる → 持ち上げる → 運ぶ → 開く」を確認する。
- 摩擦把持だけで荷物を持ち上げ、障害物なしで皿へ置けることを合格条件とする。

### Phase 2：OMPL教師軌道とMoveIt安全判定

- OMPL / RRTConnectで、5つの通過条件を満たす区間軌道を生成する。
- 軌道を細かく補間し、全時点でロボット・グリッパ・荷物が障害物や皿と衝突しないことを検証する。
- 生成された安全軌道だけを教師データおよびGazebo実行候補として保存する。

### Phase 3：LeRobotによるAI方針の学習・推論

- GazeboのRGB画像、関節状態、グリッパ操作、教師軌道をLeRobot形式で記録する。
- 配置ゆらぎを含むデータで模倣学習を行う。
- AI推論軌道は、MoveIt 2の衝突判定を通過した場合にのみGazeboへ送る。
- 本番デモでは、安全判定を通過したAI推論軌道を実行する。OMPLは教師データ生成と検証に用い、オンラインでの代替実行には使わない。

### Phase 4：統合評価と提出準備

- 異なるゆらぎ配置で複数回実行し、把持・回避・配置・所要時間を記録する。
- 最終動画は1〜3分以内にまとめ、倍速を使用する場合は動画内または概要欄に明記する。
- 実行コード、README、レポートPDF、動画のYouTube URLを確認する。

### 前倒しスケジュール

| 期日 | 到達目標 |
| --- | --- |
| 10/15 | Gazebo物理モデルと、摩擦のみの把持・障害物なし配置を確認 |
| 10/18 | MoveIt 2 / OMPLで所定経路と荷物込み衝突判定を確認 |
| 10/22 | Gazeboで3障害物回避・皿への配置を物理的に成功させる |
| 10/26 | LeRobotのデータ収集・学習・推論を統合し、複数の配置ゆらぎで評価 |
| 10/27 | 撮影用デモ、コード、READMEを凍結してGitHubへ反映 |
| 10/29 | 動画をYouTubeへアップロードし、レポートPDFと提出情報を完成 |
| 10/30–11/1 | PC・ネットワーク・動画処理などの事故に備える予備期間 |
| 11/2 10:00 | 提出締切。新規変更をせず、提出済みの状態で迎える |

## 3. 既存MLP障害物回避ベースライン


### ベースラインの生成・検証手順

#### Step 1. SO-101 / ROS 2 / Gazebo / MoveIt 2の動作確認

以下を確認します。

- SO-101がGazeboに表示される
- Joint Stateが取得できる
- arm_controllerがactive
- MoveIt 2からSO-101を操作できる
- /check_state_validityが利用できる
- /apply_planning_sceneが利用できる


#### Step 2. IKデータ生成

ik_dataset_generator.py を使用し、
手先位置からSO-101の関節角度をIKで求めます。


#### Step 3. 障害物回避教師データ生成

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


#### Step 4. 学習用データセット作成

prepare_dataset_v5.py を使用します。

可変長軌道を各Episode 30 stepに補間します。

Train : 240 episodes
Val   : 30 episodes
Test  : 30 episodes


#### Step 5. MLP学習

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


#### Step 6. MLP推論

infer_mlp.py を使用します。

Testデータでの結果：

Overall MAE  : 約4.83 deg
Overall RMSE : 約7.61 deg


#### Step 7. 推論軌道の安全性確認

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


#### Step 8. Gazeboデモ

run_mlp_gazebo_demo.py を使用します。

本番実行時にはOMPLによるオンライン再計画を行わず、
MLP推論軌道をそのまま使用します。

PRE-FLIGHT：

python3 -u run_mlp_gazebo_demo.py --episode 4

Episode 4を実行：

python3 -u run_mlp_gazebo_demo.py --execute --episode 4

3条件を連続実行：

python3 -u run_mlp_gazebo_demo.py --execute


## 4. 環境設定

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
