# `three_obstacles_description`

上流SO-101記述を**読み取り専用**で受け取り、3障害物搬送課題専用の二指グリッパURDFを
別ファイルとして生成する小さなROS 2パッケージです。

このパッケージは上流の `so101_description`、`so101_moveit_config`、
`three_obstacles_ws` を編集しません。別プロジェクトのコード、モデル、学習データも
読み込みません。

## 生成する内容

- 上流の `moving_jaw_so101_v1_link` を右指として維持
- 新規 `three_obstacles_left_finger_link` と `gripper_left` 回転関節
- 左右の通常の box collision パッド（5 × 20 × 30 mm）
- 左右指リンクの Gazebo接触係数 `mu1 = mu2 = 0.5`
- `ros2_control` に対する `gripper_left` の position command/state interface
- 二関節の `dual_finger_controller` 用YAMLフラグメント
- 生成条件・摩擦・禁止機構を明記したJSON model spec

`mimic`、荷物の固定関節、強制attach、物体追従、テレポート、衝突無効化は生成しません。
荷物は常に別の動的剛体であり、把持は衝突と摩擦だけに依存します。

## 生成例

テスト用に、展開済みURDFを入力する場合です。

```bash
cd /home/ishihara/pai_obstacle/three_obstacles/ros2/three_obstacles_description
python3 -m three_obstacles_description.overlay_generator \
  --input-urdf /absolute/path/to/so101_expanded.urdf \
  --output-urdf /tmp/three_obstacles_so101.urdf \
  --output-model-spec /tmp/three_obstacles_so101_model_spec.json \
  --output-controller-fragment /tmp/three_obstacles_dual_finger_controller.yaml
```

上流xacroを実行時に展開する場合は、ROS 2 と**独立した**
`/home/ishihara/three_obstacles_ws` をsourceしてから、次のように使います。

```bash
source /opt/ros/jazzy/setup.bash
source /home/ishihara/three_obstacles_ws/install/setup.bash
cd /home/ishihara/pai_obstacle/three_obstacles/ros2/three_obstacles_description
python3 -m three_obstacles_description.overlay_generator \
  --upstream-xacro /home/ishihara/three_obstacles_ws/src/so101_ros2/so101_description/urdf/robots/so101.urdf.xacro \
  --xacro-arg use_gazebo:=true \
  --xacro-arg use_camera:=false \
  --output-urdf /tmp/three_obstacles_so101.urdf \
  --output-model-spec /tmp/three_obstacles_so101_model_spec.json \
  --output-controller-fragment /tmp/three_obstacles_dual_finger_controller.yaml
```

出力先はプロジェクト所有ディレクトリか `/tmp` を指定します。上流入力ファイルの隣へ
出力しないでください。

## キャリブレーション上の注意

初期の左関節・パッド姿勢は、上流右顎の局所座標を基にした**候補値**です。物理実行の前に、
生成URDFをRVizで表示し、ロボットを動かさずに次を確認します。

1. 左右パッドが荷物の反対面へ向いていること
2. 両方の指が荷物・互い・手首へ初期めり込みしないこと
3. 30 mm幅の荷物を挟める開口範囲があること

不適切なら、生成コマンドの `--left-joint-xyz`、`--left-joint-rpy`、
`--right-pad-xyz`、`--left-pad-xyz` を調整し、その値と検証結果を記録します。
この候補値のままGazeboで把持を実行してはいけません。

## テスト

```bash
cd /home/ishihara/pai_obstacle/three_obstacles/ros2/three_obstacles_description
python3 -B -m unittest discover -s tests -v
```

テストはXML変換・指関節・パッド寸法・摩擦係数・ros2_control拡張・禁止機構のみを検査し、
Gazebo、MoveIt、ロボットを起動しません。
