# ROS 2統合（3_obstacles専用）

`three_obstacles_bringup` はこのプロジェクト専用のlaunchパッケージです。
既存の `~/so101_ws` を変更せず、別プロジェクトの拡張を含まない
`~/three_obstacles_ws` 上で使います。

```bash
source /opt/ros/jazzy/setup.bash
source ~/three_obstacles_ws/install/setup.bash
export THREE_OBSTACLES_PROJECT_ROOT=/home/ishihara/pai_obstacle
ros2 launch three_obstacles_bringup three_obstacles_gazebo_moveit.launch.py
```

このlaunchは、プロジェクトの物理world、SO-101、コントローラ、MoveItを起動しますが、
アームやグリッパへの目標軌道を送信しません。MoveItは教師軌道の安全判定・計画だけに使い、
最終デモ時にオンラインOMPL再計画はしません。

最初の動作確認では、headlessで起動して以下だけを確認します。

- 5物体とSO-101がGazebo内に存在する
- `arm_controller` と `gripper_controller` がactive
- `/check_state_validity` と `/apply_planning_scene` が利用できる

## MoveItの衝突形状

Gazeboは接触計算用の簡略形状を使いますが、MoveItは既定で上流の詳細メッシュ衝突形状を
使います（`moveit_use_gazebo_geometry:=false`）。これは粗いGazebo用boxによる偽の自己衝突を
避けるための**形状精度の選択**であり、衝突ペアの無効化やAllowed Collision Matrixの変更では
ありません。詳細メッシュでも実際に接触する姿勢は、MoveItの衝突判定で不合格になります。

比較・診断時にだけ、明示的に粗いGazebo形状へ切り替えられます。

```bash
ros2 launch three_obstacles_bringup three_obstacles_gazebo_moveit.launch.py \
  moveit_use_gazebo_geometry:=true
```
