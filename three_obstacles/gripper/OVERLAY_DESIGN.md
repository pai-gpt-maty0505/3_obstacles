# 二指グリッパ用のプロジェクト所有オーバーレイ設計

## 調査範囲

調査したのは、未変更の上流ワークスペース
`/home/ishihara/three_obstacles_ws/src/so101_ros2` だけです。
そのGit remote は `https://github.com/adoodevv/so101_ros2.git` であり、
`git status --short` は空でした。別プロジェクト、共有 `~/so101_ws` は
読まず、変更していません。

## 上流の現状

上流SO-101は二指独立駆動ではありません。

- `so101_description/urdf/so101_robot.urdf.xacro`
  - `moving_jaw_so101_v1_link` が唯一の可動顎（341行目）
  - `gripper` は `gripper_link` からこの顎への唯一の回転関節（369–375行目）
  - Gazebo時の顎衝突形状は 25 × 35 × 45 mm box（355–365行目）
- `so101_description/urdf/control/so101_ros2_control.urdf.xacro`
  - `ros2_control` に公開されるグリッパ関節も `gripper` 一つだけ（52–60行目）
- `so101_moveit_config/config/so101/ros2_controllers.yaml`
  - `gripper_controller` は `gripper` 一軸のみ（53–62行目）
- `so101_moveit_config/config/so101/so101.srdf`
  - MoveIt の `gripper` group も `gripper` 一関節だけ（13–15行目）

したがって、単に新しい指のリンクを追加するだけでは不十分です。URDF、
ros2_control、コントローラ設定、MoveIt group を同じモデル定義として更新する
必要があります。

## 推奨する最小オーバーレイ

上流ファイルをコピー・編集せず、プロジェクト側に次の小さな変換器と設定を置きます。

```text
上流 so101.urdf.xacro (read only)
  -> xacro 展開
  -> three_obstacles の XML 変換器
  -> 生成済み three_obstacles_so101.urdf
  -> プロジェクト所有 launch / controller YAML / MoveIt semantic config
```

将来作成する変換器は、展開済みURDFに次だけを追加します。

1. 既存の `moving_jaw_so101_v1_link` を右指としてそのまま残す。
   このリンクへ、プロジェクト所有の `right_rubber_pad_collision` を直接追加する。
2. 新規の `gripper_left_link` と、独立した回転関節 `gripper_left` を追加する。
   新しいリンクには、指本体と `left_rubber_pad_collision` を直接持たせる。
3. 既存の `ros2_control` 要素内へ `gripper_left` の position command/state interface を追加する。
4. 既存の `gripper_controller` を二関節の `JointTrajectoryController` に置き換える。
   指令順序は常に `gripper`, `gripper_left` とし、対称把持は上位ノードが二つの
   目標値を明示して送る。URDF の `mimic` は使わないので、二指は本当に独立している。
5. MoveIt の `gripper` group、open/closed state、controller mapping を二関節へ更新する。
6. 各パッドの Gazebo 衝突 surface に `mu=mu2=0.5` を与える。
   荷物にも既に `three_obstacles/gazebo/models/payload/model.sdf` で同じ値がある。

パッドは各指リンク内の通常の collision として表現します。荷物との結合、荷物への
固定関節、強制 attach、追従、テレポート、衝突無効化は一切追加しません。

## 寸法・姿勢を確定する方法

上流の右指関節の数値を機械的に反転して左指へ流用してはいけません。最初に
ロボットを動かさないRViz表示で、右指の局所座標、荷物中央、反対側に必要な左指の
回転中心を測定します。その結果をプロジェクト所有の定数として記録します。

初期パッド候補は、荷物の40 mm高さを十分覆うように **5 mm厚 × 20 mm幅 × 30 mm高**
とします。これは把持面の候補値であり、指間距離・可動域・接触安定性の測定後にのみ
確定します。ゴム厚、摩擦、押付け力を変える場合も、変更理由と物理再生結果を保存します。

## 実装順序

1. `three_obstacles` 側に変換器、二関節コントローラYAML、SRDF変換器、専用launchを作る。
2. xacro・URDF・controller設定を静的に検証し、両パッドの衝突面と `mu=0.5` を出力検査する。
3. RVizで可動域を確認して、左指の回転中心とパッド位置を確定する。この段階ではロボットを動かさない。
4. Gazeboで荷物を床から把持・持上げ・静止させる単体物理試験を行う。把持成功は荷物の接触と摩擦だけで判定する。
5. 合格したグリッパ構成だけをMoveItの把持物込み衝突判定、OMPL教師軌道、LeRobotデータ収集へ進める。

## なぜ変換器か

上流xacroには、単一顎を差し替えるための拡張フックがありません。部分的な
`xacro:include` だけでは、既存 `ros2_control` の中へ二本目の関節を安全に入れられません。
展開後URDFの限定的なXML変換なら、上流を読み取り専用に保ちながら、追加差分を
プロジェクトコードとして明示・テストできます。
