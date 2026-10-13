# MoveIt / OMPL 教師シーン準備（独立コード）

このディレクトリは `three_obstacles` 専用の新規コードです。既存プロジェクトの
スクリプト、モデル、データセットを読み込んだりコピーしたりしません。

## できること

教師要求 JSONL から、決定論的に以下を含むポータブルな MoveIt PlanningScene 文書を作ります。

- 円柱、直方体、三角錐の衝突表現
- 初期位置の動的荷物の**計画用**衝突表現
- 浅い皿（底と16分割リム）の保守的な衝突表現
- `PYRAMID_OUTER → RECTANGLE_FRONT → CYLINDER_OUTER → DISH_PLACE` の順序ゲート

三角錐は外部メッシュを使わず、底辺・高さから新たに4頂点・4面のメッシュを作ります。
皿は指定寸法だけから新たに作る計画用近似であり、最終的なGazeboの接触形状とは別に
物理再生で検証します。

## 実行しないこと

- Gazeboの起動、ロボット／コントローラへの指令
- OMPL実行、オンライン再計画
- 固定ジョイント、強制attach、物体追従、テレポート、衝突無効化

Attached CollisionObject はこのプロジェクトでは使いません。Gazeboの荷物は常に
重力・接触・摩擦だけを受ける動的剛体です。

## スモーク確認（ロボットを動かさない）

```bash
cd /home/ishihara/pai_obstacle
python3 -m three_obstacles.teacher.build_scene \
  --requests three_obstacles/data/teacher_requests/smoke_20261013.jsonl \
  --episode-id scene_00000 \
  --print

python3 -m unittest discover -s three_obstacles/teacher/tests -v
```

## ROSアダプター

`ros_adapter.py` はポータブル文書をメモリ内の `moveit_msgs/PlanningScene` に変換する
任意アダプターです。ROSメッセージを生成するだけで、publish、service call、spin、
ロボット実行はしません。ROS環境が無い通常のPythonでは import 時ではなく呼出時に明示的な
エラーになります。

実際の教師作成は、検証済みのシーンをオフラインOMPLワーカーへ渡し、各ゲート間の経路を
衝突判定・補間・Gazebo物理再生まで完了してから採用する段階で行います。最終デモでは
OMPLを呼びません。並列処理時の入出力契約は [WORKER_CONTRACT.md](WORKER_CONTRACT.md) を参照してください。

## MoveIt Planning Scene への安全な反映

`apply_teacher_scene.py` は、選んだ1シーンについて衝突オブジェクトだけを
`/apply_planning_scene` に送るためのCLIです。軌道、Action client、コントローラ、Gazebo、
グリッパ、Attached CollisionObject、Allowed Collision Matrix は一切扱いません。

既定はdry-runです。ROSメッセージをメモリ内で作成して内容を検査しますが、サービス呼出は
行いません。

```bash
cd /home/ishihara/pai_obstacle
source /opt/ros/jazzy/setup.bash
source /home/ishihara/three_obstacles_ws/install/setup.bash

python3 -m three_obstacles.teacher.apply_teacher_scene \
  --requests three_obstacles/data/teacher_requests/smoke_20261013.jsonl \
  --episode-id scene_00000 \
  --dry-run
```

MoveItの `move_group` をユーザー自身が既に起動している場合に限り、明示的な `--apply` を
付けると、上記の5個の世界衝突オブジェクトを `/apply_planning_scene` に送ります。タイムアウトは
既定5秒で、サービスが無ければ失敗して終了します。ロボットはこのCLIから動きません。

```bash
python3 -m three_obstacles.teacher.apply_teacher_scene \
  --requests three_obstacles/data/teacher_requests/smoke_20261013.jsonl \
  --episode-id scene_00000 \
  --apply --service-timeout-s 5
```

`--apply` と `--dry-run` は同時指定できません。`--apply` を付けない限り、ROSサービスを
生成・待機・呼出しません。

## OMPL教師用の計画サービス・スモーク確認

`ompl_plan_only.py` は、MoveItの `/plan_kinematic_path` に対して、5関節の明示的な
関節角目標を使った `GetMotionPlan` 要求だけを作るCLIです。`group_name=arm`、
`pipeline_id=ompl`、`planner_id=RRTConnectkConfigDefault` を固定します。

既定はdry-runで、ROSメッセージをメモリ内で構築してJSONで要求契約を表示するだけです。
サービス待機・呼出しは行いません。

```bash
cd /home/ishihara/pai_obstacle
source /opt/ros/jazzy/setup.bash

python3 -m three_obstacles.teacher.ompl_plan_only \
  --goal-rad 0.0 0.0 0.0 0.0 0.0 \
  --dry-run
```

`move_group` が既に起動済みであることを確認したうえで、明示的に `--plan` を付けた場合だけ
1回だけ `/plan_kinematic_path` を呼びます。待機と呼出しにはいずれも上限があります。返すのは
MoveItエラーコード、計画時間、関節名、点数、時間長、各点の**位置**だけです。速度・加速度・
effort・実行指令は出力も実行もしません。

```bash
python3 -m three_obstacles.teacher.ompl_plan_only \
  --goal-rad 0.0 0.0 0.0 0.0 0.0 \
  --plan --planning-time-s 2 --service-timeout-s 5
```

開始姿勢をMoveItの現在状態に任せず、自己衝突しないready姿勢を明示する必要がある場合は、
`--start-rad` に同じ関節順序の5値を渡せます。この値は
`motion.start_state.joint_state` にだけ入ります。これは**計画開始状態の指定**であり、
ロボットをその姿勢へ動かすコマンドではありません。

```bash
python3 -m three_obstacles.teacher.ompl_plan_only \
  --start-rad 0.0 0.0 0.0 0.0 0.0 \
  --goal-rad 0.1 -0.2 0.3 -0.4 0.5 \
  --dry-run
```

まず `state_validity_probe.py --check` でこの開始姿勢が衝突なしであることを確認します。
自己衝突が返った場合は、衝突無効化ではなく別の開始姿勢を探します。

これは教師データ作成前の「計画サービスが応答するか」を確認する基盤であり、把持・3障害物搬送・
完全な教師軌道生成を主張するものではありません。`--plan` でも軌道を実行せず、Gazebo、
コントローラ、グリッパ、衝突行列、Attached CollisionObjectを変更しません。

## 自己衝突を回避せず確認するプローブ

`state_validity_probe.py` は、5関節の明示的な姿勢について MoveIt の
`/check_state_validity` を**読み取り専用**で問い合わせます。これは衝突を無効化するための
ツールではありません。自己衝突や障害物衝突があれば、そのまま `valid=false` と、MoveItが
返せる場合にはエラーコード・接触したリンク名を報告します。

既定はdry-runです。要求メッセージを作るだけでサービスには接続しません。

```bash
cd /home/ishihara/pai_obstacle
source /opt/ros/jazzy/setup.bash

python3 -m three_obstacles.teacher.state_validity_probe \
  --joint-rad 0.0 0.0 0.0 0.0 0.0 \
  --dry-run
```

`move_group` が起動済みで、衝突を含む現行Planning Sceneを確認したい場合だけ、明示的に
`--check` を付けます。問い合わせは1回・時間上限付きで、ロボット、グリッパ、Gazebo、
Planning Scene、Attached CollisionObject、Allowed Collision Matrixを変更しません。

```bash
python3 -m three_obstacles.teacher.state_validity_probe \
  --joint-rad 0.0 0.0 0.0 0.0 0.0 \
  --check --service-timeout-s 5
```

自己衝突が返った場合は、衝突無効化ルールを追加せず、`valid=true` になる別の開始姿勢／目標姿勢を
探索してから、次のオフライン教師計画へ進みます。
