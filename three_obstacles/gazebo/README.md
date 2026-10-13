# 独立した Gazebo 物理シーン

このディレクトリには、`3_obstacles` 専用に新規作成した SDF アセットだけを置きます。
別プロジェクトのモデル、スクリプト、データ、学習済みモデルをコピー・参照していません。
共有の `~/so101_ws` も変更しません。

## モデル仕様

| モデル | 物理仕様 | 種別 |
| --- | --- | --- |
| `payload` | 30 × 30 × 40 mm、0.036 kg | 動的剛体 |
| `dish` | 外径 60 mm、内側深さ 10 mm | 静的衝突体 |
| `cylinder` | 直径 80 mm、高さ 100 mm | 静的衝突体 |
| `tall_rectangular_prism` | 60 × 60 × 200 mm | 静的衝突体 |
| `triangular_pyramid` | 正三角形の底辺 100 mm、高さ 120 mm | 静的衝突体 |

荷物の慣性は直方体の解析式から算出しています。荷物・皿・障害物の各接触面には
`mu = mu2 = 0.5` を設定しています。荷物は `<static>false</static>`、重力有効の
独立した剛体です。固定ジョイント、attach、追従、テレポート、衝突無効化は含みません。

SO-101 の指先パッドはまだこのシーンに結合していません。将来プロジェクト所有の
ロボット記述を作る際には、左右パッドの衝突形状を実装し、同じ摩擦係数 `0.5` を設定します。
この段階では共有ワークスペースのロボット記述を変更しないため、荷物をロボットに
固定する機構は一切ありません。

## 基準配置

`worlds/three_obstacles_baseline.world` は、`../config/mission_spec.json` の
ジッタなし基準値を使います。荷物から皿へは、三角錐の外側、直方体の手前、円柱の外側を
通る候補です。実行時には `config/baseline_spawn_poses.json` を基に、既存の候補生成器が
出力する同じ乱数シードの姿勢へ置き換えます。

### 乱数シーンをテレポートなしで起動する

`generate_episode_world.py` は候補JSONLから1シーン専用のSDF worldを**Gazebo起動前**に
生成します。実行中の `/set_pose`、モデル削除・再生成、追従、固定接続は使いません。
各エピソードは生成済みworldを新規に起動して初期姿勢を与えるため、数mmのゆらぎと
物理シミュレーションを混同しません。

```bash
cd /home/ishihara/pai_obstacle
python3 -m three_obstacles.gazebo.generate_episode_world \
  --requests three_obstacles/data/teacher_requests/candidate_300_seed20261013.jsonl \
  --episode-id scene_00000 \
  --output three_obstacles/gazebo/generated_worlds/scene_00000.world
```

出力先は `gazebo/generated_worlds/` に限定しています。生成されたworldはSDFとして検査して
から起動し、イベント途中に姿勢を変える用途には使いません。

ロボットなしでシーンだけを目視する場合は、プロジェクトルートから次を使います。

```bash
export GZ_SIM_RESOURCE_PATH="$PWD/three_obstacles/gazebo/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
gz sim three_obstacles/gazebo/worlds/three_obstacles_baseline.world
```

これは物理シーンの表示だけで、SO-101 を動かしません。実際のロボット統合は、
MoveIt の到達可能性・衝突判定を通してから、別のプロジェクト所有 launch に実装します。

## 静的検証

```bash
cd /home/ishihara/pai_obstacle
python3 -m unittest discover -s three_obstacles/gazebo/tests -v
```

この検証は XML 構文、寸法、荷重、慣性、摩擦、world の基準姿勢、および禁止された
接続機構が SDF 内にないことだけを確認します。Gazebo・ロボット・MoveIt は起動しません。
