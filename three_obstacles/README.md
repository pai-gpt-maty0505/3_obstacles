# 3障害物搬送：独立した教師データ生成パイプライン

このディレクトリは、最終課題用に新規作成するコードだけを置く場所です。
既存の MLP デモや別プロジェクトのスクリプト、データセット、学習済みモデルは
読み込まず、コピーもしません。

## 役割

最初に作るのは学習済みモデルではなく、**再現可能な教師データ要求**です。
各要求は、同一乱数シードから次を一意に決めます。

- 円柱・直方体・三角錐、荷物、皿の Gazebo / MoveIt 共通の姿勢
- 指定された回避順序と通過側
- OMPL が満たすべき経路ゲート
- データ分割（train / val / test）

その後に、独立して実装する OMPL 教師ノードが `PENDING_OMPL` の要求を読み、
安全な関節軌道、補間済みの状態・操作列、MoveIt の衝突判定結果を追加します。
Gazebo での物理実行と LeRobot 形式への変換は、安全な教師軌道だけに対して行います。

```text
scene sampler
  -> teacher request JSONL
  -> OMPL + MoveIt safety validation
  -> Gazebo physical replay (dynamic payload / friction only)
  -> LeRobot episode conversion
  -> training / held-out evaluation
```

`generate_teacher_requests.py` は ROS、Gazebo、既存の回避スクリプトを使いません。
そのため、最初に生成対象・乱数・経路制約を検証できますが、出力時点では
まだ学習データや実行可能軌道ではありません。

## 座標と経路の意味

`mission_spec.json` の座標系は SO-101 の `base_link` を想定し、`+x` をロボット正面、
`+y` をロボット左、`+z` を上とします。基準配置は、右（`-y`）の荷物から左（`+y`）の皿へ向かいます。

経路ゲートは次の順番です。

1. `PYRAMID_OUTER` — 三角錐のロボット右側（外側）
2. `RECTANGLE_FRONT` — 高い直方体のロボット側（手前側）
3. `CYLINDER_OUTER` — 円柱のロボット左側（外側）
4. `DISH_PLACE` — 皿の中心上方

これらは最初の MoveIt / RViz 到達可能性テストで確定する**候補配置**です。
OMPL が経路を見つけても、荷物を含めた衝突判定・Gazebo の物理再生を通らなければ教師データに採用しません。

## 最初のスモーク試験

以下はロボットを動かさず、80個の候補配置と経路ゲートを生成・検査します。

```bash
cd /home/ishihara/pai_obstacle
python3 three_obstacles/tools/generate_teacher_requests.py \
  --count 80 \
  --seed 20261013 \
  --output three_obstacles/data/teacher_requests/smoke_20261013.jsonl

python3 three_obstacles/tools/validate_teacher_requests.py \
  three_obstacles/data/teacher_requests/smoke_20261013.jsonl
```

## データ作成の段階目標

| 段階 | 件数 | 目的 | 次へ進む条件 |
| --- | ---: | --- | --- |
| smoke | 80候補 | スキーマ・乱数・幾何制約の確認 | 全候補が静的な配置／経路ゲート検査を通る |
| pilot | 240教師軌道 | OMPL成功率・平均生成時間の実測 | MoveIt安全判定・Gazebo物理再生の合格率を把握する |
| training | 1,200教師軌道 | LeRobot学習用の主データ | 900 / 150 / 150 の scene-level split を固定する |
| held-out replay | 60シーン以上 | 未使用シードでの物理評価 | 把持・回避・配置を記録して比較する |

大規模生成を始める前に、pilot の実測速度から所要時間を見積もります。
失敗・衝突・物理再生不合格の要求は、理由を残して除外し、数だけを水増ししません。

## 非許容事項

- 固定ジョイント、強制 attach、物体追従、テレポート
- 衝突無効化、めり込み、すり抜け
- Gazebo 上の物体を MoveIt の Attached CollisionObject で物理的に固定したものとして扱うこと

MoveIt 側の Attached CollisionObject は、必要になった場合でも**計画上の衝突表現**にのみ使います。
Gazebo の荷物は常に重力・接触・摩擦を受ける動的な剛体です。
