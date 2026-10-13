# オフライン教師ワーカー契約

`teacher/worker.py` は、教師要求 JSONL の**1シャード**を独立に検査するための
並列化境界です。現時点では準備だけを行い、OMPL、Gazebo、ロボットを実行しません。

```bash
cd /home/ishihara/pai_obstacle
python3 -m three_obstacles.teacher.worker \
  --input-shard three_obstacles/data/teacher_requests/pilot_000.jsonl \
  --result-jsonl three_obstacles/teacher/output/pilot_000.results.jsonl \
  --shard-id pilot-000
```

出力は入力の各レコードに必ず1行対応します。必須キーは次です。

| キー | 意味 |
| --- | --- |
| `episode_id` | 元のシーンID |
| `scene_hash` | 元の配置ハッシュ |
| `status` | 現在は `READY_FOR_OFFLINE_OMPL` または `REJECTED_SCENE_CONTRACT` |
| `planning_duration_s` | OMPL未実行のため現在は `null` |
| `failure_reason` | 不採用時の理由。準備成功時は `null` |

将来の**専用オフライン** OMPL 実行器は、この5キーを保ったまま、成功時に
`SUCCEEDED_OFFLINE_OMPL`、失敗時に `FAILED_OFFLINE_OMPL` を返します。実行器は
MoveItの衝突判定・ゲート順序・軌道補間の検査を終えてから結果を確定します。

この契約は最終デモ用のオンライン再計画インターフェースではありません。最終ミッションは、
検証済みの学習済み／オフライン生成済み軌道を使い、OMPLをオンラインで呼びません。

出力先は安全のため `three_obstacles/teacher/` 以下に限定しています。各シャードは別々の
結果ファイルに出力し、全シャード完了後に集約器が `scene_hash` の重複、欠損、失敗理由を
確認する想定です。
