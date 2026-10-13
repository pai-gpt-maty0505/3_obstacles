# 教師生成の並列実行

このディレクトリは、MoveIt / OMPL の教師生成を安全に並列化するための補助コードです。

各 JSONL レコードは一つのシーン（乱数シード、物体姿勢、経路ゲート）を表すため、
1シーンを複数ワーカーに重複配布しません。`partition_teacher_requests.py` は
シーンを決定的に分割し、後で各ワーカーが独立した結果 JSONL を作れるようにします。

## 推奨並列度

このPCでは、最初は次の構成で実測します。

- Gazebo 物理再生：1プロセス（同時実行しない）
- MoveIt / OMPL 教師生成：2ワーカー
- 学習：物理再生・計画と別の時間帯に GPU を使って実行

同時に Gazebo を複数起動したり、制御器を共有したりしません。OMPLワーカーは
Gazeboを動かさない planning-only 段階だけで並列化します。

```bash
python3 three_obstacles/orchestration/partition_teacher_requests.py \
  three_obstacles/data/teacher_requests/pilot.jsonl \
  --workers 2 \
  --output-dir three_obstacles/data/teacher_request_shards/pilot
```

各ワーカーが結果を返した後、完了・失敗理由・計画時間を集計してから次の件数へ進みます。

## 結果の厳密な集計

`aggregate_ompl_results.py` は、OMPLを実行せずに結果JSONLを監査します。パーティション
manifest の入力シャードを再読込して SHA-256、件数、`episode_id`、`scene_hash` を照合します。

- 全要求に対して結果がちょうど1件あること
- 結果が割り当てられたシャードのものだけであること
- 同一 `episode_id` または `scene_hash` の重複がないこと
- `READY_FOR_OFFLINE_OMPL` などの未終端状態を受け付けないこと
- 成功時は `trajectory_sha256` と計画時間、失敗時は失敗理由を必須とすること

を満たさない限り、集計は `valid: false` で終了コード1になります。入力・結果は一切変更せず、
`--output` を指定しても有効な集計結果だけを新規保存します。

```bash
python3 three_obstacles/orchestration/aggregate_ompl_results.py \
  --partition-manifest three_obstacles/data/teacher_request_shards/pilot/manifest.json \
  --result worker_00=three_obstacles/teacher/output/worker_00.results.jsonl \
  --result worker_01=three_obstacles/teacher/output/worker_01.results.jsonl
```

終端状態は `SUCCEEDED_OFFLINE_OMPL`、`FAILED_OFFLINE_OMPL`、`REJECTED_SCENE_CONTRACT` だけです。
現在の準備用ワーカーが返す `READY_FOR_OFFLINE_OMPL` は、教師軌道ではないため意図的に拒否されます。
