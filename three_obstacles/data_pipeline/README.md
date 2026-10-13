# 物理リプレイの生データ記録

この独立パッケージは、Gazebo 上の実行を「学習用の証拠付き生データ」として保存するためのものです。既存のMLPや別プロジェクトは読み込まず、LeRobotを import しません。

## 記録するもの

- シーン乱数シード、ミッション設定／ロボットモデル／ワールド／教師軌道の SHA-256
- UTC 時刻とシミュレーション時刻、カメラRGBの相対参照
- 関節位置・速度、両指グリッパ状態、教師の関節／グリッパ目標
- 荷物姿勢、接触相手・法線力・接線速度
- すべり・瞬間移動・めり込み・禁止衝突の検査
- MoveIt安全性、Gazebo衝突、皿への配置・開放・静止時間、実行時間

`accepted` になるには、荷物が動的剛体で、摩擦保持のみ（パッド摩擦係数0.5）であること、衝突・配置・完全性検査がすべて通ることが必要です。固定ジョイント、強制 attach、追従、テレポート、衝突無効化を含む記録は拒否されます。

```text
episode_000001/
├── metadata.json  # シード、ハッシュ、座標系、物理契約、しきい値
├── frames.jsonl   # 時系列観測、教師操作、接触、各検査
├── result.json    # 衝突・配置・完全性・実行時間
└── manifest.json  # 上記3ファイルのSHA-256
```

## 使用方法

```python
from pathlib import Path
from three_obstacles.data_pipeline.writer import PhysicalReplayEpisodeWriter

writer = PhysicalReplayEpisodeWriter(Path("data/raw/episode_000001"), metadata)
writer.append_frame(frame)
writer.finalize(summary)
```

検証はロボットを動かさずに行えます。

```bash
cd /home/ishihara/pai_obstacle
python3 -m three_obstacles.data_pipeline.validator data/raw/episode_000001
python3 -m unittest discover -s three_obstacles/data_pipeline/tests -t . -v
```

`lerobot_handoff.py` は、検証済みの生エピソードを将来の専用LeRobot環境へ渡すための列順・RGB参照の契約だけを作ります。画像や生データをコピーせず、LeRobotデータセットを生成しません。

## データセット準備度の確認

`dataset_readiness.py` は候補シーン、将来のOMPL軌道記録、物理リプレイを分けて数えます。候補シーンだけを「学習データ」と扱わず、LeRobot変換可は検証済みかつ `accepted` の物理リプレイだけです。

```bash
python3 -m three_obstacles.data_pipeline.dataset_readiness \
  --candidate-requests three_obstacles/data/teacher_requests/smoke_20261013.jsonl
```

出力には、後続のOMPL記録器に必要な関節名・時間付き軌道点・速度・安全性・4ゲート順序の必須フィールドも含まれます。軌道や物理リプレイを補完・生成する機能はありません。
