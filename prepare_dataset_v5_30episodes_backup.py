#!/usr/bin/env python3

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


# ============================================================
# 設定
# ============================================================

INPUT_FILE = (
    Path.home()
    / "so101_ws"
    / "obstacle_dataset_v4_2.csv"
)

OUTPUT_ALL = (
    Path.home()
    / "so101_ws"
    / "dataset_v5_all.csv"
)

OUTPUT_TRAIN = (
    Path.home()
    / "so101_ws"
    / "dataset_v5_train.csv"
)

OUTPUT_VAL = (
    Path.home()
    / "so101_ws"
    / "dataset_v5_val.csv"
)

OUTPUT_TEST = (
    Path.home()
    / "so101_ws"
    / "dataset_v5_test.csv"
)

SUMMARY_FILE = (
    Path.home()
    / "so101_ws"
    / "dataset_v5_split_summary.csv"
)


# 1エピソードを30ステップへ補間
STEPS_PER_EPISODE = 30

# 分割時の乱数固定
RANDOM_SEED = 42

# 各クラス10エピソードを
# 7 train / 1 validation / 2 test にする
TRAIN_PER_CLASS = 7
VAL_PER_CLASS = 1
TEST_PER_CLASS = 2


JOINT_COLUMNS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

CONTEXT_COLUMNS = [
    "obs_x",
    "obs_y",
    "obs_z",
    "obs_size_x",
    "obs_size_y",
    "obs_size_z",

    "start_x",
    "start_y",
    "start_z",

    "target_x",
    "target_y",
    "target_z",

    "baseline_path_length",
    "avoidance_path_length",
    "path_delta",
    "path_rms_difference",
    "baseline_blocked_points",
]


# ============================================================
# CSV読み込み
# ============================================================

def load_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    rows = []

    with path.open(
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(
            "Input CSV is empty."
        )

    return rows


# ============================================================
# episode_idごとにグループ化
# ============================================================

def group_by_episode(rows):

    episodes = defaultdict(list)

    for row in rows:

        episode_id = int(
            row["episode_id"]
        )

        episodes[
            episode_id
        ].append(row)

    # step順に並べる
    for episode_id in episodes:

        episodes[
            episode_id
        ].sort(
            key=lambda r: int(
                r["step"]
            )
        )

    return episodes


# ============================================================
# 線形補間
# ============================================================

def lerp(a, b, alpha):

    return (
        a
        + alpha * (b - a)
    )


# ============================================================
# 1エピソードを30ステップに補間
#
# progress:
#   0.0 = 開始
#   1.0 = 終了
# ============================================================

def interpolate_episode(
    episode_id,
    rows,
    num_steps,
):

    if len(rows) < 2:
        raise RuntimeError(
            f"Episode {episode_id} "
            f"has fewer than 2 points."
        )

    obstacle_class = (
        rows[0]["obstacle_class"]
    )

    # 元軌道の各関節
    source_joints = []

    for row in rows:

        source_joints.append(
            {
                joint: float(
                    row[joint]
                )
                for joint in JOINT_COLUMNS
            }
        )

    source_count = len(
        source_joints
    )

    output_rows = []

    for new_step in range(
        num_steps
    ):

        # 0.0 ～ 1.0
        progress = (
            new_step
            / float(
                num_steps - 1
            )
        )

        # 元データ上の位置
        source_position = (
            progress
            * (
                source_count - 1
            )
        )

        left_index = int(
            math.floor(
                source_position
            )
        )

        right_index = min(
            left_index + 1,
            source_count - 1,
        )

        alpha = (
            source_position
            - left_index
        )

        interpolated_joints = {}

        for joint in JOINT_COLUMNS:

            q0 = (
                source_joints[
                    left_index
                ][joint]
            )

            q1 = (
                source_joints[
                    right_index
                ][joint]
            )

            interpolated_joints[
                joint
            ] = lerp(
                q0,
                q1,
                alpha,
            )

        # エピソード内で固定の情報は
        # 最初の行から引き継ぐ
        first = rows[0]

        output = {
            "episode_id":
                episode_id,

            "obstacle_class":
                obstacle_class,

            "step":
                new_step,

            "progress":
                progress,

            "source_points":
                source_count,
        }

        for column in CONTEXT_COLUMNS:

            output[column] = float(
                first[column]
            )

        for joint in JOINT_COLUMNS:

            output[joint] = (
                interpolated_joints[
                    joint
                ]
            )

        output_rows.append(
            output
        )

    return output_rows


# ============================================================
# 全エピソードを補間
# ============================================================

def interpolate_all(
    episodes,
):

    result = []

    for episode_id in sorted(
        episodes.keys()
    ):

        interpolated = (
            interpolate_episode(
                episode_id,
                episodes[
                    episode_id
                ],
                STEPS_PER_EPISODE,
            )
        )

        result.extend(
            interpolated
        )

    return result


# ============================================================
# クラスごとにepisode IDを取得
# ============================================================

def get_episode_classes(
    episodes,
):

    class_to_ids = defaultdict(
        list
    )

    for episode_id, rows in (
        episodes.items()
    ):

        obstacle_class = (
            rows[0][
                "obstacle_class"
            ]
        )

        class_to_ids[
            obstacle_class
        ].append(
            episode_id
        )

    return class_to_ids


# ============================================================
# episode単位で
# train / validation / test 分割
# ============================================================

def split_episodes(
    class_to_ids,
):

    rng = random.Random(
        RANDOM_SEED
    )

    train_ids = []
    val_ids = []
    test_ids = []

    split_summary = []

    for obstacle_class in sorted(
        class_to_ids.keys()
    ):

        ids = list(
            class_to_ids[
                obstacle_class
            ]
        )

        rng.shuffle(ids)

        required = (
            TRAIN_PER_CLASS
            + VAL_PER_CLASS
            + TEST_PER_CLASS
        )

        if len(ids) < required:

            raise RuntimeError(
                f"Class "
                f"{obstacle_class} "
                f"has only "
                f"{len(ids)} "
                f"episodes; "
                f"{required} required."
            )

        train = ids[
            :
            TRAIN_PER_CLASS
        ]

        val_start = (
            TRAIN_PER_CLASS
        )

        val_end = (
            val_start
            + VAL_PER_CLASS
        )

        val = ids[
            val_start:
            val_end
        ]

        test_start = (
            val_end
        )

        test_end = (
            test_start
            + TEST_PER_CLASS
        )

        test = ids[
            test_start:
            test_end
        ]

        train_ids.extend(
            train
        )

        val_ids.extend(
            val
        )

        test_ids.extend(
            test
        )

        for episode_id in train:

            split_summary.append(
                {
                    "episode_id":
                        episode_id,
                    "obstacle_class":
                        obstacle_class,
                    "split":
                        "train",
                }
            )

        for episode_id in val:

            split_summary.append(
                {
                    "episode_id":
                        episode_id,
                    "obstacle_class":
                        obstacle_class,
                    "split":
                        "validation",
                }
            )

        for episode_id in test:

            split_summary.append(
                {
                    "episode_id":
                        episode_id,
                    "obstacle_class":
                        obstacle_class,
                    "split":
                        "test",
                }
            )

    return (
        set(train_ids),
        set(val_ids),
        set(test_ids),
        split_summary,
    )


# ============================================================
# 指定episodeだけ抽出
# ============================================================

def filter_rows(
    rows,
    episode_ids,
):

    return [
        row
        for row in rows
        if int(
            row["episode_id"]
        )
        in episode_ids
    ]


# ============================================================
# CSV保存
# ============================================================

def save_dataset(
    path,
    rows,
):

    fieldnames = [
        "episode_id",
        "obstacle_class",
        "step",
        "progress",
        "source_points",

        "obs_x",
        "obs_y",
        "obs_z",

        "obs_size_x",
        "obs_size_y",
        "obs_size_z",

        "start_x",
        "start_y",
        "start_z",

        "target_x",
        "target_y",
        "target_z",

        "baseline_path_length",
        "avoidance_path_length",
        "path_delta",

        "path_rms_difference",
        "baseline_blocked_points",

        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Split summary保存
# ============================================================

def save_split_summary(
    path,
    rows,
):

    rows = sorted(
        rows,
        key=lambda r: (
            r[
                "obstacle_class"
            ],
            r[
                "split"
            ],
            r[
                "episode_id"
            ],
        ),
    )

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_id",
                "obstacle_class",
                "split",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# 検証
# ============================================================

def validate_dataset(
    rows,
    expected_episodes,
):

    episode_counts = defaultdict(
        int
    )

    for row in rows:

        episode_counts[
            int(
                row["episode_id"]
            )
        ] += 1

    if len(
        episode_counts
    ) != expected_episodes:

        raise RuntimeError(
            "Unexpected number "
            "of episodes."
        )

    for episode_id, count in (
        episode_counts.items()
    ):

        if count != (
            STEPS_PER_EPISODE
        ):

            raise RuntimeError(
                f"Episode "
                f"{episode_id}: "
                f"{count} rows, "
                f"expected "
                f"{STEPS_PER_EPISODE}."
            )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "================================"
    )

    print(
        "SO-101 dataset preparation V5"
    )

    print(
        "================================"
    )

    print(
        f"Input: "
        f"{INPUT_FILE}"
    )

    # --------------------------------------------------------
    # 読み込み
    # --------------------------------------------------------

    rows = load_csv(
        INPUT_FILE
    )

    print(
        f"Original rows: "
        f"{len(rows)}"
    )

    # --------------------------------------------------------
    # episode化
    # --------------------------------------------------------

    episodes = group_by_episode(
        rows
    )

    print(
        f"Episodes: "
        f"{len(episodes)}"
    )

    # --------------------------------------------------------
    # クラス確認
    # --------------------------------------------------------

    class_to_ids = (
        get_episode_classes(
            episodes
        )
    )

    print(
        "Original episode counts:"
    )

    for class_name in sorted(
        class_to_ids.keys()
    ):

        print(
            f"  {class_name}: "
            f"{len(class_to_ids[class_name])}"
        )

    # --------------------------------------------------------
    # 30ステップへ補間
    # --------------------------------------------------------

    all_rows = (
        interpolate_all(
            episodes
        )
    )

    validate_dataset(
        all_rows,
        expected_episodes=30,
    )

    print(
        f"Interpolated rows: "
        f"{len(all_rows)}"
    )

    # --------------------------------------------------------
    # episode単位でsplit
    # --------------------------------------------------------

    (
        train_ids,
        val_ids,
        test_ids,
        split_summary,
    ) = split_episodes(
        class_to_ids
    )

    train_rows = filter_rows(
        all_rows,
        train_ids,
    )

    val_rows = filter_rows(
        all_rows,
        val_ids,
    )

    test_rows = filter_rows(
        all_rows,
        test_ids,
    )

    # --------------------------------------------------------
    # validation
    # --------------------------------------------------------

    validate_dataset(
        train_rows,
        expected_episodes=21,
    )

    validate_dataset(
        val_rows,
        expected_episodes=3,
    )

    validate_dataset(
        test_rows,
        expected_episodes=6,
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    save_dataset(
        OUTPUT_ALL,
        all_rows,
    )

    save_dataset(
        OUTPUT_TRAIN,
        train_rows,
    )

    save_dataset(
        OUTPUT_VAL,
        val_rows,
    )

    save_dataset(
        OUTPUT_TEST,
        test_rows,
    )

    save_split_summary(
        SUMMARY_FILE,
        split_summary,
    )

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print()
    print(
        "===== V5 SUMMARY ====="
    )

    print(
        f"All:"
        f"   {len(all_rows)} rows"
    )

    print(
        f"Train:"
        f" {len(train_rows)} rows "
        f"({len(train_ids)} episodes)"
    )

    print(
        f"Val:"
        f"   {len(val_rows)} rows "
        f"({len(val_ids)} episodes)"
    )

    print(
        f"Test:"
        f"  {len(test_rows)} rows "
        f"({len(test_ids)} episodes)"
    )

    print()

    print(
        f"All dataset:"
        f" {OUTPUT_ALL}"
    )

    print(
        f"Train:"
        f" {OUTPUT_TRAIN}"
    )

    print(
        f"Validation:"
        f" {OUTPUT_VAL}"
    )

    print(
        f"Test:"
        f" {OUTPUT_TEST}"
    )

    print(
        f"Split summary:"
        f" {SUMMARY_FILE}"
    )

    print()
    print(
        "Dataset preparation completed."
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)
