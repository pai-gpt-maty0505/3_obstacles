#!/usr/bin/env python3

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


# ============================================================
# 基本設定
# ============================================================

BASE_DIR = Path.home() / "so101_ws"

INPUT_FILE = (
    BASE_DIR
    / "obstacle_dataset_v4_2.csv"
)

OUTPUT_ALL = (
    BASE_DIR
    / "dataset_v5_all.csv"
)

OUTPUT_TRAIN = (
    BASE_DIR
    / "dataset_v5_train.csv"
)

OUTPUT_VAL = (
    BASE_DIR
    / "dataset_v5_val.csv"
)

OUTPUT_TEST = (
    BASE_DIR
    / "dataset_v5_test.csv"
)

SUMMARY_FILE = (
    BASE_DIR
    / "dataset_v5_split_summary.csv"
)


# ============================================================
# 300エピソード版設定
# ============================================================

EXPECTED_CLASSES = [
    "left",
    "center",
    "right",
]

EXPECTED_EPISODES_PER_CLASS = 100

STEPS_PER_EPISODE = 30

TRAIN_PER_CLASS = 80
VAL_PER_CLASS = 10
TEST_PER_CLASS = 10

RANDOM_SEED = 42


# ============================================================
# 関節列
# ============================================================

JOINT_COLUMNS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


# ============================================================
# エピソード内で固定の情報
# ============================================================

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

    # step順に整列
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
# クラスごとのエピソードID取得
# ============================================================

def get_episode_classes(
    episodes,
):

    class_to_ids = defaultdict(list)

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
# 入力データ検証
# ============================================================

def validate_raw_dataset(
    episodes,
    class_to_ids,
):

    expected_total = (
        len(EXPECTED_CLASSES)
        * EXPECTED_EPISODES_PER_CLASS
    )

    if len(episodes) != expected_total:

        raise RuntimeError(
            f"Expected "
            f"{expected_total} episodes, "
            f"but found "
            f"{len(episodes)}."
        )

    for class_name in EXPECTED_CLASSES:

        count = len(
            class_to_ids[
                class_name
            ]
        )

        if count != (
            EXPECTED_EPISODES_PER_CLASS
        ):

            raise RuntimeError(
                f"Class "
                f"{class_name}: "
                f"expected "
                f"{EXPECTED_EPISODES_PER_CLASS}, "
                f"found {count}."
            )

    unknown_classes = (
        set(
            class_to_ids.keys()
        )
        - set(
            EXPECTED_CLASSES
        )
    )

    if unknown_classes:

        raise RuntimeError(
            f"Unexpected classes: "
            f"{sorted(unknown_classes)}"
        )


# ============================================================
# 線形補間
# ============================================================

def lerp(
    a,
    b,
    alpha,
):

    return (
        a
        + alpha
        * (b - a)
    )


# ============================================================
# 1エピソードを30ステップへ補間
# ============================================================

def interpolate_episode(
    episode_id,
    rows,
    num_steps,
):

    if len(rows) < 2:

        raise RuntimeError(
            f"Episode "
            f"{episode_id} "
            f"has fewer than "
            f"2 trajectory points."
        )

    obstacle_class = (
        rows[0][
            "obstacle_class"
        ]
    )

    source_count = len(rows)

    # 元軌道の5関節角度
    source_joints = []

    for row in rows:

        source_joints.append(
            {
                joint: float(
                    row[joint]
                )
                for joint
                in JOINT_COLUMNS
            }
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

        # 元軌道上の位置
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

        rows = interpolate_episode(
            episode_id,
            episodes[
                episode_id
            ],
            STEPS_PER_EPISODE,
        )

        result.extend(
            rows
        )

    return result


# ============================================================
# Train / Val / Test 分割
#
# 行単位ではなく
# エピソード単位で分割
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

    summary_rows = []

    required = (
        TRAIN_PER_CLASS
        + VAL_PER_CLASS
        + TEST_PER_CLASS
    )

    if required != (
        EXPECTED_EPISODES_PER_CLASS
    ):

        raise RuntimeError(
            "Train/Val/Test split "
            "does not total "
            "100 episodes per class."
        )

    for obstacle_class in (
        EXPECTED_CLASSES
    ):

        ids = list(
            class_to_ids[
                obstacle_class
            ]
        )

        rng.shuffle(ids)

        train = ids[
            :TRAIN_PER_CLASS
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

            summary_rows.append(
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

            summary_rows.append(
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

            summary_rows.append(
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
        summary_rows,
    )


# ============================================================
# 指定エピソードのみ抽出
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
# エピソード単位の整合性確認
# ============================================================

def validate_interpolated_dataset(
    rows,
    expected_episode_count,
):

    episode_counts = defaultdict(
        int
    )

    for row in rows:

        episode_id = int(
            row["episode_id"]
        )

        episode_counts[
            episode_id
        ] += 1

    if len(
        episode_counts
    ) != expected_episode_count:

        raise RuntimeError(
            f"Expected "
            f"{expected_episode_count} "
            f"episodes, but found "
            f"{len(episode_counts)}."
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
# Split重複チェック
# ============================================================

def validate_split_overlap(
    train_ids,
    val_ids,
    test_ids,
):

    if (
        train_ids
        & val_ids
    ):

        raise RuntimeError(
            "Train and validation "
            "episodes overlap."
        )

    if (
        train_ids
        & test_ids
    ):

        raise RuntimeError(
            "Train and test "
            "episodes overlap."
        )

    if (
        val_ids
        & test_ids
    ):

        raise RuntimeError(
            "Validation and test "
            "episodes overlap."
        )


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
            int(
                r[
                    "episode_id"
                ]
            ),
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
# Splitごとのクラス数表示
# ============================================================

def print_split_class_counts(
    summary_rows,
):

    counts = defaultdict(
        lambda: defaultdict(
            int
        )
    )

    for row in summary_rows:

        counts[
            row["split"]
        ][
            row[
                "obstacle_class"
            ]
        ] += 1

    print()

    print(
        "Episode split by class:"
    )

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        parts = []

        for class_name in (
            EXPECTED_CLASSES
        ):

            parts.append(
                f"{class_name}="
                f"{counts[split_name][class_name]}"
            )

        print(
            f"  {split_name:10s}: "
            + ", ".join(
                parts
            )
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "================================"
    )

    print(
        "SO-101 Dataset Preparation V5"
    )

    print(
        "300 Episode Version"
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

    raw_rows = load_csv(
        INPUT_FILE
    )

    print(
        f"Original trajectory rows: "
        f"{len(raw_rows)}"
    )

    # --------------------------------------------------------
    # エピソード化
    # --------------------------------------------------------

    episodes = group_by_episode(
        raw_rows
    )

    print(
        f"Original episodes: "
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

    for class_name in (
        EXPECTED_CLASSES
    ):

        print(
            f"  {class_name}: "
            f"{len(class_to_ids[class_name])}"
        )

    validate_raw_dataset(
        episodes,
        class_to_ids,
    )

    # --------------------------------------------------------
    # 30ステップへ補間
    # --------------------------------------------------------

    all_rows = interpolate_all(
        episodes
    )

    validate_interpolated_dataset(
        all_rows,
        expected_episode_count=300,
    )

    print(
        f"Interpolated rows: "
        f"{len(all_rows)}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_ids,
        val_ids,
        test_ids,
        split_summary,
    ) = split_episodes(
        class_to_ids
    )

    validate_split_overlap(
        train_ids,
        val_ids,
        test_ids,
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
    # 各Split検証
    # --------------------------------------------------------

    validate_interpolated_dataset(
        train_rows,
        expected_episode_count=240,
    )

    validate_interpolated_dataset(
        val_rows,
        expected_episode_count=30,
    )

    validate_interpolated_dataset(
        test_rows,
        expected_episode_count=30,
    )

    # --------------------------------------------------------
    # CSV保存
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
        "===== V5 300-EPISODE SUMMARY ====="
    )

    print(
        f"All:   "
        f"{len(all_rows)} rows "
        f"(300 episodes)"
    )

    print(
        f"Train: "
        f"{len(train_rows)} rows "
        f"(240 episodes)"
    )

    print(
        f"Val:   "
        f"{len(val_rows)} rows "
        f"(30 episodes)"
    )

    print(
        f"Test:  "
        f"{len(test_rows)} rows "
        f"(30 episodes)"
    )

    print_split_class_counts(
        split_summary
    )

    print()

    print(
        "Output files:"
    )

    print(
        f"  All: "
        f"{OUTPUT_ALL}"
    )

    print(
        f"  Train: "
        f"{OUTPUT_TRAIN}"
    )

    print(
        f"  Validation: "
        f"{OUTPUT_VAL}"
    )

    print(
        f"  Test: "
        f"{OUTPUT_TEST}"
    )

    print(
        f"  Split summary: "
        f"{SUMMARY_FILE}"
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
