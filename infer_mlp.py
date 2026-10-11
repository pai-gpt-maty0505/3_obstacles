#!/usr/bin/env python3

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# ファイル設定
# ============================================================

BASE_DIR = Path.home() / "so101_ws"

MODEL_FILE = BASE_DIR / "mlp_so101_best.pt"
TEST_FILE = BASE_DIR / "dataset_v5_test.csv"

OUTPUT_FILE = BASE_DIR / "mlp_test_predictions.csv"
EPISODE_SUMMARY_FILE = BASE_DIR / "mlp_test_episode_summary.csv"
CLASS_SUMMARY_FILE = BASE_DIR / "mlp_test_class_summary.csv"
METRICS_FILE = BASE_DIR / "mlp_test_metrics.json"


# ============================================================
# 入力・出力
# ============================================================

INPUT_COLUMNS = [
    "obs_x",
    "obs_y",
    "obs_z",
    "progress",
]

OUTPUT_COLUMNS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


# ============================================================
# MLP
# ============================================================

class SO101MLP(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
    ):
        super().__init__()

        layers = []
        previous_size = input_size

        for hidden_size in hidden_sizes:

            layers.append(
                nn.Linear(
                    previous_size,
                    hidden_size,
                )
            )

            layers.append(
                nn.ReLU()
            )

            previous_size = hidden_size

        layers.append(
            nn.Linear(
                previous_size,
                output_size,
            )
        )

        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# テストCSV読み込み
# ============================================================

def load_test_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Test file not found: {path}"
        )

    metadata = []
    x_list = []
    y_list = []

    with path.open(
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            x = [
                float(row[column])
                for column in INPUT_COLUMNS
            ]

            y = [
                float(row[column])
                for column in OUTPUT_COLUMNS
            ]

            metadata.append(row)

            x_list.append(x)
            y_list.append(y)

    if not x_list:

        raise RuntimeError(
            "Test dataset is empty."
        )

    x = np.asarray(
        x_list,
        dtype=np.float32,
    )

    y = np.asarray(
        y_list,
        dtype=np.float32,
    )

    return metadata, x, y


# ============================================================
# 正規化
# ============================================================

def normalize(x, mean, std):

    return (
        x - mean
    ) / std


def denormalize(x, mean, std):

    return (
        x * std
        + mean
    )


# ============================================================
# 推論
# ============================================================

def run_inference(
    model,
    x,
    device,
    x_mean,
    x_std,
    y_mean,
    y_std,
):

    x_normalized = normalize(
        x,
        x_mean,
        x_std,
    )

    x_tensor = torch.tensor(
        x_normalized,
        dtype=torch.float32,
    ).to(device)

    model.eval()

    with torch.no_grad():

        y_pred_normalized = (
            model(x_tensor)
            .cpu()
            .numpy()
        )

    y_pred = denormalize(
        y_pred_normalized,
        y_mean,
        y_std,
    )

    return y_pred


# ============================================================
# 全体誤差
# ============================================================

def compute_metrics(
    y_true,
    y_pred,
):

    error = (
        y_pred - y_true
    )

    abs_error = np.abs(error)

    squared_error = (
        error ** 2
    )

    mae_per_joint = (
        abs_error.mean(
            axis=0
        )
    )

    rmse_per_joint = np.sqrt(
        squared_error.mean(
            axis=0
        )
    )

    max_error_per_joint = (
        abs_error.max(
            axis=0
        )
    )

    overall_mae = float(
        abs_error.mean()
    )

    overall_rmse = float(
        np.sqrt(
            squared_error.mean()
        )
    )

    overall_max = float(
        abs_error.max()
    )

    return {
        "mae_per_joint":
            mae_per_joint,

        "rmse_per_joint":
            rmse_per_joint,

        "max_error_per_joint":
            max_error_per_joint,

        "overall_mae":
            overall_mae,

        "overall_rmse":
            overall_rmse,

        "overall_max":
            overall_max,
    }


# ============================================================
# 推論結果CSV
# ============================================================

def save_predictions(
    metadata,
    y_true,
    y_pred,
    output_path,
):

    fieldnames = [
        "episode_id",
        "obstacle_class",
        "step",
        "progress",

        "obs_x",
        "obs_y",
        "obs_z",

        "target_x",
        "target_y",
        "target_z",
    ]

    for joint in OUTPUT_COLUMNS:

        fieldnames += [
            f"true_{joint}",
            f"pred_{joint}",
            f"error_{joint}",
            f"abs_error_{joint}",
        ]

    fieldnames += [
        "row_mae_rad",
        "row_rmse_rad",
        "row_mae_deg",
        "row_rmse_deg",
    ]

    with output_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i, row in enumerate(metadata):

            output = {
                "episode_id":
                    row["episode_id"],

                "obstacle_class":
                    row["obstacle_class"],

                "step":
                    row["step"],

                "progress":
                    row["progress"],

                "obs_x":
                    row["obs_x"],

                "obs_y":
                    row["obs_y"],

                "obs_z":
                    row["obs_z"],

                "target_x":
                    row["target_x"],

                "target_y":
                    row["target_y"],

                "target_z":
                    row["target_z"],
            }

            row_errors = []

            for joint_index, joint in enumerate(
                OUTPUT_COLUMNS
            ):

                true_value = float(
                    y_true[
                        i,
                        joint_index
                    ]
                )

                pred_value = float(
                    y_pred[
                        i,
                        joint_index
                    ]
                )

                error = (
                    pred_value
                    - true_value
                )

                abs_error = abs(
                    error
                )

                row_errors.append(
                    error
                )

                output[
                    f"true_{joint}"
                ] = true_value

                output[
                    f"pred_{joint}"
                ] = pred_value

                output[
                    f"error_{joint}"
                ] = error

                output[
                    f"abs_error_{joint}"
                ] = abs_error

            row_errors_np = np.asarray(
                row_errors,
                dtype=np.float32,
            )

            row_mae = float(
                np.mean(
                    np.abs(
                        row_errors_np
                    )
                )
            )

            row_rmse = float(
                np.sqrt(
                    np.mean(
                        row_errors_np ** 2
                    )
                )
            )

            output[
                "row_mae_rad"
            ] = row_mae

            output[
                "row_rmse_rad"
            ] = row_rmse

            output[
                "row_mae_deg"
            ] = math.degrees(
                row_mae
            )

            output[
                "row_rmse_deg"
            ] = math.degrees(
                row_rmse
            )

            writer.writerow(
                output
            )


# ============================================================
# エピソード単位評価
# ============================================================

def compute_episode_summary(
    metadata,
    y_true,
    y_pred,
):

    episode_indices = defaultdict(
        list
    )

    for i, row in enumerate(metadata):

        episode_id = int(
            row["episode_id"]
        )

        episode_indices[
            episode_id
        ].append(i)

    summaries = []

    for episode_id in sorted(
        episode_indices.keys()
    ):

        indices = (
            episode_indices[
                episode_id
            ]
        )

        true_values = (
            y_true[
                indices
            ]
        )

        pred_values = (
            y_pred[
                indices
            ]
        )

        error = (
            pred_values
            - true_values
        )

        mae = float(
            np.mean(
                np.abs(
                    error
                )
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        max_error = float(
            np.max(
                np.abs(
                    error
                )
            )
        )

        first = metadata[
            indices[0]
        ]

        summaries.append(
            {
                "episode_id":
                    episode_id,

                "obstacle_class":
                    first[
                        "obstacle_class"
                    ],

                "obs_x":
                    float(
                        first["obs_x"]
                    ),

                "obs_y":
                    float(
                        first["obs_y"]
                    ),

                "obs_z":
                    float(
                        first["obs_z"]
                    ),

                "steps":
                    len(indices),

                "mae_rad":
                    mae,

                "rmse_rad":
                    rmse,

                "max_error_rad":
                    max_error,

                "mae_deg":
                    math.degrees(
                        mae
                    ),

                "rmse_deg":
                    math.degrees(
                        rmse
                    ),

                "max_error_deg":
                    math.degrees(
                        max_error
                    ),
            }
        )

    return summaries


# ============================================================
# クラス単位評価
# ============================================================

def compute_class_summary(
    metadata,
    y_true,
    y_pred,
):

    class_indices = defaultdict(
        list
    )

    for i, row in enumerate(metadata):

        obstacle_class = (
            row[
                "obstacle_class"
            ]
        )

        class_indices[
            obstacle_class
        ].append(i)

    summaries = []

    for obstacle_class in sorted(
        class_indices.keys()
    ):

        indices = (
            class_indices[
                obstacle_class
            ]
        )

        error = (
            y_pred[indices]
            - y_true[indices]
        )

        mae = float(
            np.mean(
                np.abs(
                    error
                )
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        max_error = float(
            np.max(
                np.abs(
                    error
                )
            )
        )

        episode_ids = set(
            int(
                metadata[i][
                    "episode_id"
                ]
            )
            for i in indices
        )

        summaries.append(
            {
                "obstacle_class":
                    obstacle_class,

                "episodes":
                    len(
                        episode_ids
                    ),

                "rows":
                    len(indices),

                "mae_rad":
                    mae,

                "rmse_rad":
                    rmse,

                "max_error_rad":
                    max_error,

                "mae_deg":
                    math.degrees(
                        mae
                    ),

                "rmse_deg":
                    math.degrees(
                        rmse
                    ),

                "max_error_deg":
                    math.degrees(
                        max_error
                    ),
            }
        )

    return summaries


# ============================================================
# Summary CSV保存
# ============================================================

def save_episode_summary(rows):

    fieldnames = [
        "episode_id",
        "obstacle_class",
        "obs_x",
        "obs_y",
        "obs_z",
        "steps",

        "mae_rad",
        "rmse_rad",
        "max_error_rad",

        "mae_deg",
        "rmse_deg",
        "max_error_deg",
    ]

    with EPISODE_SUMMARY_FILE.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


def save_class_summary(rows):

    fieldnames = [
        "obstacle_class",
        "episodes",
        "rows",

        "mae_rad",
        "rmse_rad",
        "max_error_rad",

        "mae_deg",
        "rmse_deg",
        "max_error_deg",
    ]

    with CLASS_SUMMARY_FILE.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():

    print(
        "================================"
    )

    print(
        "SO-101 MLP Inference"
    )

    print(
        "================================"
    )

    # --------------------------------------------------------
    # GPU確認
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        print(
            "Device: CUDA"
        )

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    else:

        device = torch.device(
            "cpu"
        )

        print(
            "Device: CPU"
        )

    # --------------------------------------------------------
    # 学習済みモデル読み込み
    # --------------------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found: "
            f"{MODEL_FILE}"
        )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    if (
        checkpoint[
            "input_columns"
        ]
        != INPUT_COLUMNS
    ):

        raise RuntimeError(
            "Input columns mismatch."
        )

    if (
        checkpoint[
            "output_columns"
        ]
        != OUTPUT_COLUMNS
    ):

        raise RuntimeError(
            "Output columns mismatch."
        )

    hidden_sizes = checkpoint[
        "hidden_sizes"
    ]

    x_mean = np.asarray(
        checkpoint[
            "x_mean"
        ],
        dtype=np.float32,
    )

    x_std = np.asarray(
        checkpoint[
            "x_std"
        ],
        dtype=np.float32,
    )

    y_mean = np.asarray(
        checkpoint[
            "y_mean"
        ],
        dtype=np.float32,
    )

    y_std = np.asarray(
        checkpoint[
            "y_std"
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # モデル生成
    # --------------------------------------------------------

    model = SO101MLP(
        input_size=len(
            INPUT_COLUMNS
        ),
        output_size=len(
            OUTPUT_COLUMNS
        ),
        hidden_sizes=hidden_sizes,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "Model loaded:"
    )

    print(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # テストデータ
    # --------------------------------------------------------

    (
        metadata,
        x_test,
        y_test,
    ) = load_test_csv(
        TEST_FILE
    )

    print(
        f"Test rows: "
        f"{len(x_test)}"
    )

    episode_ids = sorted(
        set(
            int(
                row[
                    "episode_id"
                ]
            )
            for row in metadata
        )
    )

    print(
        f"Test episodes: "
        f"{len(episode_ids)}"
    )

    # --------------------------------------------------------
    # 推論
    # --------------------------------------------------------

    y_pred = run_inference(
        model,
        x_test,
        device,
        x_mean,
        x_std,
        y_mean,
        y_std,
    )

    # --------------------------------------------------------
    # 評価
    # --------------------------------------------------------

    metrics = compute_metrics(
        y_test,
        y_pred,
    )

    # --------------------------------------------------------
    # CSV保存
    # --------------------------------------------------------

    save_predictions(
        metadata,
        y_test,
        y_pred,
        OUTPUT_FILE,
    )

    episode_summary = (
        compute_episode_summary(
            metadata,
            y_test,
            y_pred,
        )
    )

    save_episode_summary(
        episode_summary
    )

    class_summary = (
        compute_class_summary(
            metadata,
            y_test,
            y_pred,
        )
    )

    save_class_summary(
        class_summary
    )

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print()
    print(
        "===== TEST RESULT ====="
    )

    print()
    print(
        "MAE per joint:"
    )

    for joint, value in zip(
        OUTPUT_COLUMNS,
        metrics[
            "mae_per_joint"
        ],
    ):

        print(
            f"  {joint:15s}: "
            f"{value:.6f} rad "
            f"("
            f"{math.degrees(value):.3f} deg"
            f")"
        )

    print()
    print(
        "RMSE per joint:"
    )

    for joint, value in zip(
        OUTPUT_COLUMNS,
        metrics[
            "rmse_per_joint"
        ],
    ):

        print(
            f"  {joint:15s}: "
            f"{value:.6f} rad "
            f"("
            f"{math.degrees(value):.3f} deg"
            f")"
        )

    print()
    print(
        f"Overall MAE: "
        f"{metrics['overall_mae']:.6f} rad "
        f"("
        f"{math.degrees(metrics['overall_mae']):.3f} deg"
        f")"
    )

    print(
        f"Overall RMSE: "
        f"{metrics['overall_rmse']:.6f} rad "
        f"("
        f"{math.degrees(metrics['overall_rmse']):.3f} deg"
        f")"
    )

    print(
        f"Overall MAX error: "
        f"{metrics['overall_max']:.6f} rad "
        f"("
        f"{math.degrees(metrics['overall_max']):.3f} deg"
        f")"
    )

    # --------------------------------------------------------
    # エピソード別結果
    # --------------------------------------------------------

    print()
    print(
        "===== EPISODE SUMMARY ====="
    )

    for row in episode_summary:

        print(
            f"Episode "
            f"{row['episode_id']:3d} "
            f"[{row['obstacle_class']:6s}] "
            f"MAE="
            f"{row['mae_deg']:.3f} deg "
            f"RMSE="
            f"{row['rmse_deg']:.3f} deg "
            f"MAX="
            f"{row['max_error_deg']:.3f} deg"
        )

    # --------------------------------------------------------
    # クラス別結果
    # --------------------------------------------------------

    print()
    print(
        "===== CLASS SUMMARY ====="
    )

    for row in class_summary:

        print(
            f"{row['obstacle_class']:6s}: "
            f"episodes="
            f"{row['episodes']} "
            f"MAE="
            f"{row['mae_deg']:.3f} deg "
            f"RMSE="
            f"{row['rmse_deg']:.3f} deg "
            f"MAX="
            f"{row['max_error_deg']:.3f} deg"
        )

    # --------------------------------------------------------
    # JSON保存
    # --------------------------------------------------------

    json_metrics = {
        "device":
            str(device),

        "test_rows":
            len(x_test),

        "test_episodes":
            len(
                episode_ids
            ),

        "overall_mae_rad":
            metrics[
                "overall_mae"
            ],

        "overall_rmse_rad":
            metrics[
                "overall_rmse"
            ],

        "overall_max_error_rad":
            metrics[
                "overall_max"
            ],

        "overall_mae_deg":
            math.degrees(
                metrics[
                    "overall_mae"
                ]
            ),

        "overall_rmse_deg":
            math.degrees(
                metrics[
                    "overall_rmse"
                ]
            ),

        "overall_max_error_deg":
            math.degrees(
                metrics[
                    "overall_max"
                ]
            ),

        "mae_per_joint_rad":
            {
                joint:
                    float(value)
                for joint, value
                in zip(
                    OUTPUT_COLUMNS,
                    metrics[
                        "mae_per_joint"
                    ],
                )
            },

        "rmse_per_joint_rad":
            {
                joint:
                    float(value)
                for joint, value
                in zip(
                    OUTPUT_COLUMNS,
                    metrics[
                        "rmse_per_joint"
                    ],
                )
            },
    }

    with METRICS_FILE.open(
        "w"
    ) as f:

        json.dump(
            json_metrics,
            f,
            indent=2,
        )

    print()
    print(
        "===== OUTPUT FILES ====="
    )

    print(
        f"Predictions: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Episode summary: "
        f"{EPISODE_SUMMARY_FILE}"
    )

    print(
        f"Class summary: "
        f"{CLASS_SUMMARY_FILE}"
    )

    print(
        f"Metrics: "
        f"{METRICS_FILE}"
    )

    print()
    print(
        "Inference completed."
    )


if __name__ == "__main__":

    main()
