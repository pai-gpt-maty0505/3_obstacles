#!/usr/bin/env python3

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
    DataLoader,
)


# ============================================================
# 設定
# ============================================================

BASE_DIR = (
    Path.home()
    / "so101_ws"
)

TRAIN_FILE = (
    BASE_DIR
    / "dataset_v5_train.csv"
)

VAL_FILE = (
    BASE_DIR
    / "dataset_v5_val.csv"
)

TEST_FILE = (
    BASE_DIR
    / "dataset_v5_test.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "mlp_so101_best.pt"
)

STATS_FILE = (
    BASE_DIR
    / "mlp_so101_stats.json"
)

HISTORY_FILE = (
    BASE_DIR
    / "mlp_so101_history.csv"
)


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


RANDOM_SEED = 42

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 500

PATIENCE = 50

HIDDEN_1 = 128
HIDDEN_2 = 128
HIDDEN_3 = 64


# ============================================================
# 乱数固定
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# CSV読み込み
# ============================================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    x_list = []
    y_list = []

    with path.open(
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            x = [
                float(
                    row[column]
                )
                for column
                in INPUT_COLUMNS
            ]

            y = [
                float(
                    row[column]
                )
                for column
                in OUTPUT_COLUMNS
            ]

            x_list.append(x)
            y_list.append(y)

    if not x_list:

        raise RuntimeError(
            f"No data in {path}"
        )

    x = np.asarray(
        x_list,
        dtype=np.float32,
    )

    y = np.asarray(
        y_list,
        dtype=np.float32,
    )

    return x, y


# ============================================================
# 標準化
# ============================================================

def compute_normalization(
    x,
    y,
):

    x_mean = x.mean(
        axis=0
    )

    x_std = x.std(
        axis=0
    )

    y_mean = y.mean(
        axis=0
    )

    y_std = y.std(
        axis=0
    )

    # 0除算防止
    x_std[
        x_std < 1e-8
    ] = 1.0

    y_std[
        y_std < 1e-8
    ] = 1.0

    return (
        x_mean,
        x_std,
        y_mean,
        y_std,
    )


def normalize_x(
    x,
    mean,
    std,
):

    return (
        x - mean
    ) / std


def normalize_y(
    y,
    mean,
    std,
):

    return (
        y - mean
    ) / std


def denormalize_y(
    y,
    mean,
    std,
):

    return (
        y * std
        + mean
    )


# ============================================================
# Dataset
# ============================================================

class SO101Dataset(Dataset):

    def __init__(
        self,
        x,
        y,
    ):

        self.x = torch.tensor(
            x,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32,
        )

    def __len__(self):

        return len(
            self.x
        )

    def __getitem__(
        self,
        index,
    ):

        return (
            self.x[index],
            self.y[index],
        )


# ============================================================
# MLP
# ============================================================

class SO101MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                len(
                    INPUT_COLUMNS
                ),
                HIDDEN_1,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_1,
                HIDDEN_2,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_2,
                HIDDEN_3,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_3,
                len(
                    OUTPUT_COLUMNS
                ),
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.net(x)


# ============================================================
# 評価
# ============================================================

def evaluate_loss(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                device
            )

            y = y.to(
                device
            )

            pred = model(x)

            loss = criterion(
                pred,
                y,
            )

            batch_size = (
                x.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_count += (
                batch_size
            )

    return (
        total_loss
        / max(
            total_count,
            1,
        )
    )


# ============================================================
# テスト評価
# ============================================================

def evaluate_test(
    model,
    x_test,
    y_test,
    device,
    x_mean,
    x_std,
    y_mean,
    y_std,
):

    model.eval()

    x_norm = normalize_x(
        x_test,
        x_mean,
        x_std,
    )

    x_tensor = torch.tensor(
        x_norm,
        dtype=torch.float32,
    ).to(
        device
    )

    with torch.no_grad():

        pred_norm = model(
            x_tensor
        ).cpu().numpy()

    pred = denormalize_y(
        pred_norm,
        y_mean,
        y_std,
    )

    error = (
        pred - y_test
    )

    mae_per_joint = (
        np.mean(
            np.abs(
                error
            ),
            axis=0,
        )
    )

    rmse_per_joint = (
        np.sqrt(
            np.mean(
                error ** 2,
                axis=0,
            )
        )
    )

    overall_mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    overall_rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    return (
        pred,
        mae_per_joint,
        rmse_per_joint,
        overall_mae,
        overall_rmse,
    )


# ============================================================
# 学習履歴保存
# ============================================================

def save_history(
    history,
):

    with HISTORY_FILE.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
            ],
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    print(
        "================================"
    )

    print(
        "SO-101 MLP Training"
    )

    print(
        "================================"
    )

    # --------------------------------------------------------
    # device
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
    # データ読み込み
    # --------------------------------------------------------

    x_train, y_train = (
        load_csv(
            TRAIN_FILE
        )
    )

    x_val, y_val = (
        load_csv(
            VAL_FILE
        )
    )

    x_test, y_test = (
        load_csv(
            TEST_FILE
        )
    )

    print(
        f"Train samples: "
        f"{len(x_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(x_val)}"
    )

    print(
        f"Test samples: "
        f"{len(x_test)}"
    )

    # --------------------------------------------------------
    # 正規化統計
    #
    # trainデータだけから計算する
    # --------------------------------------------------------

    (
        x_mean,
        x_std,
        y_mean,
        y_std,
    ) = compute_normalization(
        x_train,
        y_train,
    )

    x_train_n = normalize_x(
        x_train,
        x_mean,
        x_std,
    )

    x_val_n = normalize_x(
        x_val,
        x_mean,
        x_std,
    )

    y_train_n = normalize_y(
        y_train,
        y_mean,
        y_std,
    )

    y_val_n = normalize_y(
        y_val,
        y_mean,
        y_std,
    )

    # --------------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------------

    train_dataset = (
        SO101Dataset(
            x_train_n,
            y_train_n,
        )
    )

    val_dataset = (
        SO101Dataset(
            x_val_n,
            y_val_n,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SO101MLP().to(
        device
    )

    print()
    print(model)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count}"
    )

    criterion = nn.MSELoss()

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=15,
            min_lr=1e-6,
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = (
        float(
            "inf"
        )
    )

    best_epoch = 0

    patience_counter = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        train_total = 0.0
        train_count = 0

        for x, y in train_loader:

            x = x.to(
                device
            )

            y = y.to(
                device
            )

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(
                pred,
                y,
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                x.shape[0]
            )

            train_total += (
                loss.item()
                * batch_size
            )

            train_count += (
                batch_size
            )

        train_loss = (
            train_total
            / train_count
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(
            val_loss
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,
            }
        )

        # ----------------------------------------------------
        # best model
        # ----------------------------------------------------

        if val_loss < (
            best_val_loss
            - 1e-8
        ):

            best_val_loss = (
                val_loss
            )

            best_epoch = (
                epoch
            )

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_columns":
                        INPUT_COLUMNS,

                    "output_columns":
                        OUTPUT_COLUMNS,

                    "x_mean":
                        x_mean.tolist(),

                    "x_std":
                        x_std.tolist(),

                    "y_mean":
                        y_mean.tolist(),

                    "y_std":
                        y_std.tolist(),

                    "hidden_sizes": [
                        HIDDEN_1,
                        HIDDEN_2,
                        HIDDEN_3,
                    ],
                },
                MODEL_FILE,
            )

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # 表示
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            current_lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            print(
                f"Epoch "
                f"{epoch:4d} | "
                f"train="
                f"{train_loss:.6f} | "
                f"val="
                f"{val_loss:.6f} | "
                f"lr="
                f"{current_lr:.2e}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                f"Early stopping "
                f"at epoch "
                f"{epoch}"
            )

            break

    # --------------------------------------------------------
    # History保存
    # --------------------------------------------------------

    save_history(
        history
    )

    # --------------------------------------------------------
    # Bestモデル再読み込み
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    (
        predictions,
        mae_per_joint,
        rmse_per_joint,
        overall_mae,
        overall_rmse,
    ) = evaluate_test(
        model,
        x_test,
        y_test,
        device,
        x_mean,
        x_std,
        y_mean,
        y_std,
    )

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print()
    print(
        "================================"
    )

    print(
        "===== TRAINING RESULT ====="
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.8f}"
    )

    print()
    print(
        "Test MAE [rad]:"
    )

    for joint, value in zip(
        OUTPUT_COLUMNS,
        mae_per_joint,
    ):

        print(
            f"  {joint:15s}: "
            f"{value:.6f}"
        )

    print()
    print(
        "Test RMSE [rad]:"
    )

    for joint, value in zip(
        OUTPUT_COLUMNS,
        rmse_per_joint,
    ):

        print(
            f"  {joint:15s}: "
            f"{value:.6f}"
        )

    print()
    print(
        f"Overall MAE: "
        f"{overall_mae:.6f} rad"
    )

    print(
        f"Overall RMSE: "
        f"{overall_rmse:.6f} rad"
    )

    print(
        f"Overall MAE: "
        f"{math.degrees(overall_mae):.3f} deg"
    )

    print(
        f"Overall RMSE: "
        f"{math.degrees(overall_rmse):.3f} deg"
    )

    # --------------------------------------------------------
    # stats JSON
    # --------------------------------------------------------

    stats = {
        "device":
            str(device),

        "best_epoch":
            best_epoch,

        "best_validation_loss":
            best_val_loss,

        "overall_mae_rad":
            overall_mae,

        "overall_rmse_rad":
            overall_rmse,

        "overall_mae_deg":
            math.degrees(
                overall_mae
            ),

        "overall_rmse_deg":
            math.degrees(
                overall_rmse
            ),

        "mae_per_joint_rad":
            {
                joint:
                    float(value)
                for joint, value
                in zip(
                    OUTPUT_COLUMNS,
                    mae_per_joint,
                )
            },

        "rmse_per_joint_rad":
            {
                joint:
                    float(value)
                for joint, value
                in zip(
                    OUTPUT_COLUMNS,
                    rmse_per_joint,
                )
            },
    }

    with STATS_FILE.open(
        "w",
    ) as f:

        json.dump(
            stats,
            f,
            indent=2,
        )

    print()
    print(
        f"Best model: "
        f"{MODEL_FILE}"
    )

    print(
        f"Statistics: "
        f"{STATS_FILE}"
    )

    print(
        f"History: "
        f"{HISTORY_FILE}"
    )

    print()
    print(
        "Training completed."
    )


if __name__ == "__main__":

    main()
