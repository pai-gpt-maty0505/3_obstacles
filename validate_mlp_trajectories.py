#!/usr/bin/env python3

import csv
import math
from collections import defaultdict
from pathlib import Path

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from moveit_msgs.msg import (
    CollisionObject,
    PlanningScene,
    RobotState,
)

from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetStateValidity,
)


# ============================================================
# 設定
# ============================================================

BASE_DIR = Path.home() / "so101_ws"

INPUT_FILE = (
    BASE_DIR
    / "mlp_test_predictions.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "mlp_trajectory_validation.csv"
)

SAFE_FILE = (
    BASE_DIR
    / "mlp_safe_episodes.csv"
)

DEMO_FILE = (
    BASE_DIR
    / "mlp_demo_candidates.csv"
)


ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

GROUP_NAME = "arm"
BASE_FRAME = "base_link"


# V4.2と同じ障害物サイズ
OBSTACLE_SIZE_X = 0.055
OBSTACLE_SIZE_Y = 0.055
OBSTACLE_SIZE_Z = 0.22


# 30点の間をさらに何分割するか
#
# 5なら、
# step0 → 0.2 → 0.4 → 0.6 → 0.8 → step1
# のように確認する。
INTERPOLATION_SUBSTEPS = 5


# ============================================================
# ROS Node
# ============================================================

class TrajectoryValidator(Node):

    def __init__(self):

        super().__init__(
            "validate_mlp_trajectories"
        )

        self.scene_client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

        self.validity_client = self.create_client(
            GetStateValidity,
            "/check_state_validity",
        )

        self.get_logger().info(
            "Waiting for "
            "/apply_planning_scene ..."
        )

        if not self.scene_client.wait_for_service(
            timeout_sec=10.0
        ):
            raise RuntimeError(
                "/apply_planning_scene "
                "not available"
            )

        self.get_logger().info(
            "Waiting for "
            "/check_state_validity ..."
        )

        if not self.validity_client.wait_for_service(
            timeout_sec=10.0
        ):
            raise RuntimeError(
                "/check_state_validity "
                "not available"
            )

        self.get_logger().info(
            "MoveIt validation services "
            "are ready."
        )

    # --------------------------------------------------------
    # RobotState
    # --------------------------------------------------------

    def make_robot_state(
        self,
        joints,
    ):

        state = RobotState()

        state.joint_state = JointState()

        state.joint_state.name = (
            ARM_JOINTS.copy()
        )

        state.joint_state.position = [
            float(
                joints[name]
            )
            for name in ARM_JOINTS
        ]

        return state

    # --------------------------------------------------------
    # 状態妥当性
    # --------------------------------------------------------

    def state_is_valid(
        self,
        joints,
    ):

        req = GetStateValidity.Request()

        req.robot_state = (
            self.make_robot_state(
                joints
            )
        )

        req.group_name = GROUP_NAME

        future = (
            self.validity_client.call_async(
                req
            )
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()

        if response is None:
            return False

        return bool(
            response.valid
        )

    # --------------------------------------------------------
    # 障害物追加
    # --------------------------------------------------------

    def add_obstacle(
        self,
        obstacle_id,
        x,
        y,
        z,
    ):

        obj = CollisionObject()

        obj.id = obstacle_id

        obj.header.frame_id = (
            BASE_FRAME
        )

        primitive = SolidPrimitive()

        primitive.type = (
            SolidPrimitive.BOX
        )

        primitive.dimensions = [
            OBSTACLE_SIZE_X,
            OBSTACLE_SIZE_Y,
            OBSTACLE_SIZE_Z,
        ]

        pose = Pose()

        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)

        pose.orientation.w = 1.0

        obj.primitives = [
            primitive
        ]

        obj.primitive_poses = [
            pose
        ]

        obj.operation = (
            CollisionObject.ADD
        )

        scene = PlanningScene()

        scene.is_diff = True

        scene.world.collision_objects = [
            obj
        ]

        req = ApplyPlanningScene.Request()

        req.scene = scene

        future = (
            self.scene_client.call_async(
                req
            )
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()

        return (
            response is not None
            and response.success
        )

    # --------------------------------------------------------
    # 障害物削除
    # --------------------------------------------------------

    def remove_obstacle(
        self,
        obstacle_id,
    ):

        obj = CollisionObject()

        obj.id = obstacle_id

        obj.header.frame_id = (
            BASE_FRAME
        )

        obj.operation = (
            CollisionObject.REMOVE
        )

        scene = PlanningScene()

        scene.is_diff = True

        scene.world.collision_objects = [
            obj
        ]

        req = ApplyPlanningScene.Request()

        req.scene = scene

        future = (
            self.scene_client.call_async(
                req
            )
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        return future.result()


# ============================================================
# CSV読み込み
# ============================================================

def load_predictions(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{path}"
        )

    episodes = defaultdict(list)

    with path.open(
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            episode_id = int(
                row["episode_id"]
            )

            episodes[
                episode_id
            ].append(row)

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
# 予測関節角度取得
# ============================================================

def get_predicted_joints(row):

    return {
        joint: float(
            row[
                f"pred_{joint}"
            ]
        )
        for joint in ARM_JOINTS
    }


# ============================================================
# 2姿勢間補間
# ============================================================

def interpolate_joints(
    joints_a,
    joints_b,
    alpha,
):

    return {
        joint: (
            joints_a[joint]
            + alpha
            * (
                joints_b[joint]
                - joints_a[joint]
            )
        )
        for joint in ARM_JOINTS
    }


# ============================================================
# 最大step差
# ============================================================

def compute_max_step_delta(rows):

    max_delta = 0.0

    for i in range(
        len(rows) - 1
    ):

        q0 = get_predicted_joints(
            rows[i]
        )

        q1 = get_predicted_joints(
            rows[i + 1]
        )

        for joint in ARM_JOINTS:

            delta = abs(
                q1[joint]
                - q0[joint]
            )

            max_delta = max(
                max_delta,
                delta,
            )

    return max_delta


# ============================================================
# 1エピソード安全性確認
# ============================================================

def validate_episode(
    node,
    rows,
):

    total_checks = 0
    invalid_checks = 0

    first_invalid_progress = None
    first_invalid_step = None

    # --------------------------------------------------------
    # 各step間を補間しながら確認
    # --------------------------------------------------------

    for step_index in range(
        len(rows) - 1
    ):

        row_a = rows[
            step_index
        ]

        row_b = rows[
            step_index + 1
        ]

        joints_a = (
            get_predicted_joints(
                row_a
            )
        )

        joints_b = (
            get_predicted_joints(
                row_b
            )
        )

        progress_a = float(
            row_a[
                "progress"
            ]
        )

        progress_b = float(
            row_b[
                "progress"
            ]
        )

        for substep in range(
            INTERPOLATION_SUBSTEPS
        ):

            alpha = (
                substep
                / float(
                    INTERPOLATION_SUBSTEPS
                )
            )

            joints = (
                interpolate_joints(
                    joints_a,
                    joints_b,
                    alpha,
                )
            )

            progress = (
                progress_a
                + alpha
                * (
                    progress_b
                    - progress_a
                )
            )

            total_checks += 1

            valid = (
                node.state_is_valid(
                    joints
                )
            )

            if not valid:

                invalid_checks += 1

                if (
                    first_invalid_progress
                    is None
                ):

                    first_invalid_progress = (
                        progress
                    )

                    first_invalid_step = (
                        step_index
                    )

    # --------------------------------------------------------
    # 最終stepも確認
    # --------------------------------------------------------

    final_row = rows[-1]

    final_joints = (
        get_predicted_joints(
            final_row
        )
    )

    total_checks += 1

    if not node.state_is_valid(
        final_joints
    ):

        invalid_checks += 1

        if (
            first_invalid_progress
            is None
        ):

            first_invalid_progress = (
                float(
                    final_row[
                        "progress"
                    ]
                )
            )

            first_invalid_step = (
                int(
                    final_row[
                        "step"
                    ]
                )
            )

    safe = (
        invalid_checks == 0
    )

    return {
        "safe":
            safe,

        "total_checks":
            total_checks,

        "invalid_checks":
            invalid_checks,

        "first_invalid_step":
            first_invalid_step,

        "first_invalid_progress":
            first_invalid_progress,
    }


# ============================================================
# CSV保存
# ============================================================

def save_csv(
    path,
    rows,
):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "================================"
    )

    print(
        "SO-101 MLP Trajectory "
        "Safety Validation"
    )

    print(
        "================================"
    )

    episodes = load_predictions(
        INPUT_FILE
    )

    print(
        f"Test episodes: "
        f"{len(episodes)}"
    )

    rclpy.init()

    node = TrajectoryValidator()

    obstacle_id = (
        "mlp_validation_obstacle"
    )

    validation_rows = []

    # --------------------------------------------------------
    # 各エピソード確認
    # --------------------------------------------------------

    for episode_id in sorted(
        episodes.keys()
    ):

        rows = episodes[
            episode_id
        ]

        first = rows[0]

        obstacle_class = (
            first[
                "obstacle_class"
            ]
        )

        obs_x = float(
            first["obs_x"]
        )

        obs_y = float(
            first["obs_y"]
        )

        obs_z = float(
            first["obs_z"]
        )

        node.remove_obstacle(
            obstacle_id
        )

        if not node.add_obstacle(
            obstacle_id,
            obs_x,
            obs_y,
            obs_z,
        ):

            raise RuntimeError(
                "Failed to add obstacle."
            )

        result = validate_episode(
            node,
            rows,
        )

        mean_mae_deg = sum(
            float(
                row[
                    "row_mae_deg"
                ]
            )
            for row in rows
        ) / len(rows)

        max_step_delta = (
            compute_max_step_delta(
                rows
            )
        )

        validation_row = {
            "episode_id":
                episode_id,

            "obstacle_class":
                obstacle_class,

            "obs_x":
                obs_x,

            "obs_y":
                obs_y,

            "obs_z":
                obs_z,

            "safe":
                int(
                    result[
                        "safe"
                    ]
                ),

            "total_checks":
                result[
                    "total_checks"
                ],

            "invalid_checks":
                result[
                    "invalid_checks"
                ],

            "first_invalid_step":
                (
                    ""
                    if result[
                        "first_invalid_step"
                    ] is None
                    else result[
                        "first_invalid_step"
                    ]
                ),

            "first_invalid_progress":
                (
                    ""
                    if result[
                        "first_invalid_progress"
                    ] is None
                    else result[
                        "first_invalid_progress"
                    ]
                ),

            "mean_mae_deg":
                mean_mae_deg,

            "max_step_delta_rad":
                max_step_delta,

            "max_step_delta_deg":
                math.degrees(
                    max_step_delta
                ),
        }

        validation_rows.append(
            validation_row
        )

        status = (
            "SAFE"
            if result["safe"]
            else "COLLISION"
        )

        print(
            f"Episode "
            f"{episode_id:3d} "
            f"[{obstacle_class:6s}] "
            f"{status:9s} "
            f"invalid="
            f"{result['invalid_checks']:3d}/"
            f"{result['total_checks']:3d} "
            f"MAE="
            f"{mean_mae_deg:.3f} deg "
            f"max_step="
            f"{math.degrees(max_step_delta):.3f} deg"
        )

    node.remove_obstacle(
        obstacle_id
    )

    # --------------------------------------------------------
    # 集計
    # --------------------------------------------------------

    safe_rows = [
        row
        for row in validation_rows
        if row["safe"] == 1
    ]

    unsafe_rows = [
        row
        for row in validation_rows
        if row["safe"] == 0
    ]

    class_counts = {}

    for class_name in [
        "left",
        "center",
        "right",
    ]:

        class_rows = [
            row
            for row in validation_rows
            if row[
                "obstacle_class"
            ] == class_name
        ]

        class_safe = [
            row
            for row in class_rows
            if row[
                "safe"
            ] == 1
        ]

        class_counts[
            class_name
        ] = (
            len(class_safe),
            len(class_rows),
        )

    # --------------------------------------------------------
    # デモ候補
    #
    # 各クラスの安全なものから
    # mean MAE最小の1件
    # --------------------------------------------------------

    demo_rows = []

    for class_name in [
        "left",
        "center",
        "right",
    ]:

        candidates = [
            row
            for row in safe_rows
            if row[
                "obstacle_class"
            ] == class_name
        ]

        if not candidates:
            continue

        candidates.sort(
            key=lambda row: (
                row[
                    "mean_mae_deg"
                ],
                row[
                    "max_step_delta_rad"
                ],
            )
        )

        demo_rows.append(
            candidates[0]
        )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    save_csv(
        OUTPUT_FILE,
        validation_rows,
    )

    save_csv(
        SAFE_FILE,
        safe_rows,
    )

    save_csv(
        DEMO_FILE,
        demo_rows,
    )

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print()
    print(
        "===== SAFETY SUMMARY ====="
    )

    print(
        f"Total: "
        f"{len(validation_rows)}"
    )

    print(
        f"Safe: "
        f"{len(safe_rows)}"
    )

    print(
        f"Collision: "
        f"{len(unsafe_rows)}"
    )

    if validation_rows:

        safe_rate = (
            100.0
            * len(safe_rows)
            / len(validation_rows)
        )

        print(
            f"Safety rate: "
            f"{safe_rate:.1f}%"
        )

    print()

    for class_name in [
        "left",
        "center",
        "right",
    ]:

        safe_count, total_count = (
            class_counts[
                class_name
            ]
        )

        print(
            f"{class_name:6s}: "
            f"{safe_count}/"
            f"{total_count} safe"
        )

    print()

    print(
        "Demo candidates:"
    )

    for row in demo_rows:

        print(
            f"  "
            f"{row['obstacle_class']:6s} "
            f"Episode "
            f"{row['episode_id']} "
            f"MAE="
            f"{row['mean_mae_deg']:.3f} deg"
        )

    print()

    print(
        f"All results: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Safe episodes: "
        f"{SAFE_FILE}"
    )

    print(
        f"Demo candidates: "
        f"{DEMO_FILE}"
    )

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":

    main()
