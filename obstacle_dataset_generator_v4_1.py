#!/usr/bin/env python3

import csv
import math
import random
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    PlanningScene,
    RobotState,
)

from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetMotionPlan,
    GetPositionFK,
    GetPositionIK,
    GetStateValidity,
)


# =========================================================
# 基本設定
# =========================================================

ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

BASE_FRAME = "base_link"
TIP_LINK = "gripper_link"
GROUP_NAME = "arm"

random.seed(42)


# =========================================================
# 固定開始姿勢
# =========================================================

START_JOINTS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
}


# =========================================================
# 固定目標姿勢候補
#
# 上から順番に試し、
# IK・衝突判定・経路計画が通るものを採用
# =========================================================

GOAL_CANDIDATES = [
    {
        "shoulder_pan": 0.60,
        "shoulder_lift": -0.60,
        "elbow_flex": 0.70,
        "wrist_flex": -0.30,
        "wrist_roll": 0.40,
    },
    {
        "shoulder_pan": 0.45,
        "shoulder_lift": -0.65,
        "elbow_flex": 0.65,
        "wrist_flex": -0.25,
        "wrist_roll": 0.35,
    },
    {
        "shoulder_pan": 0.30,
        "shoulder_lift": -0.80,
        "elbow_flex": 0.60,
        "wrist_flex": -0.30,
        "wrist_roll": 0.50,
    },
]


# =========================================================
# 障害物設定
#
# v4より少し小さくして
# Endpoint collisionを減らす
# =========================================================

OBSTACLE_SIZE_X = 0.045
OBSTACLE_SIZE_Y = 0.045
OBSTACLE_SIZE_Z = 0.20

OBSTACLE_Z = OBSTACLE_SIZE_Z / 2.0


# =========================================================
# データ生成設定
# =========================================================

EPISODES_PER_CLASS = 10

CLASSES = [
    "left",
    "center",
    "right",
]

# 1クラスあたり最大探索数
MAX_ATTEMPTS_PER_CLASS = 1000


# =========================================================
# 経路中央線からの横方向オフセット
#
# left / right は少し離し、
# center は中央付近に置く
# =========================================================

SIDE_OFFSETS = {
    "left": (0.055, 0.110),
    "center": (-0.025, 0.025),
    "right": (-0.110, -0.055),
}


class ObstacleDatasetGeneratorV41(Node):

    def __init__(self):

        super().__init__(
            "obstacle_dataset_generator_v4_1"
        )

        self.fk_client = self.create_client(
            GetPositionFK,
            "/compute_fk",
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik",
        )

        self.validity_client = self.create_client(
            GetStateValidity,
            "/check_state_validity",
        )

        self.scene_client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

        self.plan_client = self.create_client(
            GetMotionPlan,
            "/plan_kinematic_path",
        )

        self.wait_for_services()

    # -----------------------------------------------------
    # MoveItサービス待機
    # -----------------------------------------------------

    def wait_for_services(self):

        services = [
            (
                self.fk_client,
                "/compute_fk",
            ),
            (
                self.ik_client,
                "/compute_ik",
            ),
            (
                self.validity_client,
                "/check_state_validity",
            ),
            (
                self.scene_client,
                "/apply_planning_scene",
            ),
            (
                self.plan_client,
                "/plan_kinematic_path",
            ),
        ]

        for client, name in services:

            self.get_logger().info(
                f"Waiting for {name} ..."
            )

            if not client.wait_for_service(
                timeout_sec=10.0
            ):
                raise RuntimeError(
                    f"{name} service not available"
                )

        self.get_logger().info(
            "All MoveIt services are ready."
        )

    # -----------------------------------------------------
    # RobotState生成
    # -----------------------------------------------------

    def make_robot_state(
        self,
        joint_values,
    ):

        state = RobotState()

        state.joint_state = JointState()

        state.joint_state.name = (
            ARM_JOINTS.copy()
        )

        state.joint_state.position = [
            float(
                joint_values[name]
            )
            for name in ARM_JOINTS
        ]

        return state

    # -----------------------------------------------------
    # FK
    # -----------------------------------------------------

    def compute_fk(
        self,
        joint_values,
    ):

        req = GetPositionFK.Request()

        req.header.frame_id = (
            BASE_FRAME
        )

        req.fk_link_names = [
            TIP_LINK
        ]

        req.robot_state = (
            self.make_robot_state(
                joint_values
            )
        )

        future = (
            self.fk_client.call_async(req)
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()

        if response is None:
            return None

        if (
            response.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            return None

        if not response.pose_stamped:
            return None

        return response.pose_stamped[0]

    # -----------------------------------------------------
    # IK
    # -----------------------------------------------------

    def compute_ik(
        self,
        target_pose,
        seed_joints,
    ):

        req = GetPositionIK.Request()

        req.ik_request.group_name = (
            GROUP_NAME
        )

        req.ik_request.ik_link_name = (
            TIP_LINK
        )

        req.ik_request.pose_stamped = (
            target_pose
        )

        req.ik_request.robot_state = (
            self.make_robot_state(
                seed_joints
            )
        )

        req.ik_request.avoid_collisions = (
            True
        )

        req.ik_request.timeout.sec = 1

        future = (
            self.ik_client.call_async(req)
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()

        if response is None:
            return None

        if (
            response.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            return None

        names = (
            response.solution
            .joint_state
            .name
        )

        positions = (
            response.solution
            .joint_state
            .position
        )

        result = dict(
            zip(
                names,
                positions,
            )
        )

        if not all(
            name in result
            for name in ARM_JOINTS
        ):
            return None

        return {
            name: float(
                result[name]
            )
            for name in ARM_JOINTS
        }

    # -----------------------------------------------------
    # 衝突判定
    # -----------------------------------------------------

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

        req.group_name = (
            GROUP_NAME
        )

        future = (
            self.validity_client
            .call_async(req)
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

    # -----------------------------------------------------
    # 障害物追加
    # -----------------------------------------------------

    def add_obstacle(
        self,
        obstacle_id,
        x,
        y,
        z,
    ):

        collision_object = (
            CollisionObject()
        )

        collision_object.id = (
            obstacle_id
        )

        collision_object.header.frame_id = (
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

        collision_object.primitives = [
            primitive
        ]

        collision_object.primitive_poses = [
            pose
        ]

        collision_object.operation = (
            CollisionObject.ADD
        )

        scene = PlanningScene()

        scene.is_diff = True

        scene.world.collision_objects = [
            collision_object
        ]

        req = (
            ApplyPlanningScene.Request()
        )

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

    # -----------------------------------------------------
    # 障害物削除
    # -----------------------------------------------------

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

        req = (
            ApplyPlanningScene.Request()
        )

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

    # -----------------------------------------------------
    # Goal constraints生成
    # -----------------------------------------------------

    def make_goal_constraints(
        self,
        goal_joints,
    ):

        constraints = Constraints()

        for name in ARM_JOINTS:

            jc = JointConstraint()

            jc.joint_name = name

            jc.position = float(
                goal_joints[name]
            )

            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01

            jc.weight = 1.0

            constraints.joint_constraints.append(
                jc
            )

        return constraints

    # -----------------------------------------------------
    # OMPL経路計画
    # -----------------------------------------------------

    def plan_path(
        self,
        start_joints,
        goal_joints,
    ):

        req = GetMotionPlan.Request()

        motion = (
            req.motion_plan_request
        )

        motion.group_name = (
            GROUP_NAME
        )

        motion.start_state = (
            self.make_robot_state(
                start_joints
            )
        )

        motion.goal_constraints = [
            self.make_goal_constraints(
                goal_joints
            )
        ]

        motion.allowed_planning_time = (
            3.0
        )

        motion.num_planning_attempts = (
            10
        )

        motion.pipeline_id = "ompl"

        future = (
            self.plan_client.call_async(
                req
            )
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()

        if response is None:
            return None

        motion_response = (
            response.motion_plan_response
        )

        if (
            motion_response.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            return None

        trajectory = (
            motion_response
            .trajectory
            .joint_trajectory
        )

        if len(
            trajectory.points
        ) < 2:
            return None

        return trajectory

    # -----------------------------------------------------
    # 軌道全点の衝突チェック
    # -----------------------------------------------------

    def trajectory_is_valid(
        self,
        trajectory,
    ):

        joint_index = {
            name: i
            for i, name in enumerate(
                trajectory.joint_names
            )
        }

        if not all(
            name in joint_index
            for name in ARM_JOINTS
        ):
            return False

        for point in trajectory.points:

            joints = {
                name:
                    point.positions[
                        joint_index[name]
                    ]
                for name in ARM_JOINTS
            }

            if not self.state_is_valid(
                joints
            ):
                return False

        return True


# =========================================================
# 軌道長計算
# =========================================================

def trajectory_length(
    trajectory,
):

    total = 0.0

    previous = None

    for point in trajectory.points:

        current = list(
            point.positions
        )

        if previous is not None:

            total += math.sqrt(
                sum(
                    (
                        a - b
                    ) ** 2
                    for a, b in zip(
                        current,
                        previous,
                    )
                )
            )

        previous = current

    return total


# =========================================================
# 障害物位置生成
#
# 開始EE位置と目標EE位置の
# XY平面上の線分に対して
# 左・中央・右へ配置
# =========================================================

def sample_obstacle_position(
    start_pose,
    goal_pose,
    obstacle_class,
):

    sx = (
        start_pose.pose.position.x
    )

    sy = (
        start_pose.pose.position.y
    )

    gx = (
        goal_pose.pose.position.x
    )

    gy = (
        goal_pose.pose.position.y
    )

    # 経路中央付近を選択
    t = random.uniform(
        0.35,
        0.65,
    )

    base_x = (
        sx
        + t * (gx - sx)
    )

    base_y = (
        sy
        + t * (gy - sy)
    )

    dx = gx - sx
    dy = gy - sy

    norm = math.sqrt(
        dx * dx
        + dy * dy
    )

    if norm < 1e-8:

        perp_x = 0.0
        perp_y = 1.0

    else:

        # XY平面で90度回転
        perp_x = (
            -dy / norm
        )

        perp_y = (
            dx / norm
        )

    low, high = (
        SIDE_OFFSETS[
            obstacle_class
        ]
    )

    offset = random.uniform(
        low,
        high,
    )

    x = (
        base_x
        + perp_x * offset
        + random.uniform(
            -0.01,
            0.01,
        )
    )

    y = (
        base_y
        + perp_y * offset
        + random.uniform(
            -0.01,
            0.01,
        )
    )

    z = OBSTACLE_Z

    return x, y, z


# =========================================================
# メイン
# =========================================================

def main():

    rclpy.init()

    node = (
        ObstacleDatasetGeneratorV41()
    )

    obstacle_id = (
        "training_obstacle_v4_1"
    )

    # -----------------------------------------
    # 障害物を消した状態にする
    # -----------------------------------------

    node.remove_obstacle(
        obstacle_id
    )

    # -----------------------------------------
    # 固定開始姿勢確認
    # -----------------------------------------

    if not node.state_is_valid(
        START_JOINTS
    ):

        node.get_logger().error(
            "START_JOINTS is invalid."
        )

        node.destroy_node()

        rclpy.shutdown()

        sys.exit(1)

    start_pose = node.compute_fk(
        START_JOINTS
    )

    if start_pose is None:

        node.get_logger().error(
            "Start FK failed."
        )

        node.destroy_node()

        rclpy.shutdown()

        sys.exit(1)

    # -----------------------------------------
    # 固定目標を自動選択
    # -----------------------------------------

    goal_joints = None
    goal_pose = None

    for candidate in GOAL_CANDIDATES:

        if not node.state_is_valid(
            candidate
        ):
            continue

        pose = node.compute_fk(
            candidate
        )

        if pose is None:
            continue

        solution = node.compute_ik(
            pose,
            candidate,
        )

        if solution is None:
            continue

        if not node.state_is_valid(
            solution
        ):
            continue

        # 障害物なしで
        # start→goalが計画可能か確認
        trajectory = (
            node.plan_path(
                START_JOINTS,
                solution,
            )
        )

        if trajectory is None:
            continue

        goal_joints = solution
        goal_pose = pose

        break

    if goal_joints is None:

        node.get_logger().error(
            "No valid goal candidate found."
        )

        node.destroy_node()

        rclpy.shutdown()

        sys.exit(1)

    # -----------------------------------------
    # 基準経路（障害物なし）
    # -----------------------------------------

    baseline_trajectory = (
        node.plan_path(
            START_JOINTS,
            goal_joints,
        )
    )

    if baseline_trajectory is None:

        node.get_logger().error(
            "Baseline planning failed."
        )

        sys.exit(1)

    baseline_length = (
        trajectory_length(
            baseline_trajectory
        )
    )

    sp = start_pose.pose.position
    gp = goal_pose.pose.position

    node.get_logger().info(
        "Fixed start and goal selected."
    )

    node.get_logger().info(
        f"Start XYZ = "
        f"({sp.x:.3f}, "
        f"{sp.y:.3f}, "
        f"{sp.z:.3f})"
    )

    node.get_logger().info(
        f"Goal XYZ = "
        f"({gp.x:.3f}, "
        f"{gp.y:.3f}, "
        f"{gp.z:.3f})"
    )

    node.get_logger().info(
        f"Baseline path length = "
        f"{baseline_length:.3f}"
    )

    # -----------------------------------------
    # CSV保存先
    # -----------------------------------------

    output_path = (
        Path.home()
        / "so101_ws"
        / "obstacle_dataset_v4_1.csv"
    )

    summary_path = (
        Path.home()
        / "so101_ws"
        / "obstacle_dataset_v4_1_summary.csv"
    )

    rows = []
    summary_rows = []

    episode_id = 0

    total_endpoint_reject = 0
    total_plan_fail = 0
    total_trajectory_invalid = 0

    # =========================================
    # 左・中央・右を順番に10件ずつ
    # =========================================

    for obstacle_class in CLASSES:

        success_count = 0
        attempt_count = 0

        node.get_logger().info(
            "--------------------------------"
        )

        node.get_logger().info(
            f"Generating class: "
            f"{obstacle_class}"
        )

        while (
            success_count
            < EPISODES_PER_CLASS
            and
            attempt_count
            < MAX_ATTEMPTS_PER_CLASS
        ):

            attempt_count += 1

            # ---------------------------------
            # 障害物位置生成
            # ---------------------------------

            obs_x, obs_y, obs_z = (
                sample_obstacle_position(
                    start_pose,
                    goal_pose,
                    obstacle_class,
                )
            )

            # 古い障害物削除
            node.remove_obstacle(
                obstacle_id
            )

            # 新しい障害物追加
            if not node.add_obstacle(
                obstacle_id,
                obs_x,
                obs_y,
                obs_z,
            ):
                continue

            # ---------------------------------
            # 開始・目標の衝突確認
            # ---------------------------------

            start_valid = (
                node.state_is_valid(
                    START_JOINTS
                )
            )

            goal_valid = (
                node.state_is_valid(
                    goal_joints
                )
            )

            if not (
                start_valid
                and goal_valid
            ):

                total_endpoint_reject += 1

                continue

            # ---------------------------------
            # 経路計画
            # ---------------------------------

            trajectory = (
                node.plan_path(
                    START_JOINTS,
                    goal_joints,
                )
            )

            if trajectory is None:

                total_plan_fail += 1

                continue

            # ---------------------------------
            # 軌道全点の妥当性再確認
            # ---------------------------------

            if not node.trajectory_is_valid(
                trajectory
            ):

                total_trajectory_invalid += 1

                continue

            # ---------------------------------
            # 成功
            # ---------------------------------

            episode_id += 1
            success_count += 1

            path_length = (
                trajectory_length(
                    trajectory
                )
            )

            path_delta = (
                path_length
                - baseline_length
            )

            joint_index = {
                name: i
                for i, name in enumerate(
                    trajectory.joint_names
                )
            }

            # ---------------------------------
            # 軌道点保存
            # ---------------------------------

            for step_index, point in enumerate(
                trajectory.points
            ):

                sec = (
                    point.time_from_start.sec
                )

                nanosec = (
                    point.time_from_start.nanosec
                )

                t = (
                    float(sec)
                    + float(nanosec)
                    * 1e-9
                )

                row = {
                    "episode_id":
                        episode_id,

                    "obstacle_class":
                        obstacle_class,

                    "step":
                        step_index,

                    "time_from_start":
                        t,

                    "obs_x":
                        obs_x,

                    "obs_y":
                        obs_y,

                    "obs_z":
                        obs_z,

                    "obs_size_x":
                        OBSTACLE_SIZE_X,

                    "obs_size_y":
                        OBSTACLE_SIZE_Y,

                    "obs_size_z":
                        OBSTACLE_SIZE_Z,

                    "start_x":
                        sp.x,

                    "start_y":
                        sp.y,

                    "start_z":
                        sp.z,

                    "target_x":
                        gp.x,

                    "target_y":
                        gp.y,

                    "target_z":
                        gp.z,

                    "baseline_path_length":
                        baseline_length,

                    "path_length":
                        path_length,

                    "path_delta":
                        path_delta,

                    "shoulder_pan":
                        point.positions[
                            joint_index[
                                "shoulder_pan"
                            ]
                        ],

                    "shoulder_lift":
                        point.positions[
                            joint_index[
                                "shoulder_lift"
                            ]
                        ],

                    "elbow_flex":
                        point.positions[
                            joint_index[
                                "elbow_flex"
                            ]
                        ],

                    "wrist_flex":
                        point.positions[
                            joint_index[
                                "wrist_flex"
                            ]
                        ],

                    "wrist_roll":
                        point.positions[
                            joint_index[
                                "wrist_roll"
                            ]
                        ],
                }

                rows.append(row)

            summary_rows.append(
                {
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

                    "trajectory_points":
                        len(
                            trajectory.points
                        ),

                    "baseline_path_length":
                        baseline_length,

                    "path_length":
                        path_length,

                    "path_delta":
                        path_delta,
                }
            )

            node.get_logger().info(
                f"{obstacle_class}: "
                f"{success_count}/"
                f"{EPISODES_PER_CLASS} "
                f"(attempt "
                f"{attempt_count}) "
                f"points="
                f"{len(trajectory.points)} "
                f"path="
                f"{path_length:.3f} "
                f"delta="
                f"{path_delta:.3f} "
                f"obstacle=("
                f"{obs_x:.3f}, "
                f"{obs_y:.3f}, "
                f"{obs_z:.3f})"
            )

        # -------------------------------------
        # クラス未達時は警告
        # -------------------------------------

        if (
            success_count
            < EPISODES_PER_CLASS
        ):

            node.get_logger().warning(
                f"Class {obstacle_class} "
                f"only generated "
                f"{success_count}/"
                f"{EPISODES_PER_CLASS}"
            )

    # =========================================
    # 障害物削除
    # =========================================

    node.remove_obstacle(
        obstacle_id
    )

    # =========================================
    # CSV保存
    # =========================================

    fieldnames = [
        "episode_id",
        "obstacle_class",
        "step",
        "time_from_start",

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
        "path_length",
        "path_delta",

        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
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
        writer.writerows(rows)

    summary_fieldnames = [
        "episode_id",
        "obstacle_class",
        "obs_x",
        "obs_y",
        "obs_z",
        "trajectory_points",
        "baseline_path_length",
        "path_length",
        "path_delta",
    ]

    with summary_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # =========================================
    # 集計
    # =========================================

    class_counts = {
        name: 0
        for name in CLASSES
    }

    for row in summary_rows:

        class_counts[
            row[
                "obstacle_class"
            ]
        ] += 1

    node.get_logger().info(
        "================================"
    )

    node.get_logger().info(
        "===== V4.1 SUMMARY ====="
    )

    node.get_logger().info(
        f"Total episodes: "
        f"{episode_id}"
    )

    node.get_logger().info(
        f"Left: "
        f"{class_counts['left']}"
    )

    node.get_logger().info(
        f"Center: "
        f"{class_counts['center']}"
    )

    node.get_logger().info(
        f"Right: "
        f"{class_counts['right']}"
    )

    node.get_logger().info(
        f"Trajectory rows: "
        f"{len(rows)}"
    )

    node.get_logger().info(
        f"Endpoint rejected: "
        f"{total_endpoint_reject}"
    )

    node.get_logger().info(
        f"Planning failures: "
        f"{total_plan_fail}"
    )

    node.get_logger().info(
        f"Trajectory invalid: "
        f"{total_trajectory_invalid}"
    )

    node.get_logger().info(
        f"Dataset: "
        f"{output_path}"
    )

    node.get_logger().info(
        f"Summary: "
        f"{summary_path}"
    )

    node.destroy_node()

    rclpy.shutdown()

    # 30件そろわなければ
    # 終了コード1
    if episode_id < 30:
        sys.exit(1)


if __name__ == "__main__":
    main()
