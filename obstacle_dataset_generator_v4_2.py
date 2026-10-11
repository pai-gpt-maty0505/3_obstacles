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


# ============================================================
# 基本設定
# ============================================================

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

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================
# 固定開始姿勢
# ============================================================

START_JOINTS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
}


# ============================================================
# 固定目標姿勢候補
#
# 上から順に、
#   1. 状態が有効
#   2. FK成功
#   3. IK成功
#   4. 障害物なしで経路計画成功
# したものを使用
# ============================================================

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


# ============================================================
# 障害物サイズ
# ============================================================

OBSTACLE_SIZE_X = 0.055
OBSTACLE_SIZE_Y = 0.055
OBSTACLE_SIZE_Z = 0.22

OBSTACLE_Z = OBSTACLE_SIZE_Z / 2.0


# ============================================================
# データ生成設定
# ============================================================

CLASSES = [
    "left",
    "center",
    "right",
]

EPISODES_PER_CLASS = 100

MAX_ATTEMPTS_PER_CLASS = 2000


# ============================================================
# 障害物の横方向オフセット
#
# 基準経路付近に置く。
# v4.1 より経路に近づけ、
# 実際に基準経路を塞ぐ候補を探す。
# ============================================================

CLASS_OFFSETS = {
    "left": (0.015, 0.040),
    "center": (-0.010, 0.010),
    "right": (-0.040, -0.015),
}


# ============================================================
# 「回避経路が基準経路と十分違う」と判定する閾値
#
# RMS joint-space distance [rad]
# ============================================================

MIN_PATH_RMS_DIFFERENCE = 0.04


class ObstacleDatasetGeneratorV42(Node):

    def __init__(self):

        super().__init__(
            "obstacle_dataset_generator_v4_2"
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

    # ========================================================
    # MoveItサービス待機
    # ========================================================

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

    # ========================================================
    # RobotState生成
    # ========================================================

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
            float(joint_values[name])
            for name in ARM_JOINTS
        ]

        return state

    # ========================================================
    # FK
    # ========================================================

    def compute_fk(
        self,
        joint_values,
    ):

        req = GetPositionFK.Request()

        req.header.frame_id = BASE_FRAME

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

    # ========================================================
    # IK
    # ========================================================

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

        req.ik_request.avoid_collisions = True

        req.ik_request.timeout.sec = 1
        req.ik_request.timeout.nanosec = 0

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

        result = dict(
            zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position,
            )
        )

        if not all(
            joint in result
            for joint in ARM_JOINTS
        ):
            return None

        return {
            joint: float(
                result[joint]
            )
            for joint in ARM_JOINTS
        }

    # ========================================================
    # 状態の衝突判定
    # ========================================================

    def state_is_valid(
        self,
        joint_values,
    ):

        req = GetStateValidity.Request()

        req.robot_state = (
            self.make_robot_state(
                joint_values
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

    # ========================================================
    # 障害物追加
    # ========================================================

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

        collision_object.id = obstacle_id

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

    # ========================================================
    # 障害物削除
    # ========================================================

    def remove_obstacle(
        self,
        obstacle_id,
    ):

        collision_object = (
            CollisionObject()
        )

        collision_object.id = obstacle_id

        collision_object.header.frame_id = (
            BASE_FRAME
        )

        collision_object.operation = (
            CollisionObject.REMOVE
        )

        scene = PlanningScene()

        scene.is_diff = True

        scene.world.collision_objects = [
            collision_object
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

    # ========================================================
    # Goal Constraints
    # ========================================================

    def make_goal_constraints(
        self,
        goal_joints,
    ):

        constraints = Constraints()

        for joint_name in ARM_JOINTS:

            jc = JointConstraint()

            jc.joint_name = joint_name

            jc.position = float(
                goal_joints[
                    joint_name
                ]
            )

            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01

            jc.weight = 1.0

            constraints.joint_constraints.append(
                jc
            )

        return constraints

    # ========================================================
    # 経路計画
    # ========================================================

    def plan_path(
        self,
        start_joints,
        goal_joints,
    ):

        req = GetMotionPlan.Request()

        motion = (
            req.motion_plan_request
        )

        motion.group_name = GROUP_NAME

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

        motion.allowed_planning_time = 3.0

        motion.num_planning_attempts = 10

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

    # ========================================================
    # trajectory point → joint dictionary
    # ========================================================

    def point_to_joint_dict(
        self,
        trajectory,
        point,
    ):

        index = {
            name: i
            for i, name in enumerate(
                trajectory.joint_names
            )
        }

        if not all(
            name in index
            for name in ARM_JOINTS
        ):
            return None

        return {
            name: float(
                point.positions[
                    index[name]
                ]
            )
            for name in ARM_JOINTS
        }

    # ========================================================
    # 軌道全点が有効か
    # ========================================================

    def trajectory_is_valid(
        self,
        trajectory,
    ):

        for point in trajectory.points:

            joints = (
                self.point_to_joint_dict(
                    trajectory,
                    point,
                )
            )

            if joints is None:
                return False

            if not self.state_is_valid(
                joints
            ):
                return False

        return True

    # ========================================================
    # 基準軌道が障害物で塞がれたか
    #
    # 戻り値：
    #   blocked
    #   collision_point_count
    # ========================================================

    def baseline_is_blocked(
        self,
        baseline_trajectory,
    ):

        collision_count = 0

        for point in (
            baseline_trajectory.points
        ):

            joints = (
                self.point_to_joint_dict(
                    baseline_trajectory,
                    point,
                )
            )

            if joints is None:
                continue

            if not self.state_is_valid(
                joints
            ):
                collision_count += 1

        blocked = (
            collision_count > 0
        )

        return (
            blocked,
            collision_count,
        )


# ============================================================
# 軌道長
# ============================================================

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
                    (a - b) ** 2
                    for a, b in zip(
                        current,
                        previous,
                    )
                )
            )

        previous = current

    return total


# ============================================================
# 軌道を等間隔に補間
# ============================================================

def interpolate_trajectory(
    trajectory,
    samples=30,
):

    points = [
        list(point.positions)
        for point in trajectory.points
    ]

    if len(points) < 2:
        return None

    result = []

    n_segments = len(points) - 1

    for i in range(samples):

        alpha = (
            i
            / float(
                samples - 1
            )
        )

        segment_float = (
            alpha * n_segments
        )

        segment = min(
            int(segment_float),
            n_segments - 1,
        )

        local_alpha = (
            segment_float
            - segment
        )

        p0 = points[segment]
        p1 = points[segment + 1]

        q = [
            (
                a
                + local_alpha
                * (b - a)
            )
            for a, b in zip(
                p0,
                p1,
            )
        ]

        result.append(q)

    return result


# ============================================================
# 基準経路と回避経路のRMS差
# ============================================================

def trajectory_rms_difference(
    baseline,
    avoidance,
):

    a = interpolate_trajectory(
        baseline,
        samples=30,
    )

    b = interpolate_trajectory(
        avoidance,
        samples=30,
    )

    if a is None or b is None:
        return 0.0

    total = 0.0
    count = 0

    for qa, qb in zip(a, b):

        for va, vb in zip(
            qa,
            qb,
        ):

            total += (
                va - vb
            ) ** 2

            count += 1

    if count == 0:
        return 0.0

    return math.sqrt(
        total / count
    )


# ============================================================
# baseline trajectoryの中間点を使って
# 障害物中心候補を決定
# ============================================================

def choose_baseline_anchor(
    node,
    baseline_trajectory,
):

    # 中央30〜70%付近から選択
    n = len(
        baseline_trajectory.points
    )

    if n < 3:
        return None

    low = max(
        1,
        int(
            0.30 * (n - 1)
        ),
    )

    high = min(
        n - 2,
        int(
            0.70 * (n - 1)
        ),
    )

    index = random.randint(
        low,
        high,
    )

    point = (
        baseline_trajectory
        .points[index]
    )

    joints = (
        node.point_to_joint_dict(
            baseline_trajectory,
            point,
        )
    )

    if joints is None:
        return None

    pose = node.compute_fk(
        joints
    )

    if pose is None:
        return None

    return pose


# ============================================================
# 障害物位置生成
# ============================================================

def sample_obstacle_position(
    node,
    baseline_trajectory,
    obstacle_class,
):

    anchor_pose = (
        choose_baseline_anchor(
            node,
            baseline_trajectory,
        )
    )

    if anchor_pose is None:
        return None

    p = (
        anchor_pose.pose.position
    )

    # 基準経路のEE位置を基準に
    # left / center / right をY方向へずらす
    low, high = (
        CLASS_OFFSETS[
            obstacle_class
        ]
    )

    offset = random.uniform(
        low,
        high,
    )

    x = (
        p.x
        + random.uniform(
            -0.015,
            0.015,
        )
    )

    y = (
        p.y
        + offset
    )

    z = OBSTACLE_Z

    return (
        x,
        y,
        z,
    )


# ============================================================
# Main
# ============================================================

def main():

    rclpy.init()

    node = (
        ObstacleDatasetGeneratorV42()
    )

    obstacle_id = (
        "training_obstacle_v4_2"
    )

    # ========================================================
    # 障害物削除
    # ========================================================

    node.remove_obstacle(
        obstacle_id
    )

    # ========================================================
    # Start確認
    # ========================================================

    if not node.state_is_valid(
        START_JOINTS
    ):

        node.get_logger().error(
            "START_JOINTS invalid."
        )

        sys.exit(1)

    start_pose = node.compute_fk(
        START_JOINTS
    )

    if start_pose is None:

        node.get_logger().error(
            "Start FK failed."
        )

        sys.exit(1)

    # ========================================================
    # 固定Goal選択
    # ========================================================

    goal_joints = None
    goal_pose = None

    for candidate in (
        GOAL_CANDIDATES
    ):

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

        trajectory = node.plan_path(
            START_JOINTS,
            solution,
        )

        if trajectory is None:
            continue

        goal_joints = solution
        goal_pose = pose

        break

    if goal_joints is None:

        node.get_logger().error(
            "No valid goal found."
        )

        sys.exit(1)

    # ========================================================
    # Baseline trajectory
    # ========================================================

    node.remove_obstacle(
        obstacle_id
    )

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
        "Fixed start / goal selected."
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
        f"Baseline points = "
        f"{len(baseline_trajectory.points)}"
    )

    node.get_logger().info(
        f"Baseline path length = "
        f"{baseline_length:.3f}"
    )

    # ========================================================
    # 出力
    # ========================================================

    output_path = (
        Path.home()
        / "so101_ws"
        / "obstacle_dataset_v4_2.csv"
    )

    summary_path = (
        Path.home()
        / "so101_ws"
        / "obstacle_dataset_v4_2_summary.csv"
    )

    rows = []
    summary_rows = []

    episode_id = 0

    counters = {
        "endpoint_rejected": 0,
        "baseline_not_blocked": 0,
        "planning_failed": 0,
        "trajectory_invalid": 0,
        "path_too_similar": 0,
    }

    # ========================================================
    # left / center / right
    # ========================================================

    for obstacle_class in CLASSES:

        success_count = 0
        attempt_count = 0

        node.get_logger().info(
            "================================"
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

            candidate = (
                sample_obstacle_position(
                    node,
                    baseline_trajectory,
                    obstacle_class,
                )
            )

            if candidate is None:
                continue

            (
                obs_x,
                obs_y,
                obs_z,
            ) = candidate

            # -----------------------------------------------
            # 障害物更新
            # -----------------------------------------------

            node.remove_obstacle(
                obstacle_id
            )

            if not node.add_obstacle(
                obstacle_id,
                obs_x,
                obs_y,
                obs_z,
            ):
                continue

            # -----------------------------------------------
            # start / goalが衝突していないこと
            # -----------------------------------------------

            if (
                not node.state_is_valid(
                    START_JOINTS
                )
                or
                not node.state_is_valid(
                    goal_joints
                )
            ):

                counters[
                    "endpoint_rejected"
                ] += 1

                continue

            # -----------------------------------------------
            # ★重要：
            # 元の基準軌道が本当に塞がれたか
            # -----------------------------------------------

            (
                blocked,
                blocked_points,
            ) = node.baseline_is_blocked(
                baseline_trajectory
            )

            if not blocked:

                counters[
                    "baseline_not_blocked"
                ] += 1

                continue

            # -----------------------------------------------
            # 障害物回避経路を再計画
            # -----------------------------------------------

            avoidance_trajectory = (
                node.plan_path(
                    START_JOINTS,
                    goal_joints,
                )
            )

            if avoidance_trajectory is None:

                counters[
                    "planning_failed"
                ] += 1

                continue

            # -----------------------------------------------
            # 新しい軌道自体が安全か
            # -----------------------------------------------

            if not node.trajectory_is_valid(
                avoidance_trajectory
            ):

                counters[
                    "trajectory_invalid"
                ] += 1

                continue

            # -----------------------------------------------
            # 基準経路と十分違うか
            # -----------------------------------------------

            rms_difference = (
                trajectory_rms_difference(
                    baseline_trajectory,
                    avoidance_trajectory,
                )
            )

            if (
                rms_difference
                < MIN_PATH_RMS_DIFFERENCE
            ):

                counters[
                    "path_too_similar"
                ] += 1

                continue

            # -----------------------------------------------
            # 成功
            # -----------------------------------------------

            episode_id += 1
            success_count += 1

            avoidance_length = (
                trajectory_length(
                    avoidance_trajectory
                )
            )

            path_delta = (
                avoidance_length
                - baseline_length
            )

            joint_index = {
                name: i
                for i, name in enumerate(
                    avoidance_trajectory
                    .joint_names
                )
            }

            # -----------------------------------------------
            # 軌道CSV
            # -----------------------------------------------

            for (
                step_index,
                point,
            ) in enumerate(
                avoidance_trajectory.points
            ):

                time_from_start = (
                    float(
                        point
                        .time_from_start
                        .sec
                    )
                    +
                    float(
                        point
                        .time_from_start
                        .nanosec
                    )
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
                        time_from_start,

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

                    "avoidance_path_length":
                        avoidance_length,

                    "path_delta":
                        path_delta,

                    "path_rms_difference":
                        rms_difference,

                    "baseline_blocked_points":
                        blocked_points,

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

            # -----------------------------------------------
            # Summary
            # -----------------------------------------------

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
                            avoidance_trajectory
                            .points
                        ),

                    "baseline_path_length":
                        baseline_length,

                    "avoidance_path_length":
                        avoidance_length,

                    "path_delta":
                        path_delta,

                    "path_rms_difference":
                        rms_difference,

                    "baseline_blocked_points":
                        blocked_points,
                }
            )

            node.get_logger().info(
                f"{obstacle_class}: "
                f"{success_count}/"
                f"{EPISODES_PER_CLASS} "
                f"(attempt "
                f"{attempt_count}) "
                f"blocked_points="
                f"{blocked_points} "
                f"rms="
                f"{rms_difference:.3f} "
                f"baseline="
                f"{baseline_length:.3f} "
                f"avoid="
                f"{avoidance_length:.3f} "
                f"delta="
                f"{path_delta:.3f} "
                f"obstacle=("
                f"{obs_x:.3f}, "
                f"{obs_y:.3f}, "
                f"{obs_z:.3f})"
            )

        if (
            success_count
            < EPISODES_PER_CLASS
        ):

            node.get_logger().warning(
                f"{obstacle_class}: "
                f"only generated "
                f"{success_count}/"
                f"{EPISODES_PER_CLASS}"
            )

    # ========================================================
    # 障害物削除
    # ========================================================

    node.remove_obstacle(
        obstacle_id
    )

    # ========================================================
    # Dataset CSV
    # ========================================================

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

    with output_path.open(
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

    # ========================================================
    # Summary CSV
    # ========================================================

    summary_fieldnames = [
        "episode_id",
        "obstacle_class",

        "obs_x",
        "obs_y",
        "obs_z",

        "trajectory_points",

        "baseline_path_length",
        "avoidance_path_length",
        "path_delta",

        "path_rms_difference",

        "baseline_blocked_points",
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

    # ========================================================
    # 集計
    # ========================================================

    class_counts = {
        class_name: 0
        for class_name in CLASSES
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
        "===== V4.2 SUMMARY ====="
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
        f"{counters['endpoint_rejected']}"
    )

    node.get_logger().info(
        f"Baseline not blocked: "
        f"{counters['baseline_not_blocked']}"
    )

    node.get_logger().info(
        f"Planning failures: "
        f"{counters['planning_failed']}"
    )

    node.get_logger().info(
        f"Trajectory invalid: "
        f"{counters['trajectory_invalid']}"
    )

    node.get_logger().info(
        f"Path too similar: "
        f"{counters['path_too_similar']}"
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

    if episode_id < (
        len(CLASSES)
        * EPISODES_PER_CLASS
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
