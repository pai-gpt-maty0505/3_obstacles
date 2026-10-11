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


ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

TIP_LINK = "gripper_link"
BASE_FRAME = "base_link"
GROUP_NAME = "arm"

# 関節角度のサンプリング範囲
JOINT_LIMITS = {
    "shoulder_pan": (-1.2, 1.2),
    "shoulder_lift": (-1.0, 1.0),
    "elbow_flex": (-1.0, 1.0),
    "wrist_flex": (-0.9, 0.9),
    "wrist_roll": (-1.5, 1.5),
}

# 使用する作業空間
WORKSPACE = {
    "x_min": 0.08,
    "x_max": 0.36,
    "y_min": -0.26,
    "y_max": 0.26,
    "z_min": 0.06,
    "z_max": 0.40,
}

# 障害物サイズ
OBSTACLE_SIZE_X = 0.05
OBSTACLE_SIZE_Y = 0.05
OBSTACLE_SIZE_Z = 0.18

# 障害物配置範囲
OBSTACLE_RANGE = {
    "x_min": 0.12,
    "x_max": 0.26,
    "y_min": -0.16,
    "y_max": 0.16,
    "z": OBSTACLE_SIZE_Z / 2.0,
}


class ObstacleDatasetGenerator(Node):

    def __init__(self):
        super().__init__("obstacle_dataset_generator")

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

    def random_joint_configuration(self):

        return {
            name: random.uniform(
                JOINT_LIMITS[name][0],
                JOINT_LIMITS[name][1],
            )
            for name in ARM_JOINTS
        }

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

    def pose_in_workspace(
        self,
        pose_stamped,
    ):

        p = pose_stamped.pose.position

        return (
            WORKSPACE["x_min"]
            <= p.x
            <= WORKSPACE["x_max"]
            and
            WORKSPACE["y_min"]
            <= p.y
            <= WORKSPACE["y_max"]
            and
            WORKSPACE["z_min"]
            <= p.z
            <= WORKSPACE["z_max"]
        )

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

        # 障害物・自己衝突を考慮
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
            name in result
            for name in ARM_JOINTS
        ):
            return None

        return {
            name: float(result[name])
            for name in ARM_JOINTS
        }

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
            self.validity_client.call_async(req)
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()

        if response is None:
            return False

        return bool(response.valid)

    def add_obstacle(
        self,
        obstacle_id,
        x,
        y,
        z,
    ):

        collision_object = CollisionObject()

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
            self.scene_client.call_async(req)
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()

        if response is None:
            return False

        return bool(response.success)

    def remove_obstacle(
        self,
        obstacle_id,
    ):

        collision_object = CollisionObject()

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
            self.scene_client.call_async(req)
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

    def make_goal_constraints(
        self,
        goal_joints,
    ):

        constraints = Constraints()

        for joint_name in ARM_JOINTS:

            jc = JointConstraint()

            jc.joint_name = joint_name

            jc.position = float(
                goal_joints[joint_name]
            )

            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01

            jc.weight = 1.0

            constraints.joint_constraints.append(
                jc
            )

        return constraints

    def plan_path(
        self,
        start_joints,
        goal_joints,
    ):

        req = GetMotionPlan.Request()

        motion_request = (
            req.motion_plan_request
        )

        motion_request.group_name = (
            GROUP_NAME
        )

        motion_request.start_state = (
            self.make_robot_state(
                start_joints
            )
        )

        motion_request.goal_constraints = [
            self.make_goal_constraints(
                goal_joints
            )
        ]

        motion_request.allowed_planning_time = (
            2.0
        )

        motion_request.num_planning_attempts = (
            5
        )

        motion_request.pipeline_id = "ompl"

        future = (
            self.plan_client.call_async(req)
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
            motion_response.trajectory
            .joint_trajectory
        )

        if len(trajectory.points) < 2:
            return None

        return trajectory

    def sample_obstacle(self):

        x = random.uniform(
            OBSTACLE_RANGE["x_min"],
            OBSTACLE_RANGE["x_max"],
        )

        y = random.uniform(
            OBSTACLE_RANGE["y_min"],
            OBSTACLE_RANGE["y_max"],
        )

        z = OBSTACLE_RANGE["z"]

        return x, y, z


def main():

    rclpy.init()

    node = ObstacleDatasetGenerator()

    # 最初は10エピソード
    target_episodes = 10

    max_attempts = 500

    output_path = (
        Path.home()
        / "so101_ws"
        / "obstacle_dataset_v3.csv"
    )

    rows = []

    episode_count = 0
    attempt_count = 0

    workspace_reject = 0
    invalid_state_reject = 0
    ik_fail = 0
    plan_fail = 0

    obstacle_id = "training_obstacle"

    while (
        episode_count < target_episodes
        and attempt_count < max_attempts
    ):

        attempt_count += 1

        obs_x, obs_y, obs_z = (
            node.sample_obstacle()
        )

        if not node.add_obstacle(
            obstacle_id,
            obs_x,
            obs_y,
            obs_z,
        ):
            node.get_logger().warning(
                "Failed to add obstacle"
            )
            continue

        start_joints = (
            node.random_joint_configuration()
        )

        if not node.state_is_valid(
            start_joints
        ):
            invalid_state_reject += 1
            continue

        goal_seed = (
            node.random_joint_configuration()
        )

        target_pose = (
            node.compute_fk(
                goal_seed
            )
        )

        if target_pose is None:
            continue

        if not node.pose_in_workspace(
            target_pose
        ):
            workspace_reject += 1
            continue

        goal_joints = (
            node.compute_ik(
                target_pose,
                goal_seed,
            )
        )

        if goal_joints is None:
            ik_fail += 1
            continue

        if not node.state_is_valid(
            goal_joints
        ):
            invalid_state_reject += 1
            continue

        trajectory = (
            node.plan_path(
                start_joints,
                goal_joints,
            )
        )

        if trajectory is None:
            plan_fail += 1
            continue

        episode_count += 1

        pose = target_pose.pose

        joint_index = {
            name: i
            for i, name in enumerate(
                trajectory.joint_names
            )
        }

        for step_index, point in enumerate(
            trajectory.points
        ):

            # 必要な関節が全部含まれているか
            if not all(
                name in joint_index
                for name in ARM_JOINTS
            ):
                continue

            sec = (
                point.time_from_start.sec
            )

            nanosec = (
                point.time_from_start.nanosec
            )

            time_from_start = (
                float(sec)
                + float(nanosec) * 1e-9
            )

            row = {
                "episode_id":
                    episode_count,

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

                "target_x":
                    pose.position.x,

                "target_y":
                    pose.position.y,

                "target_z":
                    pose.position.z,

                "target_qx":
                    pose.orientation.x,

                "target_qy":
                    pose.orientation.y,

                "target_qz":
                    pose.orientation.z,

                "target_qw":
                    pose.orientation.w,

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

        node.get_logger().info(
            f"Episode "
            f"{episode_count}/"
            f"{target_episodes} "
            f"success "
            f"(attempt "
            f"{attempt_count}, "
            f"points="
            f"{len(trajectory.points)}) "
            f"obstacle=("
            f"{obs_x:.3f}, "
            f"{obs_y:.3f}, "
            f"{obs_z:.3f})"
        )

    # 最後に障害物を消す
    node.remove_obstacle(
        obstacle_id
    )

    fieldnames = [
        "episode_id",
        "step",
        "time_from_start",

        "obs_x",
        "obs_y",
        "obs_z",

        "obs_size_x",
        "obs_size_y",
        "obs_size_z",

        "target_x",
        "target_y",
        "target_z",

        "target_qx",
        "target_qy",
        "target_qz",
        "target_qw",

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

    node.get_logger().info(
        "===== V3 summary ====="
    )

    node.get_logger().info(
        f"Attempts: "
        f"{attempt_count}"
    )

    node.get_logger().info(
        f"Episodes: "
        f"{episode_count}"
    )

    node.get_logger().info(
        f"Trajectory rows: "
        f"{len(rows)}"
    )

    node.get_logger().info(
        f"Workspace rejected: "
        f"{workspace_reject}"
    )

    node.get_logger().info(
        f"Invalid state rejected: "
        f"{invalid_state_reject}"
    )

    node.get_logger().info(
        f"IK failures: "
        f"{ik_fail}"
    )

    node.get_logger().info(
        f"Planning failures: "
        f"{plan_fail}"
    )

    node.get_logger().info(
        f"Saved dataset: "
        f"{output_path}"
    )

    node.destroy_node()

    rclpy.shutdown()

    if episode_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
