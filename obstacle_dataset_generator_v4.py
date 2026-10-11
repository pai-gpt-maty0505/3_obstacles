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

BASE_FRAME = "base_link"
TIP_LINK = "gripper_link"
GROUP_NAME = "arm"

# 開始姿勢は全エピソード共通
START_JOINTS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
}

# 目標姿勢候補。
# 有効なものを上から自動選択する。
GOAL_CANDIDATES = [
    {
        "shoulder_pan": 0.60,
        "shoulder_lift": -0.60,
        "elbow_flex": 0.70,
        "wrist_flex": -0.30,
        "wrist_roll": 0.40,
    },
    {
        "shoulder_pan": 0.30,
        "shoulder_lift": -0.80,
        "elbow_flex": 0.60,
        "wrist_flex": -0.30,
        "wrist_roll": 0.50,
    },
    {
        "shoulder_pan": -0.55,
        "shoulder_lift": -0.55,
        "elbow_flex": 0.65,
        "wrist_flex": -0.25,
        "wrist_roll": -0.35,
    },
]

# 障害物
OBSTACLE_SIZE_X = 0.06
OBSTACLE_SIZE_Y = 0.06
OBSTACLE_SIZE_Z = 0.30

EPISODES_PER_CLASS = 10

# 障害物のY方向相対位置
# midpoint_y を基準に左・中央・右へ配置
OBSTACLE_Y_OFFSETS = {
    "left": (-0.12, -0.07),
    "center": (-0.025, 0.025),
    "right": (0.07, 0.12),
}


class Generator(Node):

    def __init__(self):
        super().__init__(
            "obstacle_dataset_generator_v4"
        )

        self.fk_client = self.create_client(
            GetPositionFK,
            "/compute_fk",
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik",
        )

        self.valid_client = self.create_client(
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

        services = [
            (self.fk_client, "/compute_fk"),
            (self.ik_client, "/compute_ik"),
            (
                self.valid_client,
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
                    f"{name} not available"
                )

        self.get_logger().info(
            "All MoveIt services are ready."
        )

    def make_state(self, joints):

        state = RobotState()

        state.joint_state = JointState()

        state.joint_state.name = (
            ARM_JOINTS.copy()
        )

        state.joint_state.position = [
            float(joints[name])
            for name in ARM_JOINTS
        ]

        return state

    def fk(self, joints):

        req = GetPositionFK.Request()

        req.header.frame_id = BASE_FRAME
        req.fk_link_names = [TIP_LINK]

        req.robot_state = (
            self.make_state(joints)
        )

        future = self.fk_client.call_async(
            req
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

    def state_valid(self, joints):

        req = GetStateValidity.Request()

        req.robot_state = (
            self.make_state(joints)
        )

        req.group_name = GROUP_NAME

        future = (
            self.valid_client.call_async(req)
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()

        return (
            response is not None
            and bool(response.valid)
        )

    def ik(self, pose, seed):

        req = GetPositionIK.Request()

        req.ik_request.group_name = (
            GROUP_NAME
        )

        req.ik_request.ik_link_name = (
            TIP_LINK
        )

        req.ik_request.pose_stamped = pose

        req.ik_request.robot_state = (
            self.make_state(seed)
        )

        req.ik_request.avoid_collisions = True

        req.ik_request.timeout.sec = 1

        future = self.ik_client.call_async(
            req
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

    def add_obstacle(
        self,
        obstacle_id,
        x,
        y,
        z,
    ):

        obj = CollisionObject()

        obj.id = obstacle_id
        obj.header.frame_id = BASE_FRAME

        primitive = SolidPrimitive()

        primitive.type = SolidPrimitive.BOX

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

        obj.primitives = [primitive]
        obj.primitive_poses = [pose]

        obj.operation = CollisionObject.ADD

        scene = PlanningScene()

        scene.is_diff = True

        scene.world.collision_objects = [
            obj
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

    def remove_obstacle(
        self,
        obstacle_id,
    ):

        obj = CollisionObject()

        obj.id = obstacle_id
        obj.header.frame_id = BASE_FRAME

        obj.operation = CollisionObject.REMOVE

        scene = PlanningScene()

        scene.is_diff = True

        scene.world.collision_objects = [
            obj
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

        return future.result()

    def goal_constraints(
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

    def plan(
        self,
        start,
        goal,
    ):

        req = GetMotionPlan.Request()

        motion = req.motion_plan_request

        motion.group_name = GROUP_NAME

        motion.start_state = (
            self.make_state(start)
        )

        motion.goal_constraints = [
            self.goal_constraints(goal)
        ]

        motion.allowed_planning_time = 3.0

        motion.num_planning_attempts = 5

        motion.pipeline_id = "ompl"

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

        result = (
            response.motion_plan_response
        )

        if (
            result.error_code.val
            != MoveItErrorCodes.SUCCESS
        ):
            return None

        trajectory = (
            result.trajectory.joint_trajectory
        )

        if len(trajectory.points) < 2:
            return None

        return trajectory


def trajectory_length(
    trajectory,
):

    total = 0.0

    previous = None

    for point in trajectory.points:

        q = list(point.positions)

        if previous is not None:

            total += math.sqrt(
                sum(
                    (a - b) ** 2
                    for a, b in zip(
                        q,
                        previous,
                    )
                )
            )

        previous = q

    return total


def main():

    random.seed(42)

    rclpy.init()

    node = Generator()

    obstacle_id = "v4_obstacle"

    # 障害物なし状態で固定開始姿勢を確認
    node.remove_obstacle(obstacle_id)

    if not node.state_valid(
        START_JOINTS
    ):
        node.get_logger().error(
            "START_JOINTS is invalid."
        )

        sys.exit(1)

    start_pose = node.fk(
        START_JOINTS
    )

    if start_pose is None:
        node.get_logger().error(
            "Failed to compute start FK."
        )

        sys.exit(1)

    # 有効な固定目標を選択
    goal_joints = None
    goal_pose = None

    for candidate in GOAL_CANDIDATES:

        if not node.state_valid(
            candidate
        ):
            continue

        pose = node.fk(candidate)

        if pose is None:
            continue

        solution = node.ik(
            pose,
            candidate,
        )

        if solution is None:
            continue

        if not node.state_valid(
            solution
        ):
            continue

        goal_joints = solution
        goal_pose = pose

        break

    if goal_joints is None:

        node.get_logger().error(
            "No valid fixed goal found."
        )

        sys.exit(1)

    sp = start_pose.pose.position
    gp = goal_pose.pose.position

    midpoint_x = (
        sp.x + gp.x
    ) / 2.0

    midpoint_y = (
        sp.y + gp.y
    ) / 2.0

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

    output_path = (
        Path.home()
        / "so101_ws"
        / "obstacle_dataset_v4.csv"
    )

    rows = []

    episode_id = 0

    plan_failures = 0
    endpoint_collisions = 0

    for obstacle_class in [
        "left",
        "center",
        "right",
    ]:

        successful_for_class = 0

        attempts_for_class = 0

        while (
            successful_for_class
            < EPISODES_PER_CLASS
            and attempts_for_class < 100
        ):

            attempts_for_class += 1

            # 障害物のXは経路中央付近
            obs_x = (
                midpoint_x
                + random.uniform(
                    -0.015,
                    0.015,
                )
            )

            offset_low, offset_high = (
                OBSTACLE_Y_OFFSETS[
                    obstacle_class
                ]
            )

            obs_y = (
                midpoint_y
                + random.uniform(
                    offset_low,
                    offset_high,
                )
            )

            obs_z = (
                OBSTACLE_SIZE_Z / 2.0
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
                continue

            # 障害物が開始・目標そのものに
            # ぶつかっている条件は除外
            if (
                not node.state_valid(
                    START_JOINTS
                )
                or
                not node.state_valid(
                    goal_joints
                )
            ):
                endpoint_collisions += 1
                continue

            trajectory = node.plan(
                START_JOINTS,
                goal_joints,
            )

            if trajectory is None:
                plan_failures += 1
                continue

            episode_id += 1

            successful_for_class += 1

            length = trajectory_length(
                trajectory
            )

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
                continue

            for step, point in enumerate(
                trajectory.points
            ):

                t = (
                    float(
                        point.time_from_start.sec
                    )
                    +
                    float(
                        point.time_from_start.nanosec
                    )
                    * 1e-9
                )

                row = {
                    "episode_id":
                        episode_id,

                    "obstacle_class":
                        obstacle_class,

                    "step":
                        step,

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

                    "path_length":
                        length,

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
                f"Episode {episode_id}: "
                f"class={obstacle_class}, "
                f"points="
                f"{len(trajectory.points)}, "
                f"path_length="
                f"{length:.3f}, "
                f"obstacle=("
                f"{obs_x:.3f}, "
                f"{obs_y:.3f})"
            )

    node.remove_obstacle(
        obstacle_id
    )

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

        "path_length",

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
        "===== V4 SUMMARY ====="
    )

    node.get_logger().info(
        f"Episodes: {episode_id}"
    )

    node.get_logger().info(
        f"Trajectory rows: "
        f"{len(rows)}"
    )

    node.get_logger().info(
        f"Planning failures: "
        f"{plan_failures}"
    )

    node.get_logger().info(
        f"Endpoint collisions: "
        f"{endpoint_collisions}"
    )

    node.get_logger().info(
        f"Saved: {output_path}"
    )

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
