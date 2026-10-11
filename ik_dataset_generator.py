#!/usr/bin/env python3

import csv
import random
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node

from moveit_msgs.msg import RobotState, MoveItErrorCodes
from moveit_msgs.srv import (
    GetPositionFK,
    GetPositionIK,
    GetStateValidity,
)
from sensor_msgs.msg import JointState


ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

TIP_LINK = "gripper_link"

# 最初は実機相当の限界より少し内側で使用
JOINT_LIMITS = {
    "shoulder_pan": (-1.2, 1.2),
    "shoulder_lift": (-1.0, 1.0),
    "elbow_flex": (-1.0, 1.0),
    "wrist_flex": (-0.9, 0.9),
    "wrist_roll": (-1.5, 1.5),
}

# 教師データとして使用する作業空間
# 最初は安全側に狭める
WORKSPACE = {
    "x_min": 0.06,
    "x_max": 0.36,
    "y_min": -0.28,
    "y_max": 0.28,
    "z_min": 0.05,
    "z_max": 0.43,
}


class IKDatasetGenerator(Node):

    def __init__(self):
        super().__init__("ik_dataset_generator")

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

        self.get_logger().info(
            "Waiting for /compute_fk ..."
        )
        if not self.fk_client.wait_for_service(
            timeout_sec=10.0
        ):
            raise RuntimeError(
                "/compute_fk service not available"
            )

        self.get_logger().info(
            "Waiting for /compute_ik ..."
        )
        if not self.ik_client.wait_for_service(
            timeout_sec=10.0
        ):
            raise RuntimeError(
                "/compute_ik service not available"
            )

        self.get_logger().info(
            "Waiting for /check_state_validity ..."
        )
        if not self.validity_client.wait_for_service(
            timeout_sec=10.0
        ):
            raise RuntimeError(
                "/check_state_validity service not available"
            )

        self.get_logger().info(
            "MoveIt FK / IK / StateValidity services are ready."
        )

    def make_robot_state(self, joint_values):
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

    def compute_fk(self, joint_values):
        req = GetPositionFK.Request()

        req.header.frame_id = "base_link"

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

        if len(response.pose_stamped) == 0:
            return None

        return response.pose_stamped[0]

    def pose_in_workspace(self, pose_stamped):
        p = pose_stamped.pose.position

        if not (
            WORKSPACE["x_min"]
            <= p.x
            <= WORKSPACE["x_max"]
        ):
            return False

        if not (
            WORKSPACE["y_min"]
            <= p.y
            <= WORKSPACE["y_max"]
        ):
            return False

        if not (
            WORKSPACE["z_min"]
            <= p.z
            <= WORKSPACE["z_max"]
        ):
            return False

        return True

    def compute_ik(
        self,
        target_pose,
        seed_joints,
    ):
        req = GetPositionIK.Request()

        req.ik_request.group_name = "arm"

        req.ik_request.ik_link_name = (
            TIP_LINK
        )

        req.ik_request.pose_stamped = (
            target_pose
        )

        # 第2版では衝突判定を有効化
        req.ik_request.avoid_collisions = True

        # FK生成時の姿勢をseedとして利用
        req.ik_request.robot_state = (
            self.make_robot_state(
                seed_joints
            )
        )

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

        req.group_name = "arm"

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

        return bool(response.valid)


def main():

    rclpy.init()

    node = IKDatasetGenerator()

    # 最初は100件で確認
    num_samples = 100

    # 無限ループ防止
    max_attempts = (
        num_samples * 50
    )

    output_path = (
        Path.home()
        / "so101_ws"
        / "ik_dataset_v2.csv"
    )

    rows = []

    success_count = 0
    attempt_count = 0

    fk_fail_count = 0
    workspace_reject_count = 0
    ik_fail_count = 0
    collision_reject_count = 0

    node.get_logger().info(
        f"Using tip link: {TIP_LINK}"
    )

    node.get_logger().info(
        f"Target samples: {num_samples}"
    )

    while (
        success_count < num_samples
        and attempt_count < max_attempts
    ):

        attempt_count += 1

        source_joints = (
            node.random_joint_configuration()
        )

        target_pose = (
            node.compute_fk(
                source_joints
            )
        )

        if target_pose is None:
            fk_fail_count += 1
            continue

        if not node.pose_in_workspace(
            target_pose
        ):
            workspace_reject_count += 1
            continue

        ik_solution = (
            node.compute_ik(
                target_pose,
                source_joints,
            )
        )

        if ik_solution is None:
            ik_fail_count += 1
            continue

        # IK結果をもう一度
        # StateValidityで確認
        if not node.state_is_valid(
            ik_solution
        ):
            collision_reject_count += 1
            continue

        pose = target_pose.pose

        row = {
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
                ik_solution[
                    "shoulder_pan"
                ],

            "shoulder_lift":
                ik_solution[
                    "shoulder_lift"
                ],

            "elbow_flex":
                ik_solution[
                    "elbow_flex"
                ],

            "wrist_flex":
                ik_solution[
                    "wrist_flex"
                ],

            "wrist_roll":
                ik_solution[
                    "wrist_roll"
                ],
        }

        rows.append(row)

        success_count += 1

        if (
            success_count == 1
            or success_count % 10 == 0
        ):
            node.get_logger().info(
                f"Success "
                f"{success_count}/"
                f"{num_samples} "
                f"(attempt "
                f"{attempt_count}) "
                f"x="
                f"{pose.position.x:.3f} "
                f"y="
                f"{pose.position.y:.3f} "
                f"z="
                f"{pose.position.z:.3f}"
            )

    fieldnames = [
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

        writer.writerows(
            rows
        )

    node.get_logger().info(
        "===== Dataset generation summary ====="
    )

    node.get_logger().info(
        f"Attempts: "
        f"{attempt_count}"
    )

    node.get_logger().info(
        f"Success: "
        f"{success_count}"
    )

    node.get_logger().info(
        f"FK failures: "
        f"{fk_fail_count}"
    )

    node.get_logger().info(
        f"Workspace rejected: "
        f"{workspace_reject_count}"
    )

    node.get_logger().info(
        f"IK failures: "
        f"{ik_fail_count}"
    )

    node.get_logger().info(
        f"Collision rejected: "
        f"{collision_reject_count}"
    )

    node.get_logger().info(
        f"Saved dataset: "
        f"{output_path}"
    )

    node.destroy_node()

    rclpy.shutdown()

    if success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
