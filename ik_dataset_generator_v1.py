#!/usr/bin/env python3

import csv
import random
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node

from moveit_msgs.msg import RobotState, MoveItErrorCodes
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from sensor_msgs.msg import JointState


ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# まずは安全側に少し狭めた範囲でランダム生成
JOINT_LIMITS = {
    "shoulder_pan": (-1.2, 1.2),
    "shoulder_lift": (-1.0, 1.0),
    "elbow_flex": (-1.0, 1.0),
    "wrist_flex": (-0.9, 0.9),
    "wrist_roll": (-1.5, 1.5),
}

TIP_LINK = "gripper_link"


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

        self.get_logger().info("Waiting for /compute_fk ...")
        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("/compute_fk service not available")

        self.get_logger().info("Waiting for /compute_ik ...")
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("/compute_ik service not available")

        self.get_logger().info("MoveIt IK/FK services are ready.")

    def make_robot_state(self, joint_values):
        state = RobotState()

        state.joint_state = JointState()
        state.joint_state.name = ARM_JOINTS.copy()
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
        req.fk_link_names = [TIP_LINK]
        req.robot_state = self.make_robot_state(joint_values)

        future = self.fk_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()

        if response is None:
            return None

        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None

        if len(response.pose_stamped) == 0:
            return None

        return response.pose_stamped[0]

    def compute_ik(self, target_pose, seed_joints):
        req = GetPositionIK.Request()

        req.ik_request.group_name = "arm"
        req.ik_request.ik_link_name = TIP_LINK
        req.ik_request.pose_stamped = target_pose

        # 最初の確認では衝突判定を無効化
        req.ik_request.avoid_collisions = False

        # FK生成に使った関節角度を初期値として与える
        req.ik_request.robot_state = self.make_robot_state(seed_joints)

        req.ik_request.timeout.sec = 1
        req.ik_request.timeout.nanosec = 0

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()

        if response is None:
            return None

        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None

        names = response.solution.joint_state.name
        positions = response.solution.joint_state.position

        result = dict(zip(names, positions))

        if not all(joint in result for joint in ARM_JOINTS):
            return None

        return {
            joint: float(result[joint])
            for joint in ARM_JOINTS
        }


def main():
    rclpy.init()

    node = IKDatasetGenerator()

    # まずは30件だけ作る
    num_samples = 30
    max_attempts = num_samples * 10

    output_path = (
        Path.home()
        / "so101_ws"
        / "ik_dataset.csv"
    )

    rows = []

    success_count = 0
    attempt_count = 0

    node.get_logger().info(
        f"Using IK/FK tip link: {TIP_LINK}"
    )

    while (
        success_count < num_samples
        and attempt_count < max_attempts
    ):
        attempt_count += 1

        source_joints = (
            node.random_joint_configuration()
        )

        target_pose = node.compute_fk(
            source_joints
        )

        if target_pose is None:
            node.get_logger().info(
                f"FK failed: attempt {attempt_count}"
            )
            continue

        ik_solution = node.compute_ik(
            target_pose,
            source_joints,
        )

        if ik_solution is None:
            node.get_logger().info(
                f"IK failed: attempt {attempt_count}"
            )
            continue

        pose = target_pose.pose

        row = {
            "target_x": pose.position.x,
            "target_y": pose.position.y,
            "target_z": pose.position.z,

            "target_qx": pose.orientation.x,
            "target_qy": pose.orientation.y,
            "target_qz": pose.orientation.z,
            "target_qw": pose.orientation.w,

            "shoulder_pan": ik_solution[
                "shoulder_pan"
            ],
            "shoulder_lift": ik_solution[
                "shoulder_lift"
            ],
            "elbow_flex": ik_solution[
                "elbow_flex"
            ],
            "wrist_flex": ik_solution[
                "wrist_flex"
            ],
            "wrist_roll": ik_solution[
                "wrist_roll"
            ],
        }

        rows.append(row)

        success_count += 1

        node.get_logger().info(
            f"Success {success_count}/{num_samples} "
            f"x={pose.position.x:.3f} "
            f"y={pose.position.y:.3f} "
            f"z={pose.position.z:.3f}"
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
        writer.writerows(rows)

    node.get_logger().info(
        f"Saved {len(rows)} samples "
        f"to {output_path}"
    )

    if len(rows) == 0:
        node.get_logger().error(
            "No valid IK samples were generated."
        )
        exit_code = 1
    else:
        exit_code = 0

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
