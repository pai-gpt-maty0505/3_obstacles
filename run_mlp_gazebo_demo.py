#!/usr/bin/env python3

import argparse
import csv
import math
import subprocess
import tempfile
import time

from collections import defaultdict
from pathlib import Path

import rclpy

from rclpy.action import ActionClient
from rclpy.node import Node

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)

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
# 基本設定
# ============================================================

BASE_DIR = Path.home() / "so101_ws"

PREDICTIONS_FILE = (
    BASE_DIR
    / "mlp_test_predictions.csv"
)

DEMO_CANDIDATES_FILE = (
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


# 教師データ生成時と同じ開始姿勢
START_JOINTS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
}


GROUP_NAME = "arm"
BASE_FRAME = "base_link"

ARM_ACTION = (
    "/arm_controller/"
    "follow_joint_trajectory"
)

GAZEBO_WORLD = "default"

GAZEBO_OBSTACLE_NAME = (
    "mlp_demo_obstacle"
)

MOVEIT_OBSTACLE_ID = (
    "mlp_demo_obstacle"
)


# ============================================================
# 障害物サイズ
# V4.2と同一
# ============================================================

OBSTACLE_SIZE_X = 0.055
OBSTACLE_SIZE_Y = 0.055
OBSTACLE_SIZE_Z = 0.22


# ============================================================
# 安全性チェック
# ============================================================

INTERPOLATION_SUBSTEPS = 5


# ============================================================
# 動作時間
#
# 1エピソード
# reset 4秒 + AI軌道30秒
# → 約34秒を目標
# ============================================================

DEFAULT_RESET_SECONDS = 4.0
DEFAULT_TRAJECTORY_SECONDS = 30.0

PAUSE_BETWEEN_EPISODES = 1.0


# ============================================================
# fallback用デモ候補
# ============================================================

FALLBACK_EPISODES = {
    "left": 4,
    "center": 173,
    "right": 203,
}


# ============================================================
# Duration生成
# ============================================================

def make_duration(seconds):

    whole = int(seconds)

    nanoseconds = int(
        round(
            (
                seconds
                - whole
            )
            * 1e9
        )
    )

    if nanoseconds >= 1_000_000_000:

        whole += 1
        nanoseconds -= 1_000_000_000

    return Duration(
        sec=whole,
        nanosec=nanoseconds,
    )


# ============================================================
# 予測CSV読み込み
# ============================================================

def load_predictions(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Prediction file not found: "
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
            key=lambda row: int(
                row["step"]
            )
        )

    return episodes


# ============================================================
# デモ候補読み込み
# ============================================================

def load_demo_candidates(
    path,
):

    result = {}

    if path.exists():

        with path.open(
            "r",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                obstacle_class = (
                    row[
                        "obstacle_class"
                    ]
                )

                result[
                    obstacle_class
                ] = int(
                    row[
                        "episode_id"
                    ]
                )

    # 足りない場合は既知の安全候補を使用
    for class_name, episode_id in (
        FALLBACK_EPISODES.items()
    ):

        if class_name not in result:

            result[
                class_name
            ] = episode_id

    return result


# ============================================================
# 予測関節角度取得
# ============================================================

def predicted_joints(row):

    return {
        joint: float(
            row[
                f"pred_{joint}"
            ]
        )
        for joint in ARM_JOINTS
    }


# ============================================================
# 2姿勢間の補間
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
# Gazebo障害物SDF作成
# ============================================================

def create_obstacle_sdf():

    text = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{GAZEBO_OBSTACLE_NAME}">
    <static>true</static>

    <link name="link">

      <collision name="collision">
        <geometry>
          <box>
            <size>
              {OBSTACLE_SIZE_X}
              {OBSTACLE_SIZE_Y}
              {OBSTACLE_SIZE_Z}
            </size>
          </box>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <box>
            <size>
              {OBSTACLE_SIZE_X}
              {OBSTACLE_SIZE_Y}
              {OBSTACLE_SIZE_Z}
            </size>
          </box>
        </geometry>

        <material>
          <ambient>
            0.8 0.1 0.1 1
          </ambient>

          <diffuse>
            0.8 0.1 0.1 1
          </diffuse>
        </material>
      </visual>

    </link>
  </model>
</sdf>
"""

    path = Path(
        tempfile.gettempdir()
    ) / "so101_mlp_demo_obstacle.sdf"

    path.write_text(
        text
    )

    return path


# ============================================================
# Gazebo障害物削除
# ============================================================

def remove_gazebo_obstacle():

    command = [
        "gz",
        "service",

        "-s",
        f"/world/"
        f"{GAZEBO_WORLD}/remove",

        "--reqtype",
        "gz.msgs.Entity",

        "--reptype",
        "gz.msgs.Boolean",

        "--timeout",
        "3000",

        "--req",
        (
            f'name: "'
            f'{GAZEBO_OBSTACLE_NAME}'
            f'", type: MODEL'
        ),
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


# ============================================================
# Gazebo障害物生成
# ============================================================

def spawn_gazebo_obstacle(
    x,
    y,
    z,
):

    remove_gazebo_obstacle()

    sdf_path = (
        create_obstacle_sdf()
    )

    command = [
        "ros2",
        "run",
        "ros_gz_sim",
        "create",

        "--world",
        GAZEBO_WORLD,

        "--file",
        str(
            sdf_path
        ),

        "--name",
        GAZEBO_OBSTACLE_NAME,

        "-x",
        str(x),

        "-y",
        str(y),

        "-z",
        str(z),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:

        print(
            "Gazebo obstacle "
            "creation failed."
        )

        print(
            result.stdout
        )

        print(
            result.stderr
        )

        return False

    time.sleep(
        0.5
    )

    return True


# ============================================================
# ROS Node
# ============================================================

class MLPDemoNode(Node):

    def __init__(self):

        super().__init__(
            "run_mlp_gazebo_demo"
        )

        self.scene_client = (
            self.create_client(
                ApplyPlanningScene,
                "/apply_planning_scene",
            )
        )

        self.validity_client = (
            self.create_client(
                GetStateValidity,
                "/check_state_validity",
            )
        )

        self.arm_action = ActionClient(
            self,
            FollowJointTrajectory,
            ARM_ACTION,
        )

        self.wait_for_interfaces()

    # --------------------------------------------------------
    # interface待ち
    # --------------------------------------------------------

    def wait_for_interfaces(self):

        self.get_logger().info(
            "Waiting for "
            "/apply_planning_scene ..."
        )

        if not (
            self.scene_client
            .wait_for_service(
                timeout_sec=10.0
            )
        ):

            raise RuntimeError(
                "/apply_planning_scene "
                "not available"
            )

        self.get_logger().info(
            "Waiting for "
            "/check_state_validity ..."
        )

        if not (
            self.validity_client
            .wait_for_service(
                timeout_sec=10.0
            )
        ):

            raise RuntimeError(
                "/check_state_validity "
                "not available"
            )

        self.get_logger().info(
            "Waiting for arm "
            "trajectory action ..."
        )

        if not (
            self.arm_action
            .wait_for_server(
                timeout_sec=10.0
            )
        ):

            raise RuntimeError(
                f"{ARM_ACTION} "
                "not available"
            )

        self.get_logger().info(
            "All demo interfaces "
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

        state.joint_state = (
            JointState()
        )

        state.joint_state.name = (
            ARM_JOINTS.copy()
        )

        state.joint_state.position = [
            float(
                joints[joint]
            )
            for joint in ARM_JOINTS
        ]

        return state

    # --------------------------------------------------------
    # state validity
    # --------------------------------------------------------

    def state_is_valid(
        self,
        joints,
    ):

        request = (
            GetStateValidity.Request()
        )

        request.robot_state = (
            self.make_robot_state(
                joints
            )
        )

        request.group_name = (
            GROUP_NAME
        )

        future = (
            self.validity_client
            .call_async(
                request
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
    # MoveIt障害物追加
    # --------------------------------------------------------

    def add_moveit_obstacle(
        self,
        x,
        y,
        z,
    ):

        collision_object = (
            CollisionObject()
        )

        collision_object.id = (
            MOVEIT_OBSTACLE_ID
        )

        collision_object.header.frame_id = (
            BASE_FRAME
        )

        primitive = (
            SolidPrimitive()
        )

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

        request = (
            ApplyPlanningScene.Request()
        )

        request.scene = scene

        future = (
            self.scene_client
            .call_async(
                request
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
    # MoveIt障害物削除
    # --------------------------------------------------------

    def remove_moveit_obstacle(
        self,
    ):

        obj = CollisionObject()

        obj.id = (
            MOVEIT_OBSTACLE_ID
        )

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

        request = (
            ApplyPlanningScene.Request()
        )

        request.scene = scene

        future = (
            self.scene_client
            .call_async(
                request
            )
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        return future.result()

    # --------------------------------------------------------
    # 予測軌道の安全性確認
    #
    # START→pred0 も確認
    # pred0→pred1...も確認
    # --------------------------------------------------------

    def validate_prediction(
        self,
        rows,
    ):

        predicted = [
            predicted_joints(row)
            for row in rows
        ]

        all_points = [
            START_JOINTS
        ] + predicted

        total_checks = 0

        invalid_checks = 0

        for i in range(
            len(all_points) - 1
        ):

            q0 = all_points[i]
            q1 = all_points[i + 1]

            for substep in range(
                INTERPOLATION_SUBSTEPS
            ):

                alpha = (
                    substep
                    / float(
                        INTERPOLATION_SUBSTEPS
                    )
                )

                q = interpolate_joints(
                    q0,
                    q1,
                    alpha,
                )

                total_checks += 1

                if not self.state_is_valid(
                    q
                ):

                    invalid_checks += 1

                    return (
                        False,
                        total_checks,
                        invalid_checks,
                    )

        # 最終点
        total_checks += 1

        if not self.state_is_valid(
            all_points[-1]
        ):

            invalid_checks += 1

        safe = (
            invalid_checks == 0
        )

        return (
            safe,
            total_checks,
            invalid_checks,
        )

    # --------------------------------------------------------
    # FollowJointTrajectory実行
    # --------------------------------------------------------

    def send_trajectory(
        self,
        joint_points,
        total_seconds,
        label,
    ):

        goal = (
            FollowJointTrajectory.Goal()
        )

        trajectory = (
            JointTrajectory()
        )

        trajectory.joint_names = (
            ARM_JOINTS.copy()
        )

        count = len(
            joint_points
        )

        for i, joints in enumerate(
            joint_points
        ):

            point = (
                JointTrajectoryPoint()
            )

            point.positions = [
                float(
                    joints[joint]
                )
                for joint in ARM_JOINTS
            ]

            # 最初の点も0秒にはしない
            t = (
                total_seconds
                * (
                    i + 1
                )
                / count
            )

            point.time_from_start = (
                make_duration(t)
            )

            trajectory.points.append(
                point
            )

        goal.trajectory = trajectory

        self.get_logger().info(
            f"Sending {label}: "
            f"{count} points, "
            f"{total_seconds:.1f} s"
        )

        send_future = (
            self.arm_action
            .send_goal_async(
                goal
            )
        )

        rclpy.spin_until_future_complete(
            self,
            send_future,
        )

        goal_handle = (
            send_future.result()
        )

        if (
            goal_handle is None
            or not goal_handle.accepted
        ):

            self.get_logger().error(
                f"{label} goal rejected."
            )

            return False

        result_future = (
            goal_handle
            .get_result_async()
        )

        rclpy.spin_until_future_complete(
            self,
            result_future,
        )

        wrapped_result = (
            result_future.result()
        )

        if wrapped_result is None:

            return False

        status = (
            wrapped_result.status
        )

        result = (
            wrapped_result.result
        )

        success = (
            status
            == GoalStatus.STATUS_SUCCEEDED
            and result.error_code == 0
        )

        if success:

            self.get_logger().info(
                f"{label}: SUCCEEDED"
            )

        else:

            self.get_logger().error(
                f"{label}: FAILED "
                f"status={status}, "
                f"error_code="
                f"{result.error_code}, "
                f"error_string="
                f"{result.error_string}"
            )

        return success

    # --------------------------------------------------------
    # 開始姿勢へ戻す
    # --------------------------------------------------------

    def move_to_start(
        self,
        seconds,
    ):

        return self.send_trajectory(
            [
                START_JOINTS
            ],
            seconds,
            "reset-to-start",
        )

    # --------------------------------------------------------
    # AI軌道実行
    # --------------------------------------------------------

    def execute_prediction(
        self,
        rows,
        seconds,
    ):

        points = [
            predicted_joints(row)
            for row in rows
        ]

        return self.send_trajectory(
            points,
            seconds,
            "MLP-predicted-trajectory",
        )


# ============================================================
# main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually move the Gazebo "
            "robot. Without this flag, "
            "only preflight checks run."
        ),
    )

    parser.add_argument(
        "--reset-seconds",
        type=float,
        default=DEFAULT_RESET_SECONDS,
    )

    parser.add_argument(
        "--trajectory-seconds",
        type=float,
        default=DEFAULT_TRAJECTORY_SECONDS,
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help=(
            "Run only one episode ID."
        ),
    )

    args = parser.parse_args()

    print(
        "================================"
    )

    print(
        "SO-101 MLP Gazebo Demo"
    )

    print(
        "================================"
    )

    print(
        "ONLINE OMPL REPLANNING: OFF"
    )

    print(
        "Trajectory source: "
        "MLP predictions"
    )

    print()

    episodes = load_predictions(
        PREDICTIONS_FILE
    )

    candidates = load_demo_candidates(
        DEMO_CANDIDATES_FILE
    )

    ordered_candidates = [
        (
            "left",
            candidates["left"],
        ),
        (
            "center",
            candidates["center"],
        ),
        (
            "right",
            candidates["right"],
        ),
    ]

    if args.episode is not None:

        filtered = []

        for class_name, episode_id in (
            ordered_candidates
        ):

            if episode_id == (
                args.episode
            ):

                filtered.append(
                    (
                        class_name,
                        episode_id,
                    )
                )

        # 候補にないIDも指定可能
        if not filtered:

            if args.episode not in episodes:

                raise RuntimeError(
                    f"Episode "
                    f"{args.episode} "
                    f"not found."
                )

            class_name = (
                episodes[
                    args.episode
                ][0][
                    "obstacle_class"
                ]
            )

            filtered = [
                (
                    class_name,
                    args.episode,
                )
            ]

        ordered_candidates = (
            filtered
        )

    print(
        "Demo episodes:"
    )

    for class_name, episode_id in (
        ordered_candidates
    ):

        print(
            f"  {class_name:6s}: "
            f"Episode "
            f"{episode_id}"
        )

    print()

    if not args.execute:

        print(
            "PRE-FLIGHT MODE"
        )

        print(
            "Robot will NOT move."
        )

    else:

        print(
            "EXECUTION MODE"
        )

        print(
            "Gazebo robot WILL move."
        )

    print()

    rclpy.init()

    node = MLPDemoNode()

    results = []

    total_session_start = None
    total_session_end = None

    try:

        for index, (
            expected_class,
            episode_id,
        ) in enumerate(
            ordered_candidates,
            start=1,
        ):

            if episode_id not in episodes:

                print(
                    f"Episode "
                    f"{episode_id} "
                    f"not found. SKIP."
                )

                continue

            rows = episodes[
                episode_id
            ]

            first = rows[0]

            actual_class = (
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

            print()
            print(
                "--------------------------------"
            )

            print(
                f"Demo {index}/"
                f"{len(ordered_candidates)}"
            )

            print(
                f"Episode: {episode_id}"
            )

            print(
                f"Class: {actual_class}"
            )

            print(
                f"Obstacle: "
                f"x={obs_x:.3f}, "
                f"y={obs_y:.3f}, "
                f"z={obs_z:.3f}"
            )

            # -----------------------------------------------
            # 前エピソード障害物削除
            # -----------------------------------------------

            node.remove_moveit_obstacle()

            remove_gazebo_obstacle()

            # -----------------------------------------------
            # 実行時はまず障害物なしで開始姿勢へ
            # -----------------------------------------------

            episode_wall_start = None

            if args.execute:

                if (
                    total_session_start
                    is None
                ):

                    total_session_start = (
                        time.monotonic()
                    )

                episode_wall_start = (
                    time.monotonic()
                )

                print(
                    "Resetting robot "
                    "to START..."
                )

                reset_ok = (
                    node.move_to_start(
                        args.reset_seconds
                    )
                )

                if not reset_ok:

                    print(
                        "Reset failed. "
                        "SKIP."
                    )

                    continue

            # -----------------------------------------------
            # MoveIt障害物
            # -----------------------------------------------

            if not (
                node.add_moveit_obstacle(
                    obs_x,
                    obs_y,
                    obs_z,
                )
            ):

                print(
                    "MoveIt obstacle "
                    "creation failed."
                )

                continue

            # -----------------------------------------------
            # AI推論軌道安全性再確認
            # -----------------------------------------------

            (
                safe,
                checks,
                invalid,
            ) = node.validate_prediction(
                rows
            )

            print(
                f"Safety validation: "
                f"{'SAFE' if safe else 'UNSAFE'} "
                f"("
                f"{invalid}/"
                f"{checks} invalid"
                f")"
            )

            if not safe:

                print(
                    "Unsafe trajectory. "
                    "Execution blocked."
                )

                node.remove_moveit_obstacle()

                continue

            # -----------------------------------------------
            # preflightだけならここで終了
            # -----------------------------------------------

            if not args.execute:

                results.append(
                    {
                        "episode":
                            episode_id,

                        "class":
                            actual_class,

                        "safe":
                            True,

                        "executed":
                            False,

                        "trajectory_seconds":
                            0.0,

                        "episode_wall":
                            0.0,
                    }
                )

                node.remove_moveit_obstacle()

                continue

            # -----------------------------------------------
            # Gazebo障害物表示
            # -----------------------------------------------

            print(
                "Spawning Gazebo "
                "obstacle..."
            )

            if not (
                spawn_gazebo_obstacle(
                    obs_x,
                    obs_y,
                    obs_z,
                )
            ):

                print(
                    "Gazebo obstacle "
                    "spawn failed."
                )

                node.remove_moveit_obstacle()

                continue

            # -----------------------------------------------
            # AI軌道実行
            # -----------------------------------------------

            print(
                "Executing MLP "
                "trajectory..."
            )

            trajectory_start = (
                time.monotonic()
            )

            execute_ok = (
                node.execute_prediction(
                    rows,
                    args.trajectory_seconds,
                )
            )

            trajectory_end = (
                time.monotonic()
            )

            episode_wall_end = (
                time.monotonic()
            )

            trajectory_wall = (
                trajectory_end
                - trajectory_start
            )

            episode_wall = (
                episode_wall_end
                - episode_wall_start
            )

            print(
                f"Trajectory wall time: "
                f"{trajectory_wall:.2f} s"
            )

            print(
                f"Episode total time: "
                f"{episode_wall:.2f} s"
            )

            print(
                f"60-second requirement: "
                f"{'PASS' if episode_wall < 60.0 else 'FAIL'}"
            )

            results.append(
                {
                    "episode":
                        episode_id,

                    "class":
                        actual_class,

                    "safe":
                        safe,

                    "executed":
                        execute_ok,

                    "trajectory_seconds":
                        trajectory_wall,

                    "episode_wall":
                        episode_wall,
                }
            )

            # -----------------------------------------------
            # 障害物削除
            # -----------------------------------------------

            node.remove_moveit_obstacle()

            remove_gazebo_obstacle()

            if index < len(
                ordered_candidates
            ):

                time.sleep(
                    PAUSE_BETWEEN_EPISODES
                )

        if args.execute:

            total_session_end = (
                time.monotonic()
            )

    finally:

        node.remove_moveit_obstacle()

        remove_gazebo_obstacle()

        node.destroy_node()

        rclpy.shutdown()

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "================================"
    )

    print(
        "===== DEMO SUMMARY ====="
    )

    print(
        "================================"
    )

    for result in results:

        print(
            f"{result['class']:6s} "
            f"Episode "
            f"{result['episode']:3d}: "
            f"safe="
            f"{result['safe']} "
            f"executed="
            f"{result['executed']} "
            f"episode_time="
            f"{result['episode_wall']:.2f}s"
        )

    if (
        args.execute
        and total_session_start
        is not None
        and total_session_end
        is not None
    ):

        total_wall = (
            total_session_end
            - total_session_start
        )

        print()

        print(
            f"TOTAL demo time: "
            f"{total_wall:.2f} s"
        )

        print(
            f"3-minute requirement: "
            f"{'PASS' if total_wall < 180.0 else 'FAIL'}"
        )


if __name__ == "__main__":

    main()
