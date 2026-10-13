"""No-ROS tests for the strictly query-only GetPositionIK probe."""

from __future__ import annotations

import unittest
from array import array
from types import SimpleNamespace

from three_obstacles.teacher.ik_pose_probe import (
    IKPoseProbeError,
    build_ros_request,
    call_service_once,
    dry_run_summary,
    make_spec,
)


class _FakeTimeout:
    def __init__(self) -> None:
        self.sec = 0
        self.nanosec = 0


class _FakeIKRequest:
    def __init__(self) -> None:
        self.group_name = ""
        self.ik_link_name = ""
        self.pose_stamped = None
        self.avoid_collisions = False
        self.timeout = _FakeTimeout()


class _FakeGetPositionIKRequest:
    def __init__(self) -> None:
        self.ik_request = _FakeIKRequest()


class _FakeGetPositionIK:
    Request = _FakeGetPositionIKRequest


class _FakePoseStamped:
    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="")
        self.pose = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        )


class IKPoseProbeTest(unittest.TestCase):
    def _spec(self, **overrides: object):
        arguments: dict[str, object] = {
            "group_name": "arm",
            "link_name": "gripper_frame_link",
            "frame_id": "base_link",
            "x": 0.24,
            "y": -0.12,
            "z": 0.18,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "timeout_s": 3.0,
        }
        arguments.update(overrides)
        return make_spec(**arguments)  # type: ignore[arg-type]

    def test_builds_one_explicit_collision_avoiding_ik_request(self) -> None:
        request = build_ros_request(self._spec(), _FakeGetPositionIK, _FakePoseStamped)
        ik_request = request.ik_request
        self.assertEqual(ik_request.group_name, "arm")
        self.assertEqual(ik_request.ik_link_name, "gripper_frame_link")
        self.assertEqual(ik_request.pose_stamped.header.frame_id, "base_link")
        self.assertEqual(
            (
                ik_request.pose_stamped.pose.position.x,
                ik_request.pose_stamped.pose.position.y,
                ik_request.pose_stamped.pose.position.z,
            ),
            (0.24, -0.12, 0.18),
        )
        self.assertEqual(ik_request.pose_stamped.pose.orientation.w, 1.0)
        self.assertTrue(ik_request.avoid_collisions)
        self.assertEqual((ik_request.timeout.sec, ik_request.timeout.nanosec), (3, 0))

    def test_rejects_nonunit_quaternion_before_ros(self) -> None:
        with self.assertRaises(IKPoseProbeError):
            self._spec(qw=0.9)
        with self.assertRaises(IKPoseProbeError):
            self._spec(timeout_s=0.0)
        with self.assertRaises(IKPoseProbeError):
            self._spec(group_name=" ")

    def test_duration_rounding_remains_a_valid_duration(self) -> None:
        request = build_ros_request(
            self._spec(timeout_s=1.9999999999),
            _FakeGetPositionIK,
            _FakePoseStamped,
        )
        self.assertEqual((request.ik_request.timeout.sec, request.ik_request.timeout.nanosec), (2, 0))

    def test_dry_run_record_proves_no_service_or_robot_operation(self) -> None:
        summary = dry_run_summary(self._spec())
        self.assertEqual(summary["service_calls"], 0)
        self.assertEqual(summary["robot_or_gazebo_motion"], "not_issued")
        self.assertEqual(summary["trajectory_execution"], "not_issued")
        self.assertEqual(summary["controller_or_gripper_command"], "not_issued")
        self.assertEqual(summary["planning_scene_mutation"], "not_issued")
        self.assertEqual(summary["attached_objects"], "not_used")
        self.assertEqual(summary["collision_matrix_changes"], "not_used")

    def test_check_path_performs_exactly_one_bounded_ik_service_call(self) -> None:
        class FakeFuture:
            def done(self) -> bool:
                return True

            def result(self):
                return SimpleNamespace(
                    error_code=SimpleNamespace(val=1),
                    solution=SimpleNamespace(
                        joint_state=SimpleNamespace(
                            name=["shoulder_pan", "shoulder_lift"],
                            position=[0.1, -0.2],
                        )
                    ),
                )

        class FakeClient:
            def __init__(self) -> None:
                self.waits: list[float] = []
                self.requests: list[object] = []

            def wait_for_service(self, timeout_sec: float) -> bool:
                self.waits.append(timeout_sec)
                return True

            def call_async(self, request: object) -> FakeFuture:
                self.requests.append(request)
                return FakeFuture()

        class FakeNode:
            def __init__(self) -> None:
                self.client = FakeClient()
                self.destroyed = False

            def create_client(self, service_type: object, service_name: str) -> FakeClient:
                self.service_type = service_type
                self.service_name = service_name
                return self.client

            def destroy_node(self) -> None:
                self.destroyed = True

        class FakeRclpy:
            def __init__(self) -> None:
                self.initialized = False
                self.node = FakeNode()
                self.spins: list[tuple[FakeNode, FakeFuture, float]] = []
                self.shutdown_called = False

            def ok(self) -> bool:
                return self.initialized

            def init(self, args: object = None) -> None:
                self.initialized = True

            def create_node(self, name: str) -> FakeNode:
                self.node_name = name
                return self.node

            def spin_until_future_complete(self, node: FakeNode, future: FakeFuture, timeout_sec: float) -> None:
                self.spins.append((node, future, timeout_sec))

            def shutdown(self) -> None:
                self.shutdown_called = True
                self.initialized = False

        fake_rclpy = FakeRclpy()
        result = call_service_once(
            self._spec(timeout_s=2.5),
            fake_rclpy,
            _FakeGetPositionIK,
            _FakePoseStamped,
        )
        self.assertEqual(result, {"joint_names": ["shoulder_pan", "shoulder_lift"], "joint_positions_rad": [0.1, -0.2]})
        self.assertEqual(fake_rclpy.node.service_name, "/compute_ik")
        self.assertEqual(fake_rclpy.node.client.waits, [2.5])
        self.assertEqual(len(fake_rclpy.node.client.requests), 1)
        self.assertEqual(len(fake_rclpy.spins), 1)
        self.assertTrue(fake_rclpy.node.destroyed)
        self.assertTrue(fake_rclpy.shutdown_called)

    def test_unsuccessful_ik_response_is_rejected(self) -> None:
        class FailingFuture:
            def done(self) -> bool:
                return True

            def result(self):
                return SimpleNamespace(error_code=SimpleNamespace(val=-31), solution=SimpleNamespace(joint_state=None))

        class Client:
            def wait_for_service(self, timeout_sec: float) -> bool:
                return True

            def call_async(self, request: object) -> FailingFuture:
                return FailingFuture()

        class Node:
            def create_client(self, service_type: object, service_name: str) -> Client:
                return Client()

            def destroy_node(self) -> None:
                pass

        class Rclpy:
            def ok(self) -> bool:
                return True

            def create_node(self, name: str) -> Node:
                return Node()

            def spin_until_future_complete(self, node: Node, future: FailingFuture, timeout_sec: float) -> None:
                pass

        with self.assertRaisesRegex(IKPoseProbeError, "MoveIt error code -31"):
            call_service_once(self._spec(), Rclpy(), _FakeGetPositionIK, _FakePoseStamped)

    def test_standard_ros_sequence_style_joint_state_is_accepted(self) -> None:
        """Regression: Jazzy may use array-like containers, not only lists."""

        class Future:
            def done(self) -> bool:
                return True

            def result(self):
                return SimpleNamespace(
                    error_code=SimpleNamespace(val=1),
                    solution=SimpleNamespace(
                        joint_state=SimpleNamespace(
                            name=(name for name in ("shoulder_pan", "shoulder_lift")),
                            position=array("d", [0.125, -0.25]),
                        )
                    ),
                )

        class Client:
            def wait_for_service(self, timeout_sec: float) -> bool:
                return True

            def call_async(self, request: object) -> Future:
                return Future()

        class Node:
            def create_client(self, service_type: object, service_name: str) -> Client:
                return Client()

            def destroy_node(self) -> None:
                pass

        class Rclpy:
            def ok(self) -> bool:
                return True

            def create_node(self, name: str) -> Node:
                return Node()

            def spin_until_future_complete(self, node: Node, future: Future, timeout_sec: float) -> None:
                pass

        result = call_service_once(self._spec(), Rclpy(), _FakeGetPositionIK, _FakePoseStamped)
        self.assertEqual(
            result,
            {"joint_names": ["shoulder_pan", "shoulder_lift"], "joint_positions_rad": [0.125, -0.25]},
        )


if __name__ == "__main__":
    unittest.main()
