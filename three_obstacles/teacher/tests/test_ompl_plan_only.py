"""No-ROS request and response tests for the strict plan-only client."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from three_obstacles.teacher.ompl_plan_only import (
    DEFAULT_JOINT_NAMES,
    PlanOnlyError,
    build_plan_spec,
    populate_motion_plan_request,
    portable_request_summary,
    trajectory_report,
)


class OmplPlanOnlyTest(unittest.TestCase):
    def test_explicit_five_joint_goal_builds_ompl_rrtconnect_request_contract(self) -> None:
        spec = build_plan_spec([0.1, -0.2, 0.3, -0.4, 0.5])
        summary = portable_request_summary(spec)
        self.assertEqual(spec.joint_names, DEFAULT_JOINT_NAMES)
        self.assertEqual(summary["service"], "/plan_kinematic_path")
        self.assertEqual(summary["group_name"], "arm")
        self.assertEqual(summary["pipeline_id"], "ompl")
        self.assertEqual(summary["planner_id"], "RRTConnectkConfigDefault")
        self.assertEqual(summary["goal_joint_positions_rad"]["wrist_roll"], 0.5)
        self.assertEqual(summary["attachments"], "none")
        self.assertEqual(summary["collision_matrix_changes"], "none")

    def test_wrong_goal_count_or_duplicate_joint_names_is_rejected_before_ros(self) -> None:
        with self.assertRaises(PlanOnlyError):
            build_plan_spec([0.0, 0.1, 0.2, 0.3])
        with self.assertRaises(PlanOnlyError):
            build_plan_spec([0.0] * 5, ["j1", "j1", "j3", "j4", "j5"])

    def test_explicit_start_is_written_to_joint_state_without_ros(self) -> None:
        spec = build_plan_spec(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            start_positions_rad=[-0.5, -0.4, -0.3, -0.2, -0.1],
        )

        class FakeJointState:
            def __init__(self) -> None:
                self.name = []
                self.position = []

        class FakeStartState:
            def __init__(self) -> None:
                self.is_diff = False
                self.joint_state = FakeJointState()

        class FakeMotionPlanRequest:
            def __init__(self) -> None:
                self.group_name = ""
                self.pipeline_id = ""
                self.planner_id = ""
                self.num_planning_attempts = 0
                self.allowed_planning_time = 0.0
                self.start_state = FakeStartState()
                self.goal_constraints = []

        class FakeRequest:
            def __init__(self) -> None:
                self.motion_plan_request = FakeMotionPlanRequest()

        class FakeConstraints:
            def __init__(self) -> None:
                self.joint_constraints = []

        class FakeJointConstraint:
            pass

        request = populate_motion_plan_request(spec, FakeRequest, FakeConstraints, FakeJointConstraint)
        start = request.motion_plan_request.start_state.joint_state
        self.assertTrue(request.motion_plan_request.start_state.is_diff)
        self.assertEqual(start.name, list(DEFAULT_JOINT_NAMES))
        self.assertEqual(start.position, [-0.5, -0.4, -0.3, -0.2, -0.1])
        self.assertEqual(request.motion_plan_request.goal_constraints[0].joint_constraints[0].position, 0.1)
        self.assertEqual(
            portable_request_summary(spec)["start_state_policy"],
            "explicit_arm_joint_positions_only_no_controller_command",
        )

    def test_report_contains_positions_but_not_velocity_or_effort(self) -> None:
        first = SimpleNamespace(positions=[0.0, 0.1], time_from_start=SimpleNamespace(sec=0, nanosec=0))
        second = SimpleNamespace(positions=[0.2, 0.3], time_from_start=SimpleNamespace(sec=1, nanosec=250_000_000))
        joint_trajectory = SimpleNamespace(joint_names=["j1", "j2"], points=[first, second])
        response = SimpleNamespace(
            motion_plan_response=SimpleNamespace(
                error_code=SimpleNamespace(val=1),
                planning_time=SimpleNamespace(sec=0, nanosec=125_000_000),
                trajectory=SimpleNamespace(joint_trajectory=joint_trajectory),
            )
        )
        report = trajectory_report(response)
        self.assertEqual(report["moveit_error_code"], 1)
        self.assertEqual(report["planning_time_s"], 0.125)
        self.assertEqual(report["trajectory"]["duration_s"], 1.25)
        self.assertEqual(report["trajectory"]["positions_rad"], [[0.0, 0.1], [0.2, 0.3]])
        self.assertNotIn("velocities", str(report))
        self.assertNotIn("effort", str(report))


if __name__ == "__main__":
    unittest.main()
