"""No-ROS tests for the strictly read-only state-validity probe."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from three_obstacles.teacher.state_validity_probe import (
    DEFAULT_JOINT_NAMES,
    StateValidityProbeError,
    build_probe_spec,
    portable_probe_summary,
    validity_report,
)


class StateValidityProbeTest(unittest.TestCase):
    def test_five_joint_read_only_contract(self) -> None:
        spec = build_probe_spec([0.1, -0.2, 0.3, -0.4, 0.5])
        summary = portable_probe_summary(spec)
        self.assertEqual(spec.joint_names, DEFAULT_JOINT_NAMES)
        self.assertEqual(summary["service"], "/check_state_validity")
        self.assertEqual(summary["group_name"], "arm")
        self.assertEqual(summary["scene_mutation"], "none")
        self.assertEqual(summary["attached_objects"], "none")
        self.assertEqual(summary["collision_matrix_changes"], "none")
        self.assertEqual(summary["robot_or_gazebo_motion"], "none")

    def test_invalid_count_and_duplicate_names_fail_before_ros(self) -> None:
        with self.assertRaises(StateValidityProbeError):
            build_probe_spec([0.0, 0.1, 0.2, 0.3])
        with self.assertRaises(StateValidityProbeError):
            build_probe_spec([0.0] * 5, ["a", "a", "c", "d", "e"])

    def test_response_reports_valid_error_code_and_contact_names_only(self) -> None:
        response = SimpleNamespace(
            valid=False,
            error_code=SimpleNamespace(val=-10),
            contacts=[
                SimpleNamespace(contact_body_1="moving_jaw_so101_v1_link", contact_body_2="shoulder_link"),
                SimpleNamespace(contact_body_1="", contact_body_2=""),
            ],
        )
        report = validity_report(response)
        self.assertFalse(report["valid"])
        self.assertEqual(report["moveit_error_code"], -10)
        self.assertEqual(report["contact_count"], 1)
        self.assertEqual(report["contacts"], [{"body_1": "moving_jaw_so101_v1_link", "body_2": "shoulder_link"}])


if __name__ == "__main__":
    unittest.main()
