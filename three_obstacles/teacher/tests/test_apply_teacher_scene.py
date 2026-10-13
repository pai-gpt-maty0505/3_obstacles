"""Safety tests for the no-motion PlanningScene apply CLI helpers."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from three_obstacles.teacher.apply_teacher_scene import (
    TeacherSceneApplyError,
    prepare_scene,
    summary,
    validate_apply_document,
)
from three_obstacles.teacher.scene_builder import build_finalized_teacher_scene, load_mission_spec, load_teacher_requests


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "config/mission_spec.json"
REQUESTS_PATH = ROOT / "data/teacher_requests/smoke_20261013.jsonl"


class ApplyTeacherSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_mission_spec(SPEC_PATH)
        cls.request = load_teacher_requests(REQUESTS_PATH)[0]

    def test_prepare_builds_scene_through_injected_adapter_without_ros(self) -> None:
        calls: list[dict] = []
        marker = object()

        def fake_adapter(document: dict) -> object:
            calls.append(document)
            return marker

        document, planning_scene = prepare_scene(self.request, self.spec, adapter=fake_adapter)
        self.assertIs(planning_scene, marker)
        self.assertEqual(calls, [document])
        self.assertEqual(document["planning_scene"]["world"]["collision_objects"][0]["id"], "teacher/cylinder")

    def test_collision_matrix_mutation_is_rejected_before_ros_client_exists(self) -> None:
        document = build_finalized_teacher_scene(self.request, self.spec)
        document["planning_scene"]["allowed_collision_matrix"] = {"entry_names": []}
        with self.assertRaises(TeacherSceneApplyError):
            validate_apply_document(document)

    def test_attached_payload_representation_is_rejected_before_ros_client_exists(self) -> None:
        document = build_finalized_teacher_scene(self.request, self.spec)
        document["planning_scene"]["robot_state"]["attached_collision_objects"] = []
        with self.assertRaises(TeacherSceneApplyError):
            validate_apply_document(document)

    def test_dry_run_summary_reports_no_motion_or_attachment(self) -> None:
        document = build_finalized_teacher_scene(self.request, self.spec)
        report = summary(document, "dry_run_no_ros_service_call")
        self.assertEqual(report["robot_or_gazebo_motion"], "not_issued")
        self.assertEqual(report["attached_objects"], "not_used")
        self.assertEqual(report["collision_matrix_changes"], "not_used")
        self.assertEqual(report["route_gate_ids"][-1], "DISH_PLACE")


if __name__ == "__main__":
    unittest.main()
