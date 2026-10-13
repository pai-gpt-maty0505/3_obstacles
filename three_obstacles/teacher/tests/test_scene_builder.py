"""Standard-library tests for the offline teacher-scene preparation layer."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from three_obstacles.teacher.geometry import pose_from_yaw_pose, triangular_pyramid_mesh
from three_obstacles.teacher.scene_builder import (
    TeacherSceneError,
    build_finalized_teacher_scene,
    json_sha256,
    load_mission_spec,
    load_teacher_requests,
)
from three_obstacles.teacher.worker import RESULT_SCHEMA, process_shard


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "config/mission_spec.json"
REQUESTS_PATH = ROOT / "data/teacher_requests/smoke_20261013.jsonl"


class TeacherSceneBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_mission_spec(SPEC_PATH)
        cls.request = load_teacher_requests(REQUESTS_PATH)[0]

    def test_yaw_conversion_is_unit_quaternion(self) -> None:
        pose = pose_from_yaw_pose({"x": 1.0, "y": 2.0, "z": 3.0, "yaw_deg": 90.0})
        self.assertAlmostEqual(pose["orientation"]["z"], 2**-0.5)
        self.assertAlmostEqual(pose["orientation"]["w"], 2**-0.5)

    def test_pyramid_mesh_is_four_vertices_four_faces(self) -> None:
        mesh = triangular_pyramid_mesh(0.1, 0.12)
        self.assertEqual(len(mesh["vertices"]), 4)
        self.assertEqual(len(mesh["triangles"]), 4)
        self.assertEqual(mesh["vertices"][3]["z"], 0.06)

    def test_scene_has_all_five_collision_objects_and_ordered_gates(self) -> None:
        document = build_finalized_teacher_scene(self.request, self.spec)
        objects = document["planning_scene"]["world"]["collision_objects"]
        self.assertEqual([item["id"] for item in objects], [
            "teacher/cylinder", "teacher/rectangular_prism", "teacher/triangular_pyramid", "teacher/payload", "teacher/dish",
        ])
        self.assertEqual([gate["id"] for gate in document["route_plan"]["gates"]], [
            "PYRAMID_OUTER", "RECTANGLE_FRONT", "CYLINDER_OUTER", "DISH_PLACE",
        ])
        self.assertEqual(len(objects[2]["meshes"][0]["triangles"]), 4)
        self.assertEqual(len(objects[4]["primitives"]), 17)  # base + 16 rim segments
        digest_input = {key: value for key, value in document.items() if key != "document_sha256"}
        self.assertEqual(document["document_sha256"], json_sha256(digest_input))

    def test_tampered_scene_hash_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.request)
        tampered["scene_hash"] = "not-a-scene-hash"
        with self.assertRaises(TeacherSceneError):
            build_finalized_teacher_scene(tampered, self.spec)

    def test_worker_returns_one_preparation_result_per_record(self) -> None:
        result = process_shard(load_teacher_requests(REQUESTS_PATH)[:3], self.spec, "unit-000")
        self.assertEqual(len(result), 3)
        self.assertTrue(all(row["schema_version"] == RESULT_SCHEMA for row in result))
        self.assertTrue(all(row["status"] == "READY_FOR_OFFLINE_OMPL" for row in result))
        self.assertTrue(all(row["planning_duration_s"] is None for row in result))
        self.assertTrue(all(row["failure_reason"] is None for row in result))

    def test_worker_keeps_failure_as_a_result_row(self) -> None:
        broken = copy.deepcopy(self.request)
        broken["route_gates"] = []
        result = process_shard([broken], self.spec, "unit-broken")[0]
        self.assertEqual(result["status"], "REJECTED_SCENE_CONTRACT")
        self.assertIsNone(result["planning_duration_s"])
        self.assertIsInstance(result["failure_reason"], str)


if __name__ == "__main__":
    unittest.main()
