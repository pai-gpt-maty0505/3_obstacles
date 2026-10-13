#!/usr/bin/env python3
"""Standard-library regression tests for the independent scene sampler."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / "three_obstacles/tools/generate_teacher_requests.py"
SPEC_PATH = REPO_ROOT / "three_obstacles/config/mission_spec.json"

spec = importlib.util.spec_from_file_location("teacher_generator", GENERATOR_PATH)
assert spec and spec.loader
generator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = generator
spec.loader.exec_module(generator)


class TeacherRequestGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mission_spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def test_same_seed_produces_same_scene(self) -> None:
        first = generator.generate_requests(self.mission_spec, count=8, root_seed=20261013)
        second = generator.generate_requests(self.mission_spec, count=8, root_seed=20261013)
        self.assertEqual(first, second)

    def test_request_has_required_ordered_gates_and_clearance(self) -> None:
        record = generator.generate_requests(self.mission_spec, count=1, root_seed=7)[0]
        self.assertEqual(
            [gate["id"] for gate in record["route_gates"]],
            ["PYRAMID_OUTER", "RECTANGLE_FRONT", "CYLINDER_OUTER", "DISH_PLACE"],
        )
        self.assertTrue(record["precheck"]["passed"])
        self.assertGreaterEqual(
            record["precheck"]["route_minimum_clearance_m"],
            record["precheck"]["required_static_clearance_m"],
        )

    def test_jsonl_round_trip(self) -> None:
        records = generator.generate_requests(self.mission_spec, count=12, root_seed=99)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "requests.jsonl"
            generator.write_jsonl(output, records)
            loaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records, loaded)
        self.assertEqual(len({record["scene_hash"] for record in records}), len(records))

    def test_split_counts_are_exact_and_scene_level(self) -> None:
        records = generator.generate_requests(self.mission_spec, count=80, root_seed=42)
        self.assertEqual(generator.split_counts(records), {"train": 60, "val": 10, "test": 10})
        self.assertTrue(all(record["split"] in {"train", "val", "test"} for record in records))


if __name__ == "__main__":
    unittest.main()
