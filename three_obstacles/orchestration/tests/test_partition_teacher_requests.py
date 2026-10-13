#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "three_obstacles/orchestration/partition_teacher_requests.py"
spec = importlib.util.spec_from_file_location("partition_teacher_requests", SCRIPT_PATH)
assert spec and spec.loader
partitioner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = partitioner
spec.loader.exec_module(partitioner)


def record(index: int) -> dict[str, str]:
    return {"episode_id": f"scene_{index:05d}", "scene_hash": f"{index:016x}"}


class PartitionTeacherRequestsTest(unittest.TestCase):
    def test_partitions_are_balanced_and_complete(self) -> None:
        records = [record(index) for index in range(11)]
        shards = partitioner.partition(records, workers=3)
        self.assertEqual([len(shard) for shard in shards], [4, 4, 3])
        all_ids = [item["episode_id"] for shard in shards for item in shard]
        self.assertEqual(set(all_ids), {item["episode_id"] for item in records})

    def test_partition_is_deterministic(self) -> None:
        records = list(reversed([record(index) for index in range(8)]))
        first = partitioner.partition(records, workers=2)
        second = partitioner.partition(records, workers=2)
        self.assertEqual(first, second)

    def test_write_shards_creates_a_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "shards"
            manifest = partitioner.write_shards([[record(0)], [record(1)]], target, Path("input.jsonl"))
            self.assertEqual(manifest["total_scene_count"], 2)
            self.assertTrue((target / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
