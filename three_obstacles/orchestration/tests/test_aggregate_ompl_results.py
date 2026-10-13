#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


partitioner = load_module("partition_for_aggregate_tests", "three_obstacles/orchestration/partition_teacher_requests.py")
aggregator = load_module("aggregate_for_tests", "three_obstacles/orchestration/aggregate_ompl_results.py")


def request(index: int) -> dict[str, str]:
    return {"episode_id": f"scene_{index:05d}", "scene_hash": f"hash-{index:04d}"}


def terminal_result(source: dict[str, str], shard_id: str, status: str) -> dict:
    result = {
        "schema_version": "three_obstacles_teacher_worker_result/v1",
        "shard_id": shard_id,
        "episode_id": source["episode_id"],
        "scene_hash": source["scene_hash"],
        "status": status,
        "planning_duration_s": 0.25,
        "failure_reason": None,
    }
    if status == "SUCCEEDED_OFFLINE_OMPL":
        result["trajectory_sha256"] = "a" * 64
    elif status == "FAILED_OFFLINE_OMPL":
        result["failure_reason"] = "planner exhausted its time budget"
    else:
        result["planning_duration_s"] = None
        result["failure_reason"] = "scene contract was invalid"
    return result


class AggregateOmplResultsTests(unittest.TestCase):
    def make_partition(self, root: Path) -> tuple[Path, dict[str, list[dict[str, str]]]]:
        shard_dir = root / "shards"
        partitioner.write_shards([[request(0), request(1)], [request(2)]], shard_dir, Path("requests.jsonl"))
        manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
        records_by_shard = {
            Path(worker["input"]).stem: aggregator.load_jsonl(shard_dir / worker["input"])
            for worker in manifest["workers"]
        }
        return shard_dir / "manifest.json", records_by_shard

    @staticmethod
    def write_rows(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_complete_terminal_results_are_accepted_and_timed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, records_by_shard = self.make_partition(root)
            result_paths: dict[str, Path] = {}
            statuses = ["SUCCEEDED_OFFLINE_OMPL", "FAILED_OFFLINE_OMPL", "REJECTED_SCENE_CONTRACT"]
            position = 0
            for shard_id, records in records_by_shard.items():
                target = root / f"{shard_id}.results.jsonl"
                self.write_rows(target, [terminal_result(record, shard_id, statuses[position + index]) for index, record in enumerate(records)])
                position += len(records)
                result_paths[shard_id] = target
            report = aggregator.aggregate_results(manifest, result_paths)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["results"]["status_counts"], {"FAILED_OFFLINE_OMPL": 1, "REJECTED_SCENE_CONTRACT": 1, "SUCCEEDED_OFFLINE_OMPL": 1})
            self.assertEqual(report["results"]["timing_s"]["count"], 2)

    def test_missing_and_duplicate_results_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, records_by_shard = self.make_partition(root)
            result_paths: dict[str, Path] = {}
            for shard_id, records in records_by_shard.items():
                target = root / f"{shard_id}.results.jsonl"
                rows = [terminal_result(records[0], shard_id, "SUCCEEDED_OFFLINE_OMPL")]
                if len(records) > 1:
                    rows.append(terminal_result(records[0], shard_id, "SUCCEEDED_OFFLINE_OMPL"))
                self.write_rows(target, rows)
                result_paths[shard_id] = target
            report = aggregator.aggregate_results(manifest, result_paths)
            self.assertFalse(report["valid"])
            self.assertTrue(any("duplicate result" in error for error in report["errors"]), report["errors"])
            self.assertTrue(any("missing" in error for error in report["errors"]), report["errors"])

    def test_scene_hash_mismatch_and_nonterminal_status_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, records_by_shard = self.make_partition(root)
            result_paths: dict[str, Path] = {}
            for shard_id, records in records_by_shard.items():
                rows = [terminal_result(record, shard_id, "SUCCEEDED_OFFLINE_OMPL") for record in records]
                target = root / f"{shard_id}.results.jsonl"
                result_paths[shard_id] = target
                self.write_rows(target, rows)
            first_shard = sorted(result_paths)[0]
            first_rows = [terminal_result(record, first_shard, "SUCCEEDED_OFFLINE_OMPL") for record in records_by_shard[first_shard]]
            first_rows[0]["scene_hash"] = "fabricated-hash"
            first_rows[0]["status"] = "READY_FOR_OFFLINE_OMPL"
            first_rows[0]["planning_duration_s"] = None
            first_rows[0]["trajectory_sha256"] = None
            self.write_rows(result_paths[first_shard], first_rows)
            report = aggregator.aggregate_results(manifest, result_paths)
            self.assertFalse(report["valid"])
            self.assertTrue(any("scene_hash mismatch" in error for error in report["errors"]), report["errors"])
            self.assertTrue(any("nonterminal" in error for error in report["errors"]), report["errors"])


if __name__ == "__main__":
    unittest.main()
