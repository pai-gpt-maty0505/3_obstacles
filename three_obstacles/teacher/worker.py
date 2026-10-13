#!/usr/bin/env python3
"""Shard-safe offline-teacher preparation worker.

The current implementation only builds and validates portable PlanningScene
documents.  It intentionally does not call OMPL or any ROS service.  A later
dedicated *offline* executor can replace ``READY_FOR_OFFLINE_OMPL`` with a
terminal result while preserving this one-result-per-input record contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from .scene_builder import TeacherSceneError, build_finalized_teacher_scene, load_mission_spec, load_teacher_requests


RESULT_SCHEMA = "three_obstacles_teacher_worker_result/v1"
TEACHER_ROOT = Path(__file__).resolve().parent


def result_for_request(request: dict[str, Any], mission_spec: dict[str, Any], shard_id: str) -> dict[str, Any]:
    """Prepare a single request without launching a planner or a robot."""
    episode_id = request.get("episode_id")
    scene_hash = request.get("scene_hash")
    try:
        scene = build_finalized_teacher_scene(request, mission_spec)
    except (TeacherSceneError, ValueError, KeyError, TypeError) as error:
        return {
            "schema_version": RESULT_SCHEMA,
            "shard_id": shard_id,
            "episode_id": episode_id,
            "scene_hash": scene_hash,
            "status": "REJECTED_SCENE_CONTRACT",
            "planning_duration_s": None,
            "failure_reason": str(error),
            "scene_document_sha256": None,
            "worker_mode": "preparation_only_no_ompl_no_robot",
        }
    return {
        "schema_version": RESULT_SCHEMA,
        "shard_id": shard_id,
        "episode_id": episode_id,
        "scene_hash": scene_hash,
        "status": "READY_FOR_OFFLINE_OMPL",
        "planning_duration_s": None,
        "failure_reason": None,
        "scene_document_sha256": scene["document_sha256"],
        "worker_mode": "preparation_only_no_ompl_no_robot",
    }


def process_shard(records: Iterable[dict[str, Any]], mission_spec: dict[str, Any], shard_id: str) -> list[dict[str, Any]]:
    return [result_for_request(record, mission_spec, shard_id) for record in records]


def safe_output(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(TEACHER_ROOT)
    except ValueError as error:
        raise TeacherSceneError(f"result output must remain below {TEACHER_ROOT}") from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-shard", type=Path, required=True, help="one JSONL request shard")
    parser.add_argument("--result-jsonl", type=Path, required=True, help="output JSONL below teacher/")
    parser.add_argument("--shard-id", required=True, help="stable identifier, e.g. pilot-000")
    parser.add_argument("--mission-spec", type=Path, default=Path(__file__).resolve().parents[1] / "config/mission_spec.json")
    args = parser.parse_args()
    try:
        records = load_teacher_requests(args.input_shard)
        result_rows = process_shard(records, load_mission_spec(args.mission_spec), args.shard_id)
        destination = safe_output(args.result_jsonl)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            for row in result_rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"shard_id": args.shard_id, "records": len(result_rows), "result": str(destination)}, ensure_ascii=False))
    except TeacherSceneError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
