#!/usr/bin/env python3
"""Report the evidence still required before independent LeRobot conversion.

The tool only reads JSON/JSONL produced by this project.  It never starts ROS
or Gazebo, downloads packages, invents a trajectory, or changes an episode.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import read_jsonl, validate_episode_directory


TEACHER_REQUEST_SCHEMA = "three_obstacles_teacher_request/v1"
TEACHER_TRAJECTORY_SCHEMA = "three_obstacles_teacher_trajectory/v1"
REQUIRED_GATE_ORDER = ("PYRAMID_OUTER", "RECTANGLE_FRONT", "CYLINDER_OUTER", "DISH_PLACE")

# This is a contract for the project-owned future OMPL recorder.  It deliberately
# requires evidence instead of treating a candidate scene as a trajectory.
REQUIRED_TRAJECTORY_FIELDS = {
    "record": ("schema_version", "episode_id", "scene_seed", "scene_hash", "planner", "planning_group", "trajectory", "safety"),
    "trajectory": ("joint_names", "points"),
    "point": ("time_from_start_s", "positions_rad", "velocities_rad_s"),
    "safety": ("planning_scene_state_validity_safe", "payload_envelope_clearance_m", "route_gate_order"),
}


def _jsonl_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        path = Path(input_path)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("*.jsonl")))
        else:
            raise FileNotFoundError(f"input does not exist: {path}")
    return sorted(set(paths))


def _missing(mapping: Mapping[str, Any], required: Iterable[str]) -> list[str]:
    return [key for key in required if key not in mapping]


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_trajectory_record(record: Mapping[str, Any]) -> list[str]:
    """Return missing/invalid evidence; an empty list means replay-ready."""
    errors: list[str] = []
    if not isinstance(record, Mapping):
        return ["record must be an object"]
    for field in _missing(record, REQUIRED_TRAJECTORY_FIELDS["record"]):
        errors.append(f"missing record.{field}")
    if record.get("schema_version") != TEACHER_TRAJECTORY_SCHEMA:
        errors.append("record.schema_version is unsupported")
    if not isinstance(record.get("episode_id"), str) or not record.get("episode_id"):
        errors.append("record.episode_id must be non-empty")
    if not isinstance(record.get("scene_seed"), int) or isinstance(record.get("scene_seed"), bool):
        errors.append("record.scene_seed must be an integer")
    if not isinstance(record.get("scene_hash"), str) or not record.get("scene_hash"):
        errors.append("record.scene_hash must be non-empty")
    if not isinstance(record.get("planner"), str) or not record.get("planner"):
        errors.append("record.planner must be non-empty")
    if not isinstance(record.get("planning_group"), str) or not record.get("planning_group"):
        errors.append("record.planning_group must be non-empty")

    trajectory = record.get("trajectory")
    if not isinstance(trajectory, Mapping):
        errors.append("record.trajectory must be an object")
    else:
        for field in _missing(trajectory, REQUIRED_TRAJECTORY_FIELDS["trajectory"]):
            errors.append(f"missing trajectory.{field}")
        names = trajectory.get("joint_names")
        if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            errors.append("trajectory.joint_names must be a non-empty unique string list")
            names = []
        points = trajectory.get("points")
        if not isinstance(points, list) or len(points) < 2:
            errors.append("trajectory.points must contain at least two points")
        elif isinstance(names, list):
            previous_time: float | None = None
            for index, point in enumerate(points):
                if not isinstance(point, Mapping):
                    errors.append(f"trajectory.points[{index}] must be an object")
                    continue
                for field in _missing(point, REQUIRED_TRAJECTORY_FIELDS["point"]):
                    errors.append(f"missing trajectory.points[{index}].{field}")
                time_s = point.get("time_from_start_s")
                if not _finite_number(time_s) or float(time_s) < 0:
                    errors.append(f"trajectory.points[{index}].time_from_start_s must be non-negative")
                elif previous_time is not None and float(time_s) <= previous_time:
                    errors.append("trajectory point times must strictly increase")
                else:
                    previous_time = float(time_s)
                for key in ("positions_rad", "velocities_rad_s"):
                    values = point.get(key)
                    if not isinstance(values, list) or len(values) != len(names) or not all(_finite_number(value) for value in values):
                        errors.append(f"trajectory.points[{index}].{key} must match joint_names")

    safety = record.get("safety")
    if not isinstance(safety, Mapping):
        errors.append("record.safety must be an object")
    else:
        for field in _missing(safety, REQUIRED_TRAJECTORY_FIELDS["safety"]):
            errors.append(f"missing safety.{field}")
        if safety.get("planning_scene_state_validity_safe") is not True:
            errors.append("safety.planning_scene_state_validity_safe must be true")
        if not _finite_number(safety.get("payload_envelope_clearance_m")) or float(safety.get("payload_envelope_clearance_m", -1)) < 0:
            errors.append("safety.payload_envelope_clearance_m must be non-negative")
        if tuple(safety.get("route_gate_order", ())) != REQUIRED_GATE_ORDER:
            errors.append("safety.route_gate_order must match the required four-gate sequence")
    return errors


def summarize_candidate_requests(paths: Iterable[Path]) -> dict[str, Any]:
    """Count requested scenes; requests are not mistaken for generated data."""
    files = _jsonl_paths(paths)
    status_counts: Counter[str] = Counter()
    episode_ids: list[str] = []
    malformed: list[str] = []
    precheck_passed = 0
    for path in files:
        for line, record in enumerate(read_jsonl(path), start=1):
            if record.get("schema_version") != TEACHER_REQUEST_SCHEMA:
                malformed.append(f"{path}:{line}: unsupported request schema")
                continue
            episode_id = record.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                malformed.append(f"{path}:{line}: missing episode_id")
                continue
            episode_ids.append(episode_id)
            status_counts[str(record.get("status", "MISSING_STATUS"))] += 1
            if isinstance(record.get("precheck"), Mapping) and record["precheck"].get("passed") is True:
                precheck_passed += 1
    duplicate_ids = sorted(identifier for identifier, count in Counter(episode_ids).items() if count > 1)
    return {
        "input_files": [str(path) for path in files],
        "candidate_scene_records": len(episode_ids),
        "unique_candidate_scene_ids": len(set(episode_ids)),
        "duplicate_candidate_scene_ids": duplicate_ids,
        "status_counts": dict(sorted(status_counts.items())),
        "static_precheck_passed": precheck_passed,
        "malformed_records": malformed,
        "candidate_scenes_are_not_trajectories": True,
    }


def summarize_trajectories(paths: Iterable[Path], candidate_ids: set[str]) -> dict[str, Any]:
    files = _jsonl_paths(paths)
    valid_ids: list[str] = []
    invalid: dict[str, list[str]] = {}
    unmatched: list[str] = []
    total = 0
    for path in files:
        for line, record in enumerate(read_jsonl(path), start=1):
            total += 1
            label = f"{path}:{line}"
            errors = validate_trajectory_record(record)
            episode_id = record.get("episode_id") if isinstance(record.get("episode_id"), str) else label
            if errors:
                invalid[label] = errors
                continue
            valid_ids.append(episode_id)
            if candidate_ids and episode_id not in candidate_ids:
                unmatched.append(episode_id)
    ready = sorted(set(valid_ids) - set(unmatched))
    return {
        "input_files": [str(path) for path in files],
        "trajectory_records": total,
        "replay_ready_trajectory_records": len(ready),
        "replay_ready_episode_ids": ready,
        "invalid_trajectory_records": invalid,
        "trajectory_without_matching_candidate": sorted(set(unmatched)),
        "required_trajectory_fields": REQUIRED_TRAJECTORY_FIELDS,
    }


def _episode_dirs(roots: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"raw episode root does not exist: {root}")
        if (root / "manifest.json").is_file():
            result.append(root)
        else:
            result.extend(sorted(path.parent for path in root.rglob("manifest.json")))
    return sorted(set(result))


def summarize_raw_episodes(roots: Iterable[Path]) -> dict[str, Any]:
    directories = _episode_dirs(roots)
    valid = 0
    accepted_ids: list[str] = []
    rejected_ids: list[str] = []
    invalid: dict[str, list[str]] = {}
    for directory in directories:
        report = validate_episode_directory(directory)
        if not report.valid:
            invalid[str(directory)] = list(report.errors)
            continue
        valid += 1
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if result.get("status") == "accepted":
            accepted_ids.append(str(metadata["episode_id"]))
        else:
            rejected_ids.append(str(metadata["episode_id"]))
    return {
        "raw_episode_directories": len(directories),
        "valid_raw_episodes": valid,
        "accepted_physical_replays": len(accepted_ids),
        "eligible_for_lerobot_conversion": sorted(accepted_ids),
        "not_eligible_rejected_physical_replays": sorted(rejected_ids),
        "invalid_raw_episodes": invalid,
    }


def build_readiness_report(
    candidate_requests: Iterable[Path],
    trajectory_records: Iterable[Path] = (),
    raw_episode_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    candidate_requests = list(candidate_requests)
    trajectory_records = list(trajectory_records)
    raw_episode_roots = list(raw_episode_roots)
    candidates = summarize_candidate_requests(candidate_requests)
    candidate_ids = set()
    for path in _jsonl_paths(candidate_requests):
        candidate_ids.update(
            record["episode_id"]
            for record in read_jsonl(path)
            if record.get("schema_version") == TEACHER_REQUEST_SCHEMA and isinstance(record.get("episode_id"), str)
        )
    trajectories = summarize_trajectories(trajectory_records, candidate_ids)
    episodes = summarize_raw_episodes(raw_episode_roots)
    return {
        "schema_version": "three_obstacles_dataset_readiness/v1",
        "candidate_scenes": candidates,
        "teacher_trajectories": trajectories,
        "physical_replays": episodes,
        "summary": {
            "candidate_scene_count": candidates["candidate_scene_records"],
            "trajectory_ready_count": trajectories["replay_ready_trajectory_records"],
            "lerobot_eligible_episode_count": len(episodes["eligible_for_lerobot_conversion"]),
            "next_gate": (
                "generate_and_validate_ompl_trajectories"
                if trajectories["replay_ready_trajectory_records"] == 0
                else "run_gazebo_physical_replays"
                if not episodes["eligible_for_lerobot_conversion"]
                else "convert_only_eligible_raw_episodes_in_a_separate_lerobot_environment"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-requests", type=Path, nargs="+", required=True)
    parser.add_argument("--trajectory-records", type=Path, nargs="*", default=[])
    parser.add_argument("--raw-episodes-root", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    report = build_readiness_report(args.candidate_requests, args.trajectory_records, args.raw_episodes_root)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
