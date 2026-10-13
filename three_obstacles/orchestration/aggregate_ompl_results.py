#!/usr/bin/env python3
"""Strictly aggregate terminal offline-OMPL worker results.

This is a read-only integrity gate.  It does not import ROS, call OMPL, start
Gazebo, make trajectories, or turn a worker claim into a successful result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


PARTITION_SCHEMA = "three_obstacles_teacher_partition/v1"
RESULT_SCHEMA = "three_obstacles_teacher_worker_result/v1"
AGGREGATE_SCHEMA = "three_obstacles_ompl_result_aggregation/v1"
TERMINAL_STATUSES = frozenset(
    {
        "SUCCEEDED_OFFLINE_OMPL",
        "FAILED_OFFLINE_OMPL",
        "REJECTED_SCENE_CONTRACT",
    }
)
SUCCESS_STATUS = "SUCCEEDED_OFFLINE_OMPL"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: result must be an object")
            records.append(record)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_shard_id(worker: Mapping[str, Any]) -> str:
    """Use the partition filename as the stable ID shared by result rows."""
    input_name = worker.get("input")
    if not isinstance(input_name, str) or not input_name:
        raise ValueError("manifest worker input must be a non-empty string")
    return Path(input_name).stem


def _safe_input_path(manifest_dir: Path, relative_name: str) -> Path:
    target = (manifest_dir / relative_name).resolve()
    try:
        target.relative_to(manifest_dir.resolve())
    except ValueError as error:
        raise ValueError(f"manifest worker input escapes partition directory: {relative_name}") from error
    return target


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _request_pair(record: Mapping[str, Any], label: str) -> tuple[str, str]:
    episode_id = record.get("episode_id")
    scene_hash = record.get("scene_hash")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError(f"{label}: episode_id must be a non-empty string")
    if not isinstance(scene_hash, str) or not scene_hash:
        raise ValueError(f"{label}: scene_hash must be a non-empty string")
    return episode_id, scene_hash


def load_partition_requests(manifest_path: Path) -> tuple[dict[str, set[tuple[str, str]]], list[str]]:
    """Read a partition and prove the manifest still matches every input shard."""
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"cannot read partition manifest: {error}"]
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PARTITION_SCHEMA:
        return {}, ["partition manifest has an unsupported schema_version"]
    workers = manifest.get("workers")
    if not isinstance(workers, list) or not workers:
        return {}, ["partition manifest requires a non-empty workers array"]
    if manifest.get("worker_count") != len(workers):
        errors.append("partition manifest worker_count does not match workers")
    expected_total = manifest.get("total_scene_count")
    expected_by_shard: dict[str, set[tuple[str, str]]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    seen_episode_ids: set[str] = set()
    seen_scene_hashes: set[str] = set()
    for index, worker in enumerate(workers):
        label = f"manifest.workers[{index}]"
        if not isinstance(worker, dict):
            errors.append(f"{label} must be an object")
            continue
        try:
            shard_id = canonical_shard_id(worker)
            shard_path = _safe_input_path(manifest_path.parent, worker["input"])
        except (KeyError, ValueError) as error:
            errors.append(f"{label}: {error}")
            continue
        if shard_id in expected_by_shard:
            errors.append(f"duplicate manifest shard ID: {shard_id}")
            continue
        if not shard_path.is_file():
            errors.append(f"missing input shard for {shard_id}: {shard_path}")
            continue
        if worker.get("sha256") != sha256_file(shard_path):
            errors.append(f"input shard checksum mismatch for {shard_id}")
        try:
            records = load_jsonl(shard_path)
        except (OSError, ValueError) as error:
            errors.append(str(error))
            continue
        pairs: set[tuple[str, str]] = set()
        actual_episode_ids: list[str] = []
        for line_number, record in enumerate(records, start=1):
            try:
                pair = _request_pair(record, f"{shard_path}:{line_number}")
            except ValueError as error:
                errors.append(str(error))
                continue
            if pair in pairs:
                errors.append(f"duplicate request pair in shard {shard_id}: {pair[0]}")
            pairs.add(pair)
            actual_episode_ids.append(pair[0])
            if pair in seen_pairs:
                errors.append(f"request appears in multiple shards: {pair[0]}")
            seen_pairs.add(pair)
            if pair[0] in seen_episode_ids:
                errors.append(f"duplicate request episode_id across shards: {pair[0]}")
            seen_episode_ids.add(pair[0])
            if pair[1] in seen_scene_hashes:
                errors.append(f"duplicate request scene_hash across shards: {pair[1]}")
            seen_scene_hashes.add(pair[1])
        if worker.get("scene_count") != len(records):
            errors.append(f"manifest scene_count mismatch for {shard_id}")
        if worker.get("episode_ids") != actual_episode_ids:
            errors.append(f"manifest episode_ids mismatch for {shard_id}")
        expected_by_shard[shard_id] = pairs
    if expected_total != sum(len(items) for items in expected_by_shard.values()):
        errors.append("partition manifest total_scene_count does not match shard contents")
    return expected_by_shard, errors


def validate_terminal_result(record: Mapping[str, Any], expected_shard: str, label: str) -> list[str]:
    """Validate status-specific evidence without accepting a planner placeholder."""
    errors: list[str] = []
    required = ("schema_version", "shard_id", "episode_id", "scene_hash", "status", "planning_duration_s", "failure_reason")
    for key in required:
        if key not in record:
            errors.append(f"{label}: missing {key}")
    if record.get("schema_version") != RESULT_SCHEMA:
        errors.append(f"{label}: unsupported result schema")
    if record.get("shard_id") != expected_shard:
        errors.append(f"{label}: shard_id must equal {expected_shard!r}")
    try:
        _request_pair(record, label)
    except ValueError as error:
        errors.append(str(error))
    status = record.get("status")
    if status not in TERMINAL_STATUSES:
        errors.append(f"{label}: nonterminal or unsupported status {status!r}")
        return errors

    duration = record.get("planning_duration_s")
    reason = record.get("failure_reason")
    trajectory_digest = record.get("trajectory_sha256")
    if status == SUCCESS_STATUS:
        if not _is_number(duration) or float(duration) < 0:
            errors.append(f"{label}: successful result needs non-negative planning_duration_s")
        if reason is not None:
            errors.append(f"{label}: successful result must have failure_reason null")
        if not isinstance(trajectory_digest, str) or not re_full_sha256(trajectory_digest):
            errors.append(f"{label}: successful result needs trajectory_sha256")
    elif status == "FAILED_OFFLINE_OMPL":
        if not _is_number(duration) or float(duration) < 0:
            errors.append(f"{label}: failed result needs non-negative planning_duration_s")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{label}: failed result needs a non-empty failure_reason")
        if trajectory_digest not in (None, ""):
            errors.append(f"{label}: failed result must not claim trajectory_sha256")
    else:  # REJECTED_SCENE_CONTRACT
        if duration is not None and (not _is_number(duration) or float(duration) < 0):
            errors.append(f"{label}: rejected result has invalid planning_duration_s")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{label}: rejected result needs a non-empty failure_reason")
        if trajectory_digest not in (None, ""):
            errors.append(f"{label}: rejected result must not claim trajectory_sha256")
    return errors


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def aggregate_results(manifest_path: Path, result_paths: Mapping[str, Path]) -> dict[str, Any]:
    """Return an audit report; `valid` is false on any gap, duplicate, or claim."""
    manifest_path = Path(manifest_path)
    expected_by_shard, errors = load_partition_requests(manifest_path)
    supplied = {str(shard): Path(path) for shard, path in result_paths.items()}
    expected_shards = set(expected_by_shard)
    for shard_id in sorted(expected_shards - set(supplied)):
        errors.append(f"missing result file for shard {shard_id}")
    for shard_id in sorted(set(supplied) - expected_shards):
        errors.append(f"result file supplied for unknown shard {shard_id}")

    status_counts: Counter[str] = Counter()
    durations: list[float] = []
    terminal_rows = 0
    result_rows = 0
    missing_by_shard: dict[str, list[str]] = {}
    for shard_id, expected_pairs in sorted(expected_by_shard.items()):
        path = supplied.get(shard_id)
        if path is None:
            continue
        if not path.is_file():
            errors.append(f"result file does not exist for {shard_id}: {path}")
            continue
        try:
            rows = load_jsonl(path)
        except (OSError, ValueError) as error:
            errors.append(str(error))
            continue
        observed: set[tuple[str, str]] = set()
        seen_episode_ids: set[str] = set()
        for line_number, row in enumerate(rows, start=1):
            result_rows += 1
            label = f"{path}:{line_number}"
            row_errors = validate_terminal_result(row, shard_id, label)
            errors.extend(row_errors)
            try:
                pair = _request_pair(row, label)
            except ValueError:
                continue
            episode_id, scene_hash = pair
            if episode_id in seen_episode_ids:
                errors.append(f"duplicate result episode_id in {shard_id}: {episode_id}")
            seen_episode_ids.add(episode_id)
            if pair in observed:
                errors.append(f"duplicate result for request in {shard_id}: {episode_id}")
            observed.add(pair)
            matching_request_hashes = {item_hash for item_id, item_hash in expected_pairs if item_id == episode_id}
            if matching_request_hashes and scene_hash not in matching_request_hashes:
                errors.append(f"scene_hash mismatch for {shard_id}/{episode_id}")
            elif pair not in expected_pairs:
                errors.append(f"result does not belong to shard {shard_id}: {episode_id}")
            if not row_errors and row.get("status") in TERMINAL_STATUSES:
                terminal_rows += 1
                status_counts[str(row["status"])] += 1
                if _is_number(row.get("planning_duration_s")):
                    durations.append(float(row["planning_duration_s"]))
        missing = expected_pairs - observed
        if missing:
            missing_by_shard[shard_id] = sorted(f"{episode_id}:{scene_hash}" for episode_id, scene_hash in missing)
            errors.append(f"missing {len(missing)} result(s) for shard {shard_id}")

    timing: dict[str, Any]
    if durations:
        timing = {
            "count": len(durations),
            "total_s": round(sum(durations), 6),
            "mean_s": round(mean(durations), 6),
            "min_s": round(min(durations), 6),
            "max_s": round(max(durations), 6),
        }
    else:
        timing = {"count": 0, "total_s": 0.0, "mean_s": None, "min_s": None, "max_s": None}
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "partition": {
            "manifest": str(manifest_path),
            "shard_count": len(expected_by_shard),
            "requested_scene_count": sum(len(pairs) for pairs in expected_by_shard.values()),
        },
        "results": {
            "result_rows": result_rows,
            "valid_terminal_rows": terminal_rows,
            "status_counts": dict(sorted(status_counts.items())),
            "timing_s": timing,
            "missing_by_shard": missing_by_shard,
            "terminal_statuses": sorted(TERMINAL_STATUSES),
        },
    }


def parse_result_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        shard_id, separator, path_text = value.partition("=")
        if not separator or not shard_id or not path_text:
            raise ValueError("each --result must use SHARD_ID=RESULT_JSONL")
        if shard_id in result:
            raise ValueError(f"duplicate --result mapping for {shard_id}")
        result[shard_id] = Path(path_text)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--result", action="append", default=[], metavar="SHARD_ID=RESULT_JSONL", help="repeat once per partition shard")
    parser.add_argument("--output", type=Path, help="write only a valid aggregation report")
    args = parser.parse_args()
    try:
        result_paths = parse_result_mapping(args.result)
        report = aggregate_results(args.partition_manifest, result_paths)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["valid"]:
        return 1
    if args.output:
        if args.output.exists():
            print(f"error: refusing to overwrite output: {args.output}", file=sys.stderr)
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
