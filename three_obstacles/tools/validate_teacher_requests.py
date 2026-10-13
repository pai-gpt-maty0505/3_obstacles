#!/usr/bin/env python3
"""Validate the structural and conservative geometry checks of teacher requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "three_obstacles_teacher_request/v1"
EXPECTED_GATES = ["PYRAMID_OUTER", "RECTANGLE_FRONT", "CYLINDER_OUTER", "DISH_PLACE"]
EXPECTED_STATUS = "PENDING_OMPL"


def validate_record(record: dict[str, Any], line_number: int) -> list[str]:
    errors: list[str] = []
    prefix = f"line {line_number}"
    if record.get("schema_version") != SCHEMA:
        errors.append(f"{prefix}: unexpected schema_version")
    if record.get("status") != EXPECTED_STATUS:
        errors.append(f"{prefix}: status must be {EXPECTED_STATUS}")
    if record.get("split") not in {"train", "val", "test"}:
        errors.append(f"{prefix}: invalid split")

    poses = record.get("entity_poses")
    if not isinstance(poses, dict) or set(poses) != {
        "payload", "dish", "cylinder", "rectangular_prism", "triangular_pyramid"
    }:
        errors.append(f"{prefix}: missing or unexpected entity poses")
    else:
        for name, pose in poses.items():
            if not isinstance(pose, dict) or set(pose) != {"x", "y", "z", "yaw_deg"}:
                errors.append(f"{prefix}: malformed pose for {name}")

    gates = record.get("route_gates")
    gate_ids = [gate.get("id") for gate in gates] if isinstance(gates, list) else []
    if gate_ids != EXPECTED_GATES:
        errors.append(f"{prefix}: incorrect route gate order: {gate_ids!r}")

    precheck = record.get("precheck")
    if not isinstance(precheck, dict) or precheck.get("passed") is not True:
        errors.append(f"{prefix}: failed or missing static precheck")
    elif precheck.get("route_minimum_clearance_m", -1.0) < precheck.get("required_static_clearance_m", 0.0):
        errors.append(f"{prefix}: route clearance below required threshold")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path, help="teacher request JSONL file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_errors: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        with args.requests.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    all_errors.append(f"line {line_number}: invalid JSON: {error.msg}")
                    continue
                records.append(record)
                all_errors.extend(validate_record(record, line_number))
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if not records:
        all_errors.append("no records found")
    ids = [record.get("episode_id") for record in records]
    if len(ids) != len(set(ids)):
        all_errors.append("duplicate episode_id detected")
    hashes = [record.get("scene_hash") for record in records]
    if len(hashes) != len(set(hashes)):
        all_errors.append("duplicate scene_hash detected")

    if all_errors:
        print("INVALID")
        for error in all_errors:
            print(f"- {error}")
        return 1

    split_counts = {name: sum(record["split"] == name for record in records) for name in ("train", "val", "test")}
    print(json.dumps({"status": "VALID", "count": len(records), "splits": split_counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
