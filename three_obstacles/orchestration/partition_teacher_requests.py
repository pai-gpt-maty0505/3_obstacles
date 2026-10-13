#!/usr/bin/env python3
"""Partition complete teacher-request scenes for independent planner workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: record must be an object")
            records.append(record)
    if not records:
        raise ValueError("no records found")
    return records


def request_key(record: dict[str, Any]) -> str:
    episode_id = record.get("episode_id")
    scene_hash = record.get("scene_hash")
    if not isinstance(episode_id, str) or not isinstance(scene_hash, str):
        raise ValueError("every record requires string episode_id and scene_hash")
    return f"{scene_hash}:{episode_id}"


def partition(records: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    keys = [request_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate scene_hash + episode_id pair")

    shards: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    # Stable balancing: longest shard is always selected, ties select lower worker id.
    for record in sorted(records, key=request_key):
        target = min(range(workers), key=lambda worker_id: (len(shards[worker_id]), worker_id))
        shards[target].append(record)
    return shards


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_shards(shards: list[list[dict[str, Any]]], output_dir: Path, source: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest_workers = []
    for worker_id, records in enumerate(shards):
        filename = f"worker_{worker_id:02d}.jsonl"
        target = output_dir / filename
        with target.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
        manifest_workers.append(
            {
                "worker_id": worker_id,
                "input": filename,
                "scene_count": len(records),
                "sha256": sha256_file(target),
                "episode_ids": [record["episode_id"] for record in records],
            }
        )
    manifest = {
        "schema_version": "three_obstacles_teacher_partition/v1",
        "source": str(source),
        "worker_count": len(shards),
        "total_scene_count": sum(len(records) for records in shards),
        "workers": manifest_workers,
        "result_contract": {
            "one_result_per_input_scene": True,
            "required_fields": ["episode_id", "scene_hash", "status", "planning_duration_s", "failure_reason"],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path, help="validated teacher request JSONL")
    parser.add_argument("--workers", type=int, required=True, help="number of planning-only workers")
    parser.add_argument("--output-dir", type=Path, required=True, help="new shard directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        records = load_jsonl(args.requests)
        shards = partition(records, args.workers)
        manifest = write_shards(shards, args.output_dir, args.requests)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PARTITIONED", **manifest}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
