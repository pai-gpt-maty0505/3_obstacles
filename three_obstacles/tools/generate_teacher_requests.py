#!/usr/bin/env python3
"""Generate deterministic, geometry-checked OMPL teacher requests.

This program is intentionally independent of the legacy MLP scripts.  It does
not command a robot, start Gazebo, or create a trajectory.  It only records
candidate scenes and route gates that a later, project-owned MoveIt/OMPL node
must solve and validate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "three_obstacles_teacher_request/v1"
OBSTACLE_NAMES = ("cylinder", "rectangular_prism", "triangular_pyramid")


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float


def load_spec(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        spec = json.load(stream)
    if spec.get("schema_version") != "three_obstacles_mission_spec/v1":
        raise ValueError(f"unsupported mission spec: {spec.get('schema_version')!r}")
    return spec


def rounded(value: float) -> float:
    return round(value, 6)


def assign_scene_level_splits(records: list[dict[str, Any]], fractions: dict[str, float]) -> None:
    """Assign complete scenes to exact, deterministic split counts.

    Sorting by the scene hash rather than generation order keeps the assignment
    reproducible without placing near-neighbour seeds into a common split.
    """
    split_names = ("train", "val", "test")
    total_fraction = sum(fractions[name] for name in split_names)
    if not math.isclose(total_fraction, 1.0, abs_tol=1e-9):
        raise ValueError("dataset split fractions must sum to 1.0")

    raw_counts = {name: len(records) * fractions[name] for name in split_names}
    counts = {name: math.floor(raw_counts[name]) for name in split_names}
    remaining = len(records) - sum(counts.values())
    for name in sorted(split_names, key=lambda key: (raw_counts[key] - counts[key], key), reverse=True)[:remaining]:
        counts[name] += 1

    ordered_records = sorted(records, key=lambda record: record["scene_hash"])
    cursor = 0
    for name in split_names:
        for record in ordered_records[cursor : cursor + counts[name]]:
            record["split"] = name
        cursor += counts[name]


def footprint_radius(name: str, entity: dict[str, Any]) -> float:
    """Return a conservative XY bounding-circle radius for static checks."""
    if name == "cylinder":
        return entity["diameter_m"] / 2.0
    if name == "rectangular_prism":
        width, depth, _ = entity["size_m"]
        return math.hypot(width / 2.0, depth / 2.0)
    if name == "triangular_pyramid":
        # Circumradius of its equilateral triangular base.
        return entity["base_side_m"] / math.sqrt(3.0)
    if name == "payload":
        width, depth, _ = entity["size_m"]
        return math.hypot(width / 2.0, depth / 2.0)
    if name == "dish":
        return entity["outer_diameter_m"] / 2.0
    raise KeyError(f"no footprint radius rule for {name!r}")


def pose_with_jitter(entity: dict[str, Any], rng: random.Random) -> dict[str, float]:
    base = entity["base_pose"]
    xy_jitter = entity["xy_jitter_m"]
    yaw_jitter = entity["yaw_jitter_deg"]
    return {
        "x": rounded(base["x"] + rng.uniform(-xy_jitter, xy_jitter)),
        "y": rounded(base["y"] + rng.uniform(-xy_jitter, xy_jitter)),
        "z": rounded(base["z"]),
        "yaw_deg": rounded(base["yaw_deg"] + rng.uniform(-yaw_jitter, yaw_jitter)),
    }


def distance_point_to_segment(point: Circle, start: tuple[float, float], end: tuple[float, float]) -> float:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(point.x - sx, point.y - sy)
    projection = ((point.x - sx) * dx + (point.y - sy) * dy) / length_sq
    projection = max(0.0, min(1.0, projection))
    closest_x = sx + projection * dx
    closest_y = sy + projection * dy
    return math.hypot(point.x - closest_x, point.y - closest_y)


def make_route_gates(spec: dict[str, Any], poses: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    route = spec["route"]
    height = route["carry_height_m"]
    offset = route["gate_offsets_from_entity_m"]

    pyramid = poses["triangular_pyramid"]
    rectangle = poses["rectangular_prism"]
    cylinder = poses["cylinder"]
    dish = poses["dish"]

    gates_by_id = {
        "PYRAMID_OUTER": {
            "id": "PYRAMID_OUTER",
            "entity": "triangular_pyramid",
            "semantic_side": route["semantic_constraints"]["PYRAMID_OUTER"],
            "position": {
                "x": rounded(pyramid["x"] + offset["PYRAMID_OUTER"][0]),
                "y": rounded(pyramid["y"] + offset["PYRAMID_OUTER"][1]),
                "z": height,
            },
        },
        "RECTANGLE_FRONT": {
            "id": "RECTANGLE_FRONT",
            "entity": "rectangular_prism",
            "semantic_side": route["semantic_constraints"]["RECTANGLE_FRONT"],
            "position": {
                "x": rounded(rectangle["x"] + offset["RECTANGLE_FRONT"][0]),
                "y": rounded(rectangle["y"] + offset["RECTANGLE_FRONT"][1]),
                "z": height,
            },
        },
        "CYLINDER_OUTER": {
            "id": "CYLINDER_OUTER",
            "entity": "cylinder",
            "semantic_side": route["semantic_constraints"]["CYLINDER_OUTER"],
            "position": {
                "x": rounded(cylinder["x"] + offset["CYLINDER_OUTER"][0]),
                "y": rounded(cylinder["y"] + offset["CYLINDER_OUTER"][1]),
                "z": height,
            },
        },
        "DISH_PLACE": {
            "id": "DISH_PLACE",
            "entity": "dish",
            "semantic_side": route["semantic_constraints"]["DISH_PLACE"],
            "position": {"x": dish["x"], "y": dish["y"], "z": height},
        },
    }
    return [gates_by_id[gate_id] for gate_id in route["ordered_gate_ids"]]


def minimum_entity_gap(
    spec: dict[str, Any], poses: dict[str, dict[str, float]], names: Iterable[str]
) -> float:
    circles = [
        Circle(poses[name]["x"], poses[name]["y"], footprint_radius(name, spec["entities"][name]))
        for name in names
    ]
    gaps = []
    for index, first in enumerate(circles):
        for second in circles[index + 1 :]:
            gaps.append(math.hypot(first.x - second.x, first.y - second.y) - first.radius - second.radius)
    return min(gaps) if gaps else math.inf


def route_clearance_check(
    spec: dict[str, Any], poses: dict[str, dict[str, float]], gates: list[dict[str, Any]]
) -> tuple[bool, float]:
    """Conservative 2D payload-envelope test before expensive MoveIt planning."""
    payload = poses["payload"]
    points = [(payload["x"], payload["y"])]
    points.extend((gate["position"]["x"], gate["position"]["y"]) for gate in gates)

    envelope = spec["route"]["payload_envelope_radius_m"]
    required = envelope + spec["route"]["minimum_static_clearance_m"]
    min_clearance = math.inf
    for start, end in zip(points, points[1:]):
        for name in OBSTACLE_NAMES:
            pose = poses[name]
            obstacle = Circle(pose["x"], pose["y"], footprint_radius(name, spec["entities"][name]))
            clearance = distance_point_to_segment(obstacle, start, end) - obstacle.radius - envelope
            min_clearance = min(min_clearance, clearance)
    return min_clearance >= spec["route"]["minimum_static_clearance_m"], min_clearance


def canonical_scene_hash(poses: dict[str, dict[str, float]]) -> str:
    normalized = json.dumps(poses, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def build_request(spec: dict[str, Any], root_seed: int, episode_index: int) -> dict[str, Any]:
    scene_seed = root_seed + episode_index
    rng = random.Random(scene_seed)
    poses = {name: pose_with_jitter(entity, rng) for name, entity in spec["entities"].items()}
    gates = make_route_gates(spec, poses)
    entity_gap = minimum_entity_gap(spec, poses, spec["entities"].keys())
    route_ok, route_clearance = route_clearance_check(spec, poses, gates)
    required_gap = spec["route"]["minimum_static_clearance_m"]
    if entity_gap < required_gap:
        raise ValueError(f"scene objects too close: gap={entity_gap:.6f} m")
    if not route_ok:
        raise ValueError(f"route envelope too close: clearance={route_clearance:.6f} m")

    return {
        "schema_version": SCHEMA,
        "episode_id": f"scene_{episode_index:05d}",
        "scene_seed": scene_seed,
        "split": None,
        "layout_id": "right_to_left_three_obstacle_v1",
        "coordinate_frame": spec["coordinate_frame"]["frame_id"],
        "entity_poses": poses,
        "route_gates": gates,
        "teacher_contract": {
            "planner": spec["teacher"]["planner"],
            "planning_group": spec["teacher"]["planning_group"],
            "online_replanning": spec["teacher"]["online_replanning"],
            "interpolation_hz": spec["teacher"]["trajectory_interpolation_hz"],
            "required_checks": spec["teacher"]["required_checks"],
        },
        "precheck": {
            "entity_minimum_gap_m": rounded(entity_gap),
            "route_minimum_clearance_m": rounded(route_clearance),
            "required_static_clearance_m": required_gap,
            "passed": True,
            "method": "conservative_2d_bounding_circle_precheck",
        },
        "scene_hash": canonical_scene_hash(poses),
        "status": spec["dataset"]["candidate_status"],
    }


def generate_requests(spec: dict[str, Any], count: int, root_seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    requests: list[dict[str, Any]] = []
    max_attempts = spec["dataset"]["scene_resample_limit"]
    for episode_index in range(count):
        for retry in range(max_attempts):
            try:
                # Separate retry bands preserve reproducibility while avoiding a rejected scene.
                candidate = build_request(spec, root_seed + retry * 1_000_000, episode_index)
                requests.append(candidate)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(f"unable to sample a valid scene for episode {episode_index}")
    assign_scene_level_splits(requests, spec["dataset"]["target_splits"])
    return requests


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def split_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for record in records:
        counts[record["split"]] += 1
    return counts


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True, help="number of candidate scenes")
    parser.add_argument("--seed", type=int, required=True, help="root random seed")
    parser.add_argument(
        "--spec",
        type=Path,
        default=repo_root / "three_obstacles/config/mission_spec.json",
        help="mission specification JSON",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSONL destination")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        spec = load_spec(args.spec)
        records = generate_requests(spec, args.count, args.seed)
        write_jsonl(args.output, records)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    summary = {
        "generator": "three_obstacles/tools/generate_teacher_requests.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "root_seed": args.seed,
        "splits": split_counts(records),
        "output": str(args.output),
        "status": "PENDING_OMPL (not yet a trajectory dataset)",
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
