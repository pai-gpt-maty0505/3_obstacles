#!/usr/bin/env python3
"""Generate one Gazebo world at a sampled scene's *initial* poses.

The generated SDF is intended to be selected before starting Gazebo.  It never
connects to a running simulator and therefore cannot move, teleport, attach,
or otherwise alter an object during an episode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ENTITY_TO_INCLUDE = {
    "payload": "payload",
    "dish": "dish",
    "cylinder": "cylinder",
    "rectangular_prism": "tall_rectangular_prism",
    "triangular_pyramid": "triangular_pyramid",
}
EXPECTED_ENTITIES = frozenset(ENTITY_TO_INCLUDE)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = PROJECT_ROOT / "three_obstacles/gazebo/worlds/three_obstacles_baseline.world"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "three_obstacles/gazebo/generated_worlds"


class EpisodeWorldError(ValueError):
    """A candidate scene cannot safely become an initial Gazebo world."""


def load_episode(requests_path: Path, episode_id: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    try:
        lines = requests_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EpisodeWorldError(f"cannot read candidate requests: {requests_path}") from error
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise EpisodeWorldError(f"{requests_path}:{number}: invalid JSON") from error
        if isinstance(record, dict) and record.get("episode_id") == episode_id:
            matches.append(record)
    if len(matches) != 1:
        raise EpisodeWorldError(f"episode_id must occur exactly once: {episode_id}")
    return matches[0]


def _finite_pose(value: Any, entity: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise EpisodeWorldError(f"{entity} pose must be an object")
    fields = ("x", "y", "z", "yaw_deg")
    try:
        pose = {field: float(value[field]) for field in fields}
    except (KeyError, TypeError, ValueError) as error:
        raise EpisodeWorldError(f"{entity} pose is incomplete") from error
    if not all(math.isfinite(component) for component in pose.values()):
        raise EpisodeWorldError(f"{entity} pose must be finite")
    return pose


def validate_episode(record: dict[str, Any]) -> dict[str, dict[str, float]]:
    if record.get("schema_version") != "three_obstacles_teacher_request/v1":
        raise EpisodeWorldError("unsupported candidate request schema")
    if record.get("status") != "PENDING_OMPL":
        raise EpisodeWorldError("only unmodified PENDING_OMPL candidate scenes may be spawned")
    if record.get("coordinate_frame") != "base_link":
        raise EpisodeWorldError("candidate frame must be base_link")
    precheck = record.get("precheck")
    if not isinstance(precheck, dict) or precheck.get("passed") is not True:
        raise EpisodeWorldError("candidate static precheck did not pass")
    poses = record.get("entity_poses")
    if not isinstance(poses, dict) or set(poses) != EXPECTED_ENTITIES:
        raise EpisodeWorldError("candidate entity set does not match the physical world")
    return {entity: _finite_pose(poses[entity], entity) for entity in ENTITY_TO_INCLUDE}


def _pose_text(pose: dict[str, float]) -> str:
    # Candidate yaw is in degrees, while SDF uses radians.  Roll/pitch remain
    # zero because only the initial tabletop layout is randomized.
    return "{:.6f} {:.6f} {:.6f} 0 0 {:.9f}".format(
        pose["x"], pose["y"], pose["z"], math.radians(pose["yaw_deg"])
    )


def build_episode_world(template_path: Path, record: dict[str, Any]) -> str:
    """Return SDF text with sampled poses baked in before simulation starts."""
    poses = validate_episode(record)
    try:
        tree = ET.parse(template_path)
    except (OSError, ET.ParseError) as error:
        raise EpisodeWorldError(f"cannot parse world template: {template_path}") from error
    root = tree.getroot()
    world = root.find("world")
    if root.tag != "sdf" or world is None:
        raise EpisodeWorldError("world template does not contain an SDF world")
    episode_id = record.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise EpisodeWorldError("candidate has no usable episode_id")
    world.set("name", f"three_obstacles_{episode_id}")

    includes: dict[str, ET.Element] = {}
    for include in world.findall("include"):
        name = include.findtext("name")
        if name:
            includes[name] = include
    for entity, include_name in ENTITY_TO_INCLUDE.items():
        include = includes.get(include_name)
        if include is None:
            raise EpisodeWorldError(f"template lacks include named {include_name}")
        pose_element = include.find("pose")
        if pose_element is None:
            pose_element = ET.SubElement(include, "pose")
        pose_element.text = _pose_text(poses[entity])

    xml = ET.tostring(root, encoding="unicode")
    forbidden_tokens = ("<joint", "attach", "follow", "teleport", "disable_collisions")
    lowered = xml.lower()
    if any(token in lowered for token in forbidden_tokens):
        raise EpisodeWorldError("generated world contains a prohibited transport mechanism")
    return "<?xml version=\"1.0\"?>\n" + xml + "\n"


def safe_output_path(path: Path) -> Path:
    resolved = path.resolve()
    root = DEFAULT_OUTPUT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise EpisodeWorldError(f"output must remain below {root}") from error
    if resolved.suffix != ".world":
        raise EpisodeWorldError("generated world output must end with .world")
    return resolved


def write_world(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(contents)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True, help="candidate request JSONL")
    parser.add_argument("--episode-id", required=True, help="candidate scene to bake into a fresh SDF world")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="baseline SDF world template")
    parser.add_argument("--output", type=Path, required=True, help=".world path below gazebo/generated_worlds/")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record = load_episode(args.requests, args.episode_id)
        destination = safe_output_path(args.output)
        write_world(destination, build_episode_world(args.template, record))
    except EpisodeWorldError as error:
        print(f"error: {error}")
        return 2
    print(json.dumps({
        "episode_id": args.episode_id,
        "scene_hash": record["scene_hash"],
        "output": str(destination),
        "application_time": "before_gazebo_start_only",
        "teleport_or_attachment": "not_used",
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
