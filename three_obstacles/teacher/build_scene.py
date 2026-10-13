#!/usr/bin/env python3
"""Prepare one portable PlanningScene JSON document; no planner is run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .scene_builder import TeacherSceneError, build_finalized_teacher_scene, load_mission_spec, load_teacher_requests, project_root, select_episode, serialize_scene


TEACHER_ROOT = Path(__file__).resolve().parent


def safe_output(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(TEACHER_ROOT)
    except ValueError as error:
        raise TeacherSceneError(f"output must remain below {TEACHER_ROOT}") from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--mission-spec", type=Path, default=project_root() / "config/mission_spec.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--print", action="store_true")
    group.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        document = build_finalized_teacher_scene(select_episode(load_teacher_requests(args.requests), args.episode_id), load_mission_spec(args.mission_spec))
        payload = serialize_scene(document)
        if args.print:
            sys.stdout.write(payload)
        else:
            destination = safe_output(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
            print(destination)
    except TeacherSceneError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
