#!/usr/bin/env python3
"""Validate one raw episode without ROS, Gazebo, or LeRobot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .schema import validate_episode_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    args = parser.parse_args()
    report = validate_episode_directory(args.episode_dir)
    if report.valid:
        print(f"VALID: {args.episode_dir}")
        return 0
    print(f"INVALID: {args.episode_dir}", file=sys.stderr)
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
