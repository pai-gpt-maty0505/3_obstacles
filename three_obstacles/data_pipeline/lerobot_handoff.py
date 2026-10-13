"""Validated raw-data handoff contract for a later, separate LeRobot export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import LEROBOT_HANDOFF_SCHEMA_VERSION, EpisodeValidationError, read_jsonl, validate_episode_directory


def build_handoff_contract(episode_dir: Path) -> dict[str, Any]:
    """Describe mappings only; deliberately never imports or writes LeRobot data."""
    episode_dir = Path(episode_dir)
    report = validate_episode_directory(episode_dir)
    if not report.valid:
        raise EpisodeValidationError("cannot hand off invalid raw episode: " + "; ".join(report.errors))
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = read_jsonl(episode_dir / "frames.jsonl")
    first = frames[0]
    return {
        "schema_version": LEROBOT_HANDOFF_SCHEMA_VERSION,
        "conversion_status": "contract_only_no_lerobot_import",
        "source": {
            "episode_id": metadata["episode_id"],
            "raw_episode_directory": episode_dir.name,
            "scene_seed": metadata["scene_seed"],
            "artifact_hashes": metadata["artifact_hashes"],
            "raw_manifest": manifest,
        },
        "observation_mapping": {
            "rgb_cameras": {name: f"frames[*].rgb.{name}.path" for name in sorted(first["rgb"])},
            "robot_joint_position_order": sorted(first["robot_state"]["joint_positions_rad"]),
            "gripper_position_order": sorted(first["robot_state"]["gripper"]["finger_positions_rad"]),
            "payload_pose_source": "frames[*].payload_pose",
            "contacts_source": "frames[*].contacts",
        },
        "action_mapping": {
            "joint_target_order": sorted(first["teacher_action"]["joint_position_targets_rad"]),
            "gripper_target_source": "frames[*].teacher_action.gripper_position_target_rad",
            "phase_source": "frames[*].teacher_action.phase",
        },
        "integrity_gate": {
            "required_raw_validation": "pass",
            "dynamic_payload_only": True,
            "no_teleport_or_forced_attachment": True,
            "collision_and_placement_verified": True,
        },
        "frame_count": len(frames),
    }


def write_handoff_contract(episode_dir: Path, destination: Path) -> Path:
    contract = build_handoff_contract(episode_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return destination
