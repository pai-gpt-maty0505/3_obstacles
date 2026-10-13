"""ROS/Gazebo/LeRobot-independent schema validation for physical replays."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


EPISODE_SCHEMA_VERSION = "three_obstacles_physical_replay/v1"
MANIFEST_SCHEMA_VERSION = "three_obstacles_raw_episode_manifest/v1"
LEROBOT_HANDOFF_SCHEMA_VERSION = "three_obstacles_lerobot_handoff/v1"
REQUIRED_HASHES = {"mission_config", "robot_model", "world_model", "teacher_trajectory"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EpisodeValidationError(ValueError):
    """The record lacks evidence for a physically valid replay."""


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def _relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    item = PurePosixPath(value)
    return not item.is_absolute() and ".." not in item.parts and item.parts[0] != "."


def _object(value: Any, label: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return None
    return value


def _required(value: Mapping[str, Any], label: str, *keys: str) -> None:
    # Used only after the mapping type was checked.
    missing = [key for key in keys if key not in value]
    if missing:
        raise KeyError(f"{label}: missing {', '.join(missing)}")


def _numeric_mapping(value: Any, label: str, errors: list[str]) -> None:
    item = _object(value, label, errors)
    if item is None:
        return
    if not item:
        errors.append(f"{label} must not be empty")
    for key, number in item.items():
        if not isinstance(key, str) or not key or not _number(number):
            errors.append(f"{label} must map non-empty names to finite numbers")
            return


def _vector(value: Any, count: int, label: str, errors: list[str]) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != count:
        errors.append(f"{label} must have {count} finite numbers")
        return None
    if not all(_number(x) for x in value):
        errors.append(f"{label} must have {count} finite numbers")
        return None
    return tuple(float(x) for x in value)


def validate_metadata(metadata: Mapping[str, Any]) -> ValidationReport:
    errors: list[str] = []
    if not isinstance(metadata, Mapping):
        return ValidationReport(False, ("metadata must be an object",))
    required = ("schema_version", "episode_id", "scene_seed", "created_at_utc", "coordinate_frames", "artifact_hashes", "physics_contract", "quality_limits")
    for key in required:
        if key not in metadata:
            errors.append(f"metadata.{key} is required")
    if metadata.get("schema_version") != EPISODE_SCHEMA_VERSION:
        errors.append("metadata.schema_version is unsupported")
    if not isinstance(metadata.get("episode_id"), str) or not metadata.get("episode_id"):
        errors.append("metadata.episode_id must be a non-empty string")
    if not isinstance(metadata.get("scene_seed"), int) or isinstance(metadata.get("scene_seed"), bool):
        errors.append("metadata.scene_seed must be an integer")
    if not _timestamp(metadata.get("created_at_utc")):
        errors.append("metadata.created_at_utc must be timezone-aware ISO-8601")

    frames = _object(metadata.get("coordinate_frames"), "metadata.coordinate_frames", errors)
    if frames is not None:
        for key in ("world", "robot_base", "payload"):
            if not isinstance(frames.get(key), str) or not frames.get(key):
                errors.append(f"metadata.coordinate_frames.{key} must be a non-empty string")
    hashes = _object(metadata.get("artifact_hashes"), "metadata.artifact_hashes", errors)
    if hashes is not None:
        for key in sorted(REQUIRED_HASHES):
            if not SHA256.fullmatch(str(hashes.get(key, ""))):
                errors.append(f"metadata.artifact_hashes.{key} must be a lowercase SHA-256")

    contract = _object(metadata.get("physics_contract"), "metadata.physics_contract", errors)
    if contract is not None:
        expected = {
            "payload_mode": "dynamic_rigid_body",
            "attachment_mode": "friction_contact_only",
            "collision_filtering": "enabled",
            "teleportation": "prohibited",
            "forced_attachment": "prohibited",
        }
        for key, expected_value in expected.items():
            if contract.get(key) != expected_value:
                errors.append(f"metadata.physics_contract.{key} must equal {expected_value!r}")
        friction = contract.get("gripper_pad_friction_coefficient")
        if not _number(friction) or not math.isclose(float(friction), 0.5, abs_tol=1e-9):
            errors.append("metadata.physics_contract.gripper_pad_friction_coefficient must equal 0.5")

    limits = _object(metadata.get("quality_limits"), "metadata.quality_limits", errors)
    if limits is not None:
        for key in ("max_payload_speed_m_s", "teleport_position_tolerance_m", "max_slip_speed_m_s", "max_placement_center_error_m", "max_placement_tilt_deg", "minimum_rest_duration_s"):
            if not _number(limits.get(key)) or float(limits.get(key, -1.0)) < 0.0:
                errors.append(f"metadata.quality_limits.{key} must be a non-negative finite number")
    return ValidationReport(not errors, tuple(errors))


def _payload_position(frame: Mapping[str, Any], label: str, errors: list[str]) -> tuple[float, float, float] | None:
    pose = _object(frame.get("payload_pose"), f"{label}.payload_pose", errors)
    if pose is None:
        return None
    if not isinstance(pose.get("frame_id"), str) or not pose.get("frame_id"):
        errors.append(f"{label}.payload_pose.frame_id must be a non-empty string")
    position = _vector(pose.get("position_m"), 3, f"{label}.payload_pose.position_m", errors)
    _vector(pose.get("orientation_xyzw"), 4, f"{label}.payload_pose.orientation_xyzw", errors)
    return position if position is None else (position[0], position[1], position[2])


def validate_frame(frame: Mapping[str, Any], metadata: Mapping[str, Any], *, expected_index: int, previous: Mapping[str, Any] | None) -> ValidationReport:
    errors: list[str] = []
    label = f"frame[{expected_index}]"
    if not isinstance(frame, Mapping):
        return ValidationReport(False, (f"{label} must be an object",))
    for key in ("frame_index", "sim_time_s", "recorded_at_utc", "rgb", "robot_state", "teacher_action", "payload_pose", "contacts", "checks"):
        if key not in frame:
            errors.append(f"{label}.{key} is required")
    if frame.get("frame_index") != expected_index:
        errors.append(f"{label}.frame_index must equal {expected_index}")
    if not _number(frame.get("sim_time_s")) or float(frame.get("sim_time_s", -1)) < 0:
        errors.append(f"{label}.sim_time_s must be non-negative and finite")
    if not _timestamp(frame.get("recorded_at_utc")):
        errors.append(f"{label}.recorded_at_utc must be timezone-aware ISO-8601")

    rgb = _object(frame.get("rgb"), f"{label}.rgb", errors)
    if rgb is not None:
        if not rgb:
            errors.append(f"{label}.rgb must include a camera reference")
        for camera, reference in rgb.items():
            item = _object(reference, f"{label}.rgb.{camera}", errors)
            if item is None:
                continue
            if not _relative_path(item.get("path")):
                errors.append(f"{label}.rgb.{camera}.path must be a safe relative path")
            if not _timestamp(item.get("capture_timestamp_utc")):
                errors.append(f"{label}.rgb.{camera}.capture_timestamp_utc must be timezone-aware ISO-8601")
            if "sha256" in item and not SHA256.fullmatch(str(item["sha256"])):
                errors.append(f"{label}.rgb.{camera}.sha256 must be a lowercase SHA-256")

    robot = _object(frame.get("robot_state"), f"{label}.robot_state", errors)
    if robot is not None:
        _numeric_mapping(robot.get("joint_positions_rad"), f"{label}.robot_state.joint_positions_rad", errors)
        _numeric_mapping(robot.get("joint_velocities_rad_s"), f"{label}.robot_state.joint_velocities_rad_s", errors)
        gripper = _object(robot.get("gripper"), f"{label}.robot_state.gripper", errors)
        if gripper is not None:
            _numeric_mapping(gripper.get("finger_positions_rad"), f"{label}.robot_state.gripper.finger_positions_rad", errors)
            if not isinstance(gripper.get("command"), str) or not gripper.get("command"):
                errors.append(f"{label}.robot_state.gripper.command must be non-empty")

    action = _object(frame.get("teacher_action"), f"{label}.teacher_action", errors)
    if action is not None:
        _numeric_mapping(action.get("joint_position_targets_rad"), f"{label}.teacher_action.joint_position_targets_rad", errors)
        if not _number(action.get("gripper_position_target_rad")):
            errors.append(f"{label}.teacher_action.gripper_position_target_rad must be finite")
        if not isinstance(action.get("phase"), str) or not action.get("phase"):
            errors.append(f"{label}.teacher_action.phase must be non-empty")

    position = _payload_position(frame, label, errors)
    contacts = frame.get("contacts")
    if not isinstance(contacts, list):
        errors.append(f"{label}.contacts must be an array")
    else:
        for contact_index, contact in enumerate(contacts):
            item = _object(contact, f"{label}.contacts[{contact_index}]", errors)
            if item is None:
                continue
            if not isinstance(item.get("body_a"), str) or not isinstance(item.get("body_b"), str):
                errors.append(f"{label}.contacts[{contact_index}] needs body_a and body_b")
            for key in ("normal_force_n", "tangential_speed_m_s"):
                if not _number(item.get(key)) or float(item.get(key, -1)) < 0:
                    errors.append(f"{label}.contacts[{contact_index}].{key} must be non-negative and finite")

    checks = _object(frame.get("checks"), f"{label}.checks", errors)
    check_names = ("payload_in_gripper_contact", "slip_detected", "teleport_detected", "penetration_detected", "forbidden_collision_detected")
    if checks is not None:
        for key in check_names:
            if not isinstance(checks.get(key), bool):
                errors.append(f"{label}.checks.{key} must be boolean")

    if previous is not None and position is not None and _number(previous.get("sim_time_s")) and _number(frame.get("sim_time_s")):
        previous_errors: list[str] = []
        previous_position = _payload_position(previous, "previous", previous_errors)
        elapsed = float(frame["sim_time_s"]) - float(previous["sim_time_s"])
        if elapsed <= 0:
            errors.append(f"{label}.sim_time_s must strictly increase")
        elif previous_position is not None:
            limits = metadata.get("quality_limits") if isinstance(metadata.get("quality_limits"), Mapping) else {}
            bound = float(limits.get("max_payload_speed_m_s", 0.0)) * elapsed + float(limits.get("teleport_position_tolerance_m", 0.0))
            if math.dist(previous_position, position) > bound and (checks is None or checks.get("teleport_detected") is not True):
                errors.append(f"{label} payload displacement exceeds teleport bound without detection")
            if isinstance(contacts, list):
                max_slip = float(limits.get("max_slip_speed_m_s", math.inf))
                measured_slip = any(isinstance(item, Mapping) and "payload" in {item.get("body_a"), item.get("body_b")} and _number(item.get("tangential_speed_m_s")) and float(item["tangential_speed_m_s"]) > max_slip for item in contacts)
                if measured_slip and (checks is None or checks.get("slip_detected") is not True):
                    errors.append(f"{label} has excessive contact slip without detection")
    return ValidationReport(not errors, tuple(errors))


def validate_summary(summary: Mapping[str, Any], metadata: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]) -> ValidationReport:
    errors: list[str] = []
    if not isinstance(summary, Mapping):
        return ValidationReport(False, ("summary must be an object",))
    if summary.get("status") not in {"accepted", "rejected"}:
        errors.append("summary.status must be accepted or rejected")
    if not _number(summary.get("duration_s")) or float(summary.get("duration_s", -1)) < 0:
        errors.append("summary.duration_s must be non-negative and finite")
    elif frames and not math.isclose(float(summary["duration_s"]), float(frames[-1]["sim_time_s"]) - float(frames[0]["sim_time_s"]), abs_tol=1e-6):
        errors.append("summary.duration_s must equal observed simulation duration")
    collision = _object(summary.get("collision_result"), "summary.collision_result", errors)
    placement = _object(summary.get("placement_result"), "summary.placement_result", errors)
    integrity = _object(summary.get("integrity_result"), "summary.integrity_result", errors)
    if collision is not None:
        for key in ("moveit_state_validity_safe", "gazebo_forbidden_collision_free", "all_samples_checked"):
            if not isinstance(collision.get(key), bool):
                errors.append(f"summary.collision_result.{key} must be boolean")
    if placement is not None:
        for key in ("placed_on_dish", "release_confirmed"):
            if not isinstance(placement.get(key), bool):
                errors.append(f"summary.placement_result.{key} must be boolean")
        for key in ("center_error_m", "tilt_deg", "rest_duration_s"):
            if not _number(placement.get(key)) or float(placement.get(key, -1)) < 0:
                errors.append(f"summary.placement_result.{key} must be non-negative and finite")
    if integrity is not None:
        for key in ("teleport_detected", "penetration_detected", "forced_attachment_detected", "collision_filter_disabled"):
            if not isinstance(integrity.get(key), bool):
                errors.append(f"summary.integrity_result.{key} must be boolean")
    if summary.get("status") == "accepted":
        if collision is None or not all(collision.get(key) is True for key in ("moveit_state_validity_safe", "gazebo_forbidden_collision_free", "all_samples_checked")):
            errors.append("accepted summary requires all collision checks")
        limits = metadata.get("quality_limits") if isinstance(metadata.get("quality_limits"), Mapping) else {}
        if placement is None or placement.get("placed_on_dish") is not True or placement.get("release_confirmed") is not True or float(placement.get("center_error_m", math.inf)) > float(limits.get("max_placement_center_error_m", -1)) or float(placement.get("tilt_deg", math.inf)) > float(limits.get("max_placement_tilt_deg", -1)) or float(placement.get("rest_duration_s", -1)) < float(limits.get("minimum_rest_duration_s", math.inf)):
            errors.append("accepted summary does not meet placement thresholds")
        if integrity is None or any(integrity.get(key) is True for key in ("teleport_detected", "penetration_detected", "forced_attachment_detected", "collision_filter_disabled")):
            errors.append("accepted summary contains a forbidden physical-integrity event")
        if any(isinstance(frame.get("checks"), Mapping) and any(frame["checks"].get(key) is True for key in ("slip_detected", "teleport_detected", "penetration_detected", "forbidden_collision_detected")) for frame in frames):
            errors.append("accepted summary contains a failed frame check")
    return ValidationReport(not errors, tuple(errors))


def validate_episode_records(metadata: Mapping[str, Any], frames: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> ValidationReport:
    errors = list(validate_metadata(metadata).errors)
    if not frames:
        errors.append("episode must contain at least one frame")
    previous = None
    for index, frame in enumerate(frames):
        report = validate_frame(frame, metadata, expected_index=index, previous=previous)
        errors.extend(report.errors)
        previous = frame
    errors.extend(validate_summary(summary, metadata, frames).errors)
    return ValidationReport(not errors, tuple(errors))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EpisodeValidationError(f"invalid JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise EpisodeValidationError(f"JSONL line {line_number} must be an object")
            records.append(value)
    return records


def validate_episode_directory(episode_dir: Path) -> ValidationReport:
    required = ("metadata.json", "frames.jsonl", "result.json", "manifest.json")
    if not episode_dir.is_dir():
        return ValidationReport(False, (f"episode directory does not exist: {episode_dir}",))
    missing = [name for name in required if not (episode_dir / name).is_file()]
    if missing:
        return ValidationReport(False, tuple(f"missing raw episode file: {name}" for name in missing))
    try:
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        frames = read_jsonl(episode_dir / "frames.jsonl")
        summary = json.loads((episode_dir / "result.json").read_text(encoding="utf-8"))
        manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, EpisodeValidationError) as exc:
        return ValidationReport(False, (str(exc),))
    errors: list[str] = []
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("manifest schema_version is unsupported")
    elif manifest.get("frame_count") != len(frames):
        errors.append("manifest frame_count does not match frames.jsonl")
    hashes = manifest.get("file_hashes") if isinstance(manifest, Mapping) else None
    if not isinstance(hashes, Mapping):
        errors.append("manifest.file_hashes must be an object")
    else:
        for filename in ("metadata.json", "frames.jsonl", "result.json"):
            if hashes.get(filename) != sha256_file(episode_dir / filename):
                errors.append(f"manifest hash mismatch for {filename}")
    if isinstance(metadata, Mapping) and isinstance(summary, Mapping):
        errors.extend(validate_episode_records(metadata, frames, summary).errors)
    else:
        errors.append("metadata.json and result.json must be objects")
    return ValidationReport(not errors, tuple(errors))
