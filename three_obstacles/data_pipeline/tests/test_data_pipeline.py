from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from three_obstacles.data_pipeline.lerobot_handoff import build_handoff_contract, write_handoff_contract
from three_obstacles.data_pipeline.schema import EpisodeValidationError, validate_episode_directory
from three_obstacles.data_pipeline.writer import PhysicalReplayEpisodeWriter


def metadata() -> dict:
    return {
        "schema_version": "three_obstacles_physical_replay/v1",
        "episode_id": "physical_replay_000001",
        "scene_seed": 20261013,
        "created_at_utc": "2026-10-13T09:00:00+00:00",
        "coordinate_frames": {"world": "world", "robot_base": "base_link", "payload": "payload"},
        "artifact_hashes": {
            "mission_config": "a" * 64,
            "robot_model": "b" * 64,
            "world_model": "c" * 64,
            "teacher_trajectory": "d" * 64,
        },
        "physics_contract": {
            "payload_mode": "dynamic_rigid_body",
            "attachment_mode": "friction_contact_only",
            "collision_filtering": "enabled",
            "teleportation": "prohibited",
            "forced_attachment": "prohibited",
            "gripper_pad_friction_coefficient": 0.5,
        },
        "quality_limits": {
            "max_payload_speed_m_s": 0.5,
            "teleport_position_tolerance_m": 0.001,
            "max_slip_speed_m_s": 0.05,
            "max_placement_center_error_m": 0.007,
            "max_placement_tilt_deg": 5.0,
            "minimum_rest_duration_s": 0.5,
        },
    }


def frame(index: int, sim_time_s: float, payload_x: float) -> dict:
    timestamp = f"2026-10-13T09:00:0{index}+00:00"
    joints = {"shoulder_pan": 0.1, "shoulder_lift": -0.2, "elbow_flex": 0.3}
    return {
        "frame_index": index,
        "sim_time_s": sim_time_s,
        "recorded_at_utc": timestamp,
        "rgb": {"overhead": {"path": f"rgb/overhead/frame_{index:06d}.png", "capture_timestamp_utc": timestamp, "sha256": "e" * 64}},
        "robot_state": {
            "joint_positions_rad": joints,
            "joint_velocities_rad_s": {name: 0.0 for name in joints},
            "gripper": {"finger_positions_rad": {"gripper": 0.1, "gripper_left": 0.1}, "command": "close"},
        },
        "teacher_action": {"joint_position_targets_rad": joints, "gripper_position_target_rad": 0.1, "phase": "carry"},
        "payload_pose": {"frame_id": "world", "position_m": [payload_x, 0.0, 0.08], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "contacts": [{"body_a": "payload", "body_b": "gripper_left_pad", "normal_force_n": 2.0, "tangential_speed_m_s": 0.01}],
        "checks": {
            "payload_in_gripper_contact": True,
            "slip_detected": False,
            "teleport_detected": False,
            "penetration_detected": False,
            "forbidden_collision_detected": False,
        },
    }


def accepted_summary(duration: float) -> dict:
    return {
        "status": "accepted",
        "duration_s": duration,
        "collision_result": {"moveit_state_validity_safe": True, "gazebo_forbidden_collision_free": True, "all_samples_checked": True},
        "placement_result": {"placed_on_dish": True, "center_error_m": 0.003, "tilt_deg": 1.0, "release_confirmed": True, "rest_duration_s": 0.6},
        "integrity_result": {"teleport_detected": False, "penetration_detected": False, "forced_attachment_detected": False, "collision_filter_disabled": False},
    }


class DataPipelineTests(unittest.TestCase):
    def test_valid_episode_writes_and_creates_le_robot_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode_dir = Path(temporary) / "episode_000001"
            writer = PhysicalReplayEpisodeWriter(episode_dir, metadata())
            writer.append_frame(frame(0, 0.0, 0.0))
            writer.append_frame(frame(1, 0.1, 0.01))
            writer.finalize(accepted_summary(0.1))
            report = validate_episode_directory(episode_dir)
            self.assertTrue(report.valid, report.errors)
            handoff = build_handoff_contract(episode_dir)
            self.assertEqual(handoff["conversion_status"], "contract_only_no_lerobot_import")
            self.assertEqual(handoff["frame_count"], 2)
            output = write_handoff_contract(episode_dir, Path(temporary) / "handoff.json")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["source"]["episode_id"], "physical_replay_000001")

    def test_undeclared_payload_teleport_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = PhysicalReplayEpisodeWriter(Path(temporary) / "jump", metadata())
            writer.append_frame(frame(0, 0.0, 0.0))
            with self.assertRaises(EpisodeValidationError) as raised:
                writer.append_frame(frame(1, 0.1, 0.5))
            self.assertIn("teleport bound", str(raised.exception))

    def test_manifest_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode_dir = Path(temporary) / "tampered"
            writer = PhysicalReplayEpisodeWriter(episode_dir, metadata())
            writer.append_frame(frame(0, 0.0, 0.0))
            writer.append_frame(frame(1, 0.1, 0.01))
            writer.finalize(accepted_summary(0.1))
            result = episode_dir / "result.json"
            result.write_text(result.read_text(encoding="utf-8").replace('"accepted"', '"rejected"'), encoding="utf-8")
            self.assertIn("manifest hash mismatch for result.json", validate_episode_directory(episode_dir).errors)

    def test_accepted_episode_cannot_hide_forced_attachment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = PhysicalReplayEpisodeWriter(Path(temporary) / "bad", metadata())
            writer.append_frame(frame(0, 0.0, 0.0))
            writer.append_frame(frame(1, 0.1, 0.01))
            summary = accepted_summary(0.1)
            summary["integrity_result"]["forced_attachment_detected"] = True
            with self.assertRaises(EpisodeValidationError) as raised:
                writer.finalize(summary)
            self.assertIn("forbidden physical-integrity event", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
