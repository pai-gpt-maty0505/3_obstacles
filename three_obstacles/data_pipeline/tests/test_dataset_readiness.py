from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from three_obstacles.data_pipeline.dataset_readiness import build_readiness_report, validate_trajectory_record
from three_obstacles.data_pipeline.writer import PhysicalReplayEpisodeWriter


def request(episode_id: str = "scene_00000") -> dict:
    return {
        "schema_version": "three_obstacles_teacher_request/v1",
        "episode_id": episode_id,
        "scene_seed": 20261013,
        "scene_hash": "scene-hash",
        "status": "PENDING_OMPL",
        "precheck": {"passed": True},
    }


def trajectory(episode_id: str = "scene_00000") -> dict:
    return {
        "schema_version": "three_obstacles_teacher_trajectory/v1",
        "episode_id": episode_id,
        "scene_seed": 20261013,
        "scene_hash": "scene-hash",
        "planner": "OMPL RRTConnect",
        "planning_group": "arm",
        "trajectory": {
            "joint_names": ["shoulder_pan", "shoulder_lift"],
            "points": [
                {"time_from_start_s": 0.0, "positions_rad": [0.0, 0.0], "velocities_rad_s": [0.0, 0.0]},
                {"time_from_start_s": 0.1, "positions_rad": [0.1, -0.1], "velocities_rad_s": [0.2, -0.2]},
            ],
        },
        "safety": {
            "planning_scene_state_validity_safe": True,
            "payload_envelope_clearance_m": 0.02,
            "route_gate_order": ["PYRAMID_OUTER", "RECTANGLE_FRONT", "CYLINDER_OUTER", "DISH_PLACE"],
        },
    }


def raw_metadata() -> dict:
    return {
        "schema_version": "three_obstacles_physical_replay/v1",
        "episode_id": "physical_00000",
        "scene_seed": 20261013,
        "created_at_utc": "2026-10-13T09:00:00+00:00",
        "coordinate_frames": {"world": "world", "robot_base": "base_link", "payload": "payload"},
        "artifact_hashes": {"mission_config": "a" * 64, "robot_model": "b" * 64, "world_model": "c" * 64, "teacher_trajectory": "d" * 64},
        "physics_contract": {"payload_mode": "dynamic_rigid_body", "attachment_mode": "friction_contact_only", "collision_filtering": "enabled", "teleportation": "prohibited", "forced_attachment": "prohibited", "gripper_pad_friction_coefficient": 0.5},
        "quality_limits": {"max_payload_speed_m_s": 1.0, "teleport_position_tolerance_m": 0.001, "max_slip_speed_m_s": 0.1, "max_placement_center_error_m": 0.007, "max_placement_tilt_deg": 5.0, "minimum_rest_duration_s": 0.5},
    }


def raw_frame(index: int, time_s: float, x: float) -> dict:
    stamp = f"2026-10-13T09:00:0{index}+00:00"
    joints = {"shoulder_pan": 0.0}
    return {
        "frame_index": index,
        "sim_time_s": time_s,
        "recorded_at_utc": stamp,
        "rgb": {"overhead": {"path": f"rgb/{index}.png", "capture_timestamp_utc": stamp}},
        "robot_state": {"joint_positions_rad": joints, "joint_velocities_rad_s": joints, "gripper": {"finger_positions_rad": {"gripper": 0.1, "gripper_left": 0.1}, "command": "close"}},
        "teacher_action": {"joint_position_targets_rad": joints, "gripper_position_target_rad": 0.1, "phase": "carry"},
        "payload_pose": {"frame_id": "world", "position_m": [x, 0.0, 0.08], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "contacts": [{"body_a": "payload", "body_b": "gripper_pad", "normal_force_n": 1.0, "tangential_speed_m_s": 0.0}],
        "checks": {"payload_in_gripper_contact": True, "slip_detected": False, "teleport_detected": False, "penetration_detected": False, "forbidden_collision_detected": False},
    }


def raw_summary() -> dict:
    return {
        "status": "accepted",
        "duration_s": 0.1,
        "collision_result": {"moveit_state_validity_safe": True, "gazebo_forbidden_collision_free": True, "all_samples_checked": True},
        "placement_result": {"placed_on_dish": True, "center_error_m": 0.001, "tilt_deg": 0.5, "release_confirmed": True, "rest_duration_s": 0.6},
        "integrity_result": {"teleport_detected": False, "penetration_detected": False, "forced_attachment_detected": False, "collision_filter_disabled": False},
    }


class DatasetReadinessTests(unittest.TestCase):
    def test_candidate_scenes_are_not_counted_as_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            requests = Path(temporary) / "requests.jsonl"
            requests.write_text(json.dumps(request()) + "\n" + json.dumps(request("scene_00001")) + "\n", encoding="utf-8")
            report = build_readiness_report([requests])
            self.assertEqual(report["summary"]["candidate_scene_count"], 2)
            self.assertEqual(report["summary"]["trajectory_ready_count"], 0)
            self.assertEqual(report["summary"]["next_gate"], "generate_and_validate_ompl_trajectories")

    def test_ready_trajectory_and_accepted_raw_episode_are_reported_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = root / "requests.jsonl"
            trajectories = root / "trajectories.jsonl"
            requests.write_text(json.dumps(request()) + "\n", encoding="utf-8")
            trajectories.write_text(json.dumps(trajectory()) + "\n", encoding="utf-8")
            writer = PhysicalReplayEpisodeWriter(root / "raw" / "physical_00000", raw_metadata())
            writer.append_frame(raw_frame(0, 0.0, 0.0))
            writer.append_frame(raw_frame(1, 0.1, 0.01))
            writer.finalize(raw_summary())
            report = build_readiness_report([requests], [trajectories], [root / "raw"])
            self.assertEqual(report["summary"]["trajectory_ready_count"], 1)
            self.assertEqual(report["physical_replays"]["eligible_for_lerobot_conversion"], ["physical_00000"])
            self.assertEqual(report["summary"]["next_gate"], "convert_only_eligible_raw_episodes_in_a_separate_lerobot_environment")

    def test_missing_teacher_evidence_is_rejected(self) -> None:
        incomplete = trajectory()
        del incomplete["trajectory"]["points"]
        errors = validate_trajectory_record(incomplete)
        self.assertTrue(any("trajectory.points" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
