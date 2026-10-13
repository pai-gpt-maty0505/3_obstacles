from __future__ import annotations

import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from three_obstacles.gazebo.generate_episode_world import (
    DEFAULT_TEMPLATE,
    EpisodeWorldError,
    build_episode_world,
    load_episode,
    safe_output_path,
    validate_episode,
)


ROOT = Path(__file__).resolve().parents[2]
REQUESTS = ROOT / "data/teacher_requests/smoke_20261013.jsonl"


def includes_by_name(world_text: str) -> dict[str, str]:
    root = ET.fromstring(world_text)
    world = root.find("world")
    assert world is not None
    return {include.findtext("name"): include.findtext("pose") for include in world.findall("include")}


class GenerateEpisodeWorldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.record = load_episode(REQUESTS, "scene_00000")

    def test_bakes_candidate_pose_before_simulation(self) -> None:
        text = build_episode_world(DEFAULT_TEMPLATE, self.record)
        root = ET.fromstring(text)
        world = root.find("world")
        self.assertEqual(world.attrib["name"], "three_obstacles_scene_00000")
        includes = includes_by_name(text)
        payload_values = [float(value) for value in includes["payload"].split()]
        expected = self.record["entity_poses"]["payload"]
        self.assertAlmostEqual(payload_values[0], expected["x"], places=6)
        self.assertAlmostEqual(payload_values[1], expected["y"], places=6)
        self.assertAlmostEqual(payload_values[2], expected["z"], places=6)
        self.assertAlmostEqual(payload_values[5], math.radians(expected["yaw_deg"]), places=8)

    def test_does_not_emit_prohibited_transport_mechanisms(self) -> None:
        text = build_episode_world(DEFAULT_TEMPLATE, self.record).lower()
        for token in ("<joint", "attach", "follow", "teleport", "disable_collisions"):
            self.assertNotIn(token, text)

    def test_rejects_non_prechecked_scene(self) -> None:
        broken = dict(self.record)
        broken["precheck"] = {"passed": False}
        with self.assertRaises(EpisodeWorldError):
            validate_episode(broken)

    def test_output_must_remain_in_generated_worlds(self) -> None:
        allowed = safe_output_path(ROOT / "gazebo/generated_worlds/test.world")
        self.assertTrue(str(allowed).endswith("generated_worlds/test.world"))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(EpisodeWorldError):
                safe_output_path(Path(temporary) / "outside.world")
