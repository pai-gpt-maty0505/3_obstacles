"""No-simulator tests for the project-owned dual-finger URDF transform."""

from __future__ import annotations

import hashlib
import json
import sys
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from three_obstacles_description.overlay_generator import (  # noqa: E402
    LEFT_JOINT,
    LEFT_LINK,
    PAD_SIZE_M,
    RIGHT_JAW_LINK,
    OverlayError,
    expand_xacro,
    main,
    transform_urdf,
)


def minimal_upstream_urdf() -> str:
    """A handwritten small URDF fixture, not copied from an upstream file."""
    return """<?xml version=\"1.0\"?>
<robot name=\"fixture_so101\">
  <link name=\"gripper_link\"/>
  <link name=\"moving_jaw_so101_v1_link\"><collision name=\"jaw_body\"><geometry><box><size>0.02 0.02 0.02</size></box></geometry></collision></link>
  <joint name=\"gripper\" type=\"revolute\"><parent link=\"gripper_link\"/><child link=\"moving_jaw_so101_v1_link\"/><axis xyz=\"0 0 1\"/><limit lower=\"-0.17\" upper=\"1.74\" effort=\"10\" velocity=\"10\"/></joint>
  <ros2_control name=\"RobotSystem\" type=\"system\"><hardware><plugin>example/System</plugin></hardware><joint name=\"gripper\"><command_interface name=\"position\"/><state_interface name=\"position\"/></joint></ros2_control>
</robot>
"""


class OverlayGeneratorTests(unittest.TestCase):
    def test_transform_adds_two_independently_commanded_fingers_and_pads(self) -> None:
        tree, spec = transform_urdf(minimal_upstream_urdf())
        root = tree.getroot()
        left_joint = next(joint for joint in root.findall("joint") if joint.get("name") == LEFT_JOINT)
        self.assertEqual(left_joint.get("type"), "revolute")
        self.assertEqual(left_joint.find("parent").get("link"), "gripper_link")
        self.assertEqual(left_joint.find("child").get("link"), LEFT_LINK)
        self.assertNotIn("mimic", [element.tag for element in left_joint.iter()])

        right_link = next(link for link in root.findall("link") if link.get("name") == RIGHT_JAW_LINK)
        left_link = next(link for link in root.findall("link") if link.get("name") == LEFT_LINK)
        self.assertIsNotNone(right_link.find("collision[@name='three_obstacles_right_rubber_pad_collision']"))
        self.assertIsNotNone(left_link.find("collision[@name='three_obstacles_left_rubber_pad_collision']"))
        for link in (right_link, left_link):
            pad = next(collision for collision in link.findall("collision") if "rubber_pad" in collision.get("name", ""))
            self.assertEqual(pad.find("geometry/box").get("size"), " ".join(str(value) for value in PAD_SIZE_M))

        self.assertEqual(spec["fingers"]["right"]["independently_commanded"], True)
        self.assertEqual(spec["fingers"]["left"]["independently_commanded"], True)
        self.assertEqual(spec["pads"]["friction_coefficient"], 0.5)

    def test_ros2_control_and_gazebo_contact_extension_are_added(self) -> None:
        tree, _ = transform_urdf(minimal_upstream_urdf())
        root = tree.getroot()
        control = root.find("ros2_control")
        self.assertIsNotNone(control)
        left_control_joint = control.find("joint[@name='gripper_left']")
        self.assertIsNotNone(left_control_joint)
        self.assertEqual(left_control_joint.find("command_interface").get("name"), "position")
        self.assertEqual([state.get("name") for state in left_control_joint.findall("state_interface")], ["position", "velocity", "effort"])

        contacts = {gazebo.get("reference"): gazebo for gazebo in root.findall("gazebo")}
        self.assertEqual(contacts[RIGHT_JAW_LINK].findtext("mu1"), "0.5")
        self.assertEqual(contacts[RIGHT_JAW_LINK].findtext("mu2"), "0.5")
        self.assertEqual(contacts[LEFT_LINK].findtext("mu1"), "0.5")
        self.assertEqual(contacts[LEFT_LINK].findtext("mu2"), "0.5")

    def test_source_string_is_unchanged_and_duplicate_application_is_rejected(self) -> None:
        source = minimal_upstream_urdf()
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        tree, _ = transform_urdf(source)
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), source_hash)
        with self.assertRaises(OverlayError):
            transform_urdf(ET.tostring(tree.getroot(), encoding="unicode"))

    def test_cli_writes_derived_files_without_touching_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "upstream.urdf"
            source.write_text(minimal_upstream_urdf(), encoding="utf-8")
            original = source.read_bytes()
            output_urdf = directory / "output" / "robot.urdf"
            output_spec = directory / "output" / "model.json"
            output_controller = directory / "output" / "controller.yaml"
            result = main([
                "--input-urdf", str(source),
                "--output-urdf", str(output_urdf),
                "--output-model-spec", str(output_spec),
                "--output-controller-fragment", str(output_controller),
            ])
            self.assertEqual(result, 0)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(ET.parse(output_urdf).getroot().tag, "robot")
            self.assertEqual(json.loads(output_spec.read_text(encoding="utf-8"))["schema_version"], "three_obstacles_dual_finger_overlay/v1")
            controller = output_controller.read_text(encoding="utf-8")
            self.assertIn("- gripper", controller)
            self.assertIn("- gripper_left", controller)

    def test_xacro_runtime_mode_invokes_read_only_expander(self) -> None:
        completed = Mock(stdout=minimal_upstream_urdf(), stderr="")
        with patch("three_obstacles_description.overlay_generator.subprocess.run", return_value=completed) as run:
            result = expand_xacro(Path("/safe/read_only/upstream.xacro"), ["use_gazebo:=true"])
        self.assertEqual(result, minimal_upstream_urdf())
        run.assert_called_once_with(
            ["xacro", "/safe/read_only/upstream.xacro", "use_gazebo:=true"],
            check=True,
            text=True,
            capture_output=True,
        )

    def test_xacro_runtime_mode_falls_back_only_for_missing_console_metadata(self) -> None:
        console_failure = subprocess.CalledProcessError(
            1,
            ["xacro"],
            stderr="PackageNotFoundError: No package metadata was found for xacro",
        )
        module_success = Mock(stdout=minimal_upstream_urdf(), stderr="")
        with patch(
            "three_obstacles_description.overlay_generator.subprocess.run",
            side_effect=[console_failure, module_success],
        ) as run:
            result = expand_xacro(Path("/safe/read_only/upstream.xacro"), ["use_gazebo:=true"])
        self.assertEqual(result, minimal_upstream_urdf())
        self.assertEqual(run.call_count, 2)
        fallback = run.call_args_list[1].args[0]
        self.assertEqual(fallback[0], sys.executable)
        self.assertEqual(fallback[1:3], ["-c", "import xacro; xacro.main()"])
        self.assertIn("/safe/read_only/upstream.xacro", fallback)


if __name__ == "__main__":
    unittest.main()
