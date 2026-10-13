#!/usr/bin/env python3
"""Generate a project-owned dual-finger URDF from a read-only SO-101 URDF.

This program never edits its input.  It accepts either a pre-expanded URDF or
an upstream xacro path, expands the latter through the installed ``xacro``
command, and writes a separate derived URDF.  The generated extension keeps
the upstream moving jaw as one finger and adds only the independent opposite
finger, two ordinary collision pads, Gazebo contact parameters, and a matching
ros2_control joint interface.

It deliberately does *not* create payload joints, mimic joints, attachment
plugins, tracking logic, teleportation logic, or collision-disable tags.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA_VERSION = "three_obstacles_dual_finger_overlay/v1"
RIGHT_JAW_LINK = "moving_jaw_so101_v1_link"
RIGHT_JOINT = "gripper"
LEFT_LINK = "three_obstacles_left_finger_link"
LEFT_JOINT = "gripper_left"
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
INITIALIZABLE_JOINTS = (*ARM_JOINTS, RIGHT_JOINT, LEFT_JOINT)
INITIAL_JOINT_STATE_SCHEMA_VERSION = "three_obstacles_initial_joint_state/v1"
PAD_SIZE_M = (0.005, 0.020, 0.030)
FINGER_BODY_SIZE_M = (0.025, 0.035, 0.045)
FRICTION_COEFFICIENT = 0.5


class OverlayError(ValueError):
    """Raised when the source robot cannot safely receive this overlay."""


@dataclass(frozen=True)
class OverlayGeometry:
    """Candidate geometry, deliberately configurable before physical calibration."""

    left_joint_xyz: tuple[float, float, float] = (0.0202, -0.0188, -0.0234)
    left_joint_rpy: tuple[float, float, float] = (-1.5708, -0.0000000524284, 0.0)
    right_pad_xyz: tuple[float, float, float] = (0.0, -0.0200, 0.0189)
    left_pad_xyz: tuple[float, float, float] = (0.0, 0.0200, 0.0189)
    left_body_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0189)


def _float_string(values: Iterable[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _find_named_direct(root: ET.Element, tag: str, name: str) -> ET.Element:
    for element in root.findall(tag):
        if element.get("name") == name:
            return element
    raise OverlayError(f"required {tag} named {name!r} is absent from source URDF")


def _find_robot_system(root: ET.Element) -> ET.Element:
    controls = root.findall("ros2_control")
    if not controls:
        raise OverlayError("source URDF has no ros2_control element")
    for control in controls:
        if control.get("name") == "RobotSystem":
            return control
    if len(controls) == 1:
        return controls[0]
    raise OverlayError("source URDF has multiple ros2_control elements but no RobotSystem")


def _find_top_level_joint(root: ET.Element, name: str) -> ET.Element:
    return _find_named_direct(root, "joint", name)


def _position_state_interface(control_joint: ET.Element, joint_name: str) -> ET.Element:
    matches = [
        interface
        for interface in control_joint.findall("state_interface")
        if interface.get("name") == "position"
    ]
    if len(matches) != 1:
        raise OverlayError(
            f"ros2_control joint {joint_name!r} must have exactly one position state interface"
        )
    return matches[0]


def _joint_limit_range(root: ET.Element, joint_name: str) -> tuple[float, float]:
    joint = _find_top_level_joint(root, joint_name)
    limit = joint.find("limit")
    if limit is None or limit.get("lower") is None or limit.get("upper") is None:
        raise OverlayError(f"joint {joint_name!r} must have finite lower and upper limits")
    try:
        lower = float(limit.get("lower"))
        upper = float(limit.get("upper"))
    except (TypeError, ValueError) as error:
        raise OverlayError(f"joint {joint_name!r} has a non-numeric limit") from error
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise OverlayError(f"joint {joint_name!r} has an invalid limit range")
    return lower, upper


def _assert_preconditions(root: ET.Element) -> None:
    _find_named_direct(root, "link", "gripper_link")
    _find_named_direct(root, "link", RIGHT_JAW_LINK)
    right_joint = _find_named_direct(root, "joint", RIGHT_JOINT)
    if right_joint.get("type") != "revolute":
        raise OverlayError(f"{RIGHT_JOINT!r} must be a revolute joint")
    child = right_joint.find("child")
    if child is None or child.get("link") != RIGHT_JAW_LINK:
        raise OverlayError(f"{RIGHT_JOINT!r} must actuate {RIGHT_JAW_LINK!r}")
    _find_robot_system(root)
    existing = {element.get("name") for element in root.findall("link") + root.findall("joint")}
    forbidden_existing = {LEFT_LINK, LEFT_JOINT}
    if existing.intersection(forbidden_existing):
        raise OverlayError("dual-finger overlay is already present; refusing to duplicate it")


def _normalize_initial_joint_positions(positions: Mapping[str, object]) -> dict[str, float]:
    """Validate the narrow robot-only startup-state mapping."""
    expected_names = set(INITIALIZABLE_JOINTS)
    actual_names = set(positions)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise OverlayError("initial joint state must list exactly the robot joints (" + "; ".join(details) + ")")

    normalized: dict[str, float] = {}
    for joint_name in INITIALIZABLE_JOINTS:
        value = positions[joint_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OverlayError(f"initial value for {joint_name!r} must be a finite number in radians")
        normalized_value = float(value)
        if not math.isfinite(normalized_value):
            raise OverlayError(f"initial value for {joint_name!r} must be finite")
        normalized[joint_name] = normalized_value
    return normalized


def load_initial_joint_state(path: Path) -> dict[str, float]:
    """Load a complete, project-owned robot startup joint-state JSON file.

    The deliberately narrow schema makes it impossible for this generator to
    receive a payload, obstacle, link pose, controller goal, or arbitrary
    joint name.  Values are later checked against the transformed URDF's
    physical joint limits before they are written.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OverlayError(f"cannot read initial joint state file {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise OverlayError(f"initial joint state file {path} is not valid JSON: {error}") from error

    if not isinstance(document, dict):
        raise OverlayError("initial joint state document must be a JSON object")
    if document.get("schema_version") != INITIAL_JOINT_STATE_SCHEMA_VERSION:
        raise OverlayError(
            "initial joint state has an unsupported schema_version; expected "
            f"{INITIAL_JOINT_STATE_SCHEMA_VERSION!r}"
        )
    positions = document.get("initial_joint_positions_rad")
    if not isinstance(positions, dict):
        raise OverlayError("initial joint state must contain an initial_joint_positions_rad object")

    return _normalize_initial_joint_positions(positions)


def _inject_initial_joint_positions(root: ET.Element, positions: Mapping[str, object]) -> dict[str, float]:
    """Add startup-only initial values to controlled robot position states.

    ``gz_ros2_control`` reads this value during model-plugin initialization.
    It is intentionally not a controller command and is limited to the seven
    articulated SO-101 joints listed in ``INITIALIZABLE_JOINTS``.
    """
    normalized = _normalize_initial_joint_positions(positions)

    system = _find_robot_system(root)
    control_joints = {
        joint.get("name"): joint
        for joint in system.findall("joint")
        if joint.get("name") is not None
    }
    for joint_name in INITIALIZABLE_JOINTS:
        control_joint = control_joints.get(joint_name)
        if control_joint is None:
            raise OverlayError(f"ros2_control joint {joint_name!r} is absent")
        state_interface = _position_state_interface(control_joint, joint_name)
        existing = [
            parameter
            for parameter in state_interface.findall("param")
            if parameter.get("name") == "initial_value"
        ]
        if existing:
            raise OverlayError(
                f"ros2_control joint {joint_name!r} already defines initial_value; refusing to overwrite it"
            )

        lower, upper = _joint_limit_range(root, joint_name)
        value = normalized[joint_name]
        if value < lower or value > upper:
            raise OverlayError(
                f"initial value {value:.9g} for {joint_name!r} is outside its URDF limits "
                f"[{lower:.9g}, {upper:.9g}]"
            )
        ET.SubElement(state_interface, "param", {"name": "initial_value"}).text = f"{value:.9g}"
    return normalized


def _make_box_geometry(parent: ET.Element, size: tuple[float, float, float]) -> None:
    geometry = ET.SubElement(parent, "geometry")
    # URDF represents a box size as an attribute (unlike SDF, which uses a
    # child <size> element).  The output is a URDF that Gazebo later converts.
    ET.SubElement(geometry, "box", {"size": _float_string(size)})


def _make_origin(parent: ET.Element, xyz: tuple[float, float, float], rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    ET.SubElement(parent, "origin", {"xyz": _float_string(xyz), "rpy": _float_string(rpy)})


def _add_collision_pad(
    link: ET.Element,
    *,
    name: str,
    xyz: tuple[float, float, float],
    color: str,
) -> None:
    """Add a normal URDF collision box and matching visual to a finger link."""
    collision = ET.SubElement(link, "collision", {"name": f"{name}_collision"})
    _make_origin(collision, xyz)
    _make_box_geometry(collision, PAD_SIZE_M)

    visual = ET.SubElement(link, "visual", {"name": f"{name}_visual"})
    _make_origin(visual, xyz)
    _make_box_geometry(visual, PAD_SIZE_M)
    material = ET.SubElement(visual, "material", {"name": f"{name}_rubber"})
    ET.SubElement(material, "color", {"rgba": color})


def _add_left_finger(root: ET.Element, geometry: OverlayGeometry) -> None:
    link = ET.SubElement(root, "link", {"name": LEFT_LINK})
    inertial = ET.SubElement(link, "inertial")
    _make_origin(inertial, geometry.left_body_xyz)
    ET.SubElement(inertial, "mass", {"value": "0.012"})
    inertia = ET.SubElement(inertial, "inertia")
    inertia.attrib.update({"ixx": "0.000006", "iyy": "0.000002", "izz": "0.000005", "ixy": "0", "ixz": "0", "iyz": "0"})

    body_collision = ET.SubElement(link, "collision", {"name": "three_obstacles_left_finger_body_collision"})
    _make_origin(body_collision, geometry.left_body_xyz)
    _make_box_geometry(body_collision, FINGER_BODY_SIZE_M)
    body_visual = ET.SubElement(link, "visual", {"name": "three_obstacles_left_finger_body_visual"})
    _make_origin(body_visual, geometry.left_body_xyz)
    _make_box_geometry(body_visual, FINGER_BODY_SIZE_M)
    body_material = ET.SubElement(body_visual, "material", {"name": "three_obstacles_left_finger"})
    ET.SubElement(body_material, "color", {"rgba": "0.95 0.75 0.15 1"})

    _add_collision_pad(
        link,
        name="three_obstacles_left_rubber_pad",
        xyz=geometry.left_pad_xyz,
        color="0.08 0.08 0.08 1",
    )

    joint = ET.SubElement(root, "joint", {"name": LEFT_JOINT, "type": "revolute"})
    _make_origin(joint, geometry.left_joint_xyz, geometry.left_joint_rpy)
    ET.SubElement(joint, "parent", {"link": "gripper_link"})
    ET.SubElement(joint, "child", {"link": LEFT_LINK})
    ET.SubElement(joint, "axis", {"xyz": "0 0 1"})
    ET.SubElement(joint, "limit", {"lower": "-0.174533", "upper": "1.74533", "effort": "10", "velocity": "10"})

    transmission = ET.SubElement(root, "transmission", {"name": "three_obstacles_gripper_left_trans"})
    ET.SubElement(transmission, "type").text = "transmission_interface/SimpleTransmission"
    transmission_joint = ET.SubElement(transmission, "joint", {"name": LEFT_JOINT})
    ET.SubElement(transmission_joint, "hardwareInterface").text = "hardware_interface/PositionJointInterface"
    actuator = ET.SubElement(transmission, "actuator", {"name": "three_obstacles_gripper_left_actuator"})
    ET.SubElement(actuator, "hardwareInterface").text = "hardware_interface/PositionJointInterface"
    ET.SubElement(actuator, "mechanicalReduction").text = "1"


def _add_joint_to_ros2_control(root: ET.Element) -> None:
    system = _find_robot_system(root)
    joint = ET.SubElement(system, "joint", {"name": LEFT_JOINT})
    command = ET.SubElement(joint, "command_interface", {"name": "position"})
    ET.SubElement(command, "param", {"name": "min"}).text = "-0.174533"
    ET.SubElement(command, "param", {"name": "max"}).text = "1.74533"
    ET.SubElement(joint, "state_interface", {"name": "position"})
    ET.SubElement(joint, "state_interface", {"name": "velocity"})
    ET.SubElement(joint, "state_interface", {"name": "effort"})


def _add_gazebo_friction(root: ET.Element, reference: str) -> None:
    """Set a link's ordinary contact surface parameters through URDF Gazebo tags."""
    gazebo = ET.SubElement(root, "gazebo", {"reference": reference})
    ET.SubElement(gazebo, "mu1").text = str(FRICTION_COEFFICIENT)
    ET.SubElement(gazebo, "mu2").text = str(FRICTION_COEFFICIENT)
    ET.SubElement(gazebo, "kp").text = "1000000"
    ET.SubElement(gazebo, "kd").text = "1"
    ET.SubElement(gazebo, "maxVel").text = "0.01"
    ET.SubElement(gazebo, "minDepth").text = "0.0001"


def transform_urdf(
    source_xml: str,
    geometry: OverlayGeometry = OverlayGeometry(),
    initial_joint_positions: dict[str, float] | None = None,
) -> tuple[ET.ElementTree, dict[str, object]]:
    """Return a derived URDF tree and self-describing project-owned model spec.

    ``initial_joint_positions`` is optional to preserve a pure geometry-only
    transform.  When supplied, it is an explicit complete startup state for
    the SO-101's articulated joints, never a trajectory or an object pose.
    """
    try:
        root = ET.fromstring(source_xml)
    except ET.ParseError as error:
        raise OverlayError(f"source URDF is not valid XML: {error}") from error
    if root.tag != "robot":
        raise OverlayError(f"source document root must be <robot>, got <{root.tag}>")
    _assert_preconditions(root)

    right_link = _find_named_direct(root, "link", RIGHT_JAW_LINK)
    _add_collision_pad(
        right_link,
        name="three_obstacles_right_rubber_pad",
        xyz=geometry.right_pad_xyz,
        color="0.08 0.08 0.08 1",
    )
    _add_left_finger(root, geometry)
    _add_joint_to_ros2_control(root)
    _add_gazebo_friction(root, RIGHT_JAW_LINK)
    _add_gazebo_friction(root, LEFT_LINK)
    if initial_joint_positions is not None:
        _inject_initial_joint_positions(root, initial_joint_positions)

    model_spec: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_contract": {
            "required_right_link": RIGHT_JAW_LINK,
            "required_right_joint": RIGHT_JOINT,
            "source_is_read_only": True,
        },
        "fingers": {
            "right": {"link": RIGHT_JAW_LINK, "joint": RIGHT_JOINT, "independently_commanded": True},
            "left": {"link": LEFT_LINK, "joint": LEFT_JOINT, "independently_commanded": True},
        },
        "pads": {
            "size_m": list(PAD_SIZE_M),
            "friction_coefficient": FRICTION_COEFFICIENT,
            "right_link": RIGHT_JAW_LINK,
            "left_link": LEFT_LINK,
        },
        "candidate_geometry": asdict(geometry),
        "calibration_status": "candidate_geometry_requires_static_rviz_review",
        "controller_contract": {
            "controller_name": "dual_finger_controller",
            "joints": [RIGHT_JOINT, LEFT_JOINT],
            "command_interface": "position",
            "state_interfaces": ["position", "velocity", "effort"],
        },
        "robot_description_contract": {
            "canonical_generated_urdf": True,
            "consumers": [
                "robot_state_publisher.robot_description",
                "ros_gz_sim_create_topic_robot_description",
                "move_group.robot_description",
                "rviz.robot_description",
            ],
            "requirement": "All consumers must receive identical generated URDF XML.",
        },
        "startup_initialization": {
            "configured": initial_joint_positions is not None,
            "interface": "ros2_control state_interface/position initial_value",
            "scope": "SO-101 articulated robot joints only",
            "phase": "simulator model startup before physics begins",
            "not_applied_to": ["payload", "obstacles", "dish", "world poses"],
            "not_a": ["controller_trajectory", "payload_reset", "object_following", "teleportation"],
            "joint_positions_rad": initial_joint_positions or {},
        },
        "forbidden_mechanisms": [
            "mimic_joint",
            "payload_fixed_joint",
            "forced_attach",
            "object_following",
            "teleportation",
            "collision_disable",
        ],
    }
    return ET.ElementTree(root), model_spec


def expand_xacro(upstream_xacro: Path, xacro_args: list[str]) -> str:
    """Expand an upstream file without changing it or its containing workspace."""
    command = ["xacro", str(upstream_xacro), *xacro_args]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as error:
        raise OverlayError("xacro executable was not found; source ROS 2 before using --upstream-xacro") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        # Some ROS installations expose the xacro module on PYTHONPATH but
        # ship a console-script shim whose Python package metadata is absent.
        # Use the same current interpreter as a narrowly scoped fallback; it
        # still only reads the xacro input and emits XML on stdout.
        metadata_failure = "No package metadata was found for xacro" in detail
        if not metadata_failure:
            raise OverlayError(f"xacro expansion failed: {detail}") from error
        fallback_command = [
            sys.executable,
            "-c",
            "import xacro; xacro.main()",
            str(upstream_xacro),
            *xacro_args,
        ]
        try:
            result = subprocess.run(fallback_command, check=True, text=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as fallback_error:
            fallback_detail = getattr(fallback_error, "stderr", "") or getattr(fallback_error, "stdout", "") or str(fallback_error)
            raise OverlayError(f"xacro expansion failed via console script and module fallback: {fallback_detail.strip()}") from fallback_error
    return result.stdout


def controller_fragment() -> str:
    """A project-owned YAML fragment; a later dedicated launch will load it."""
    return """# Generated by three_obstacles_description. Do not merge into upstream files.\ncontroller_manager:\n  ros__parameters:\n    dual_finger_controller:\n      type: joint_trajectory_controller/JointTrajectoryController\n\ndual_finger_controller:\n  ros__parameters:\n    joints:\n      - gripper\n      - gripper_left\n    command_interfaces:\n      - position\n    state_interfaces:\n      - position\n      - velocity\n      - effort\n    allow_integration_in_goal_trajectories: false\n    allow_nonzero_velocity_at_trajectory_end: false\n"""


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-urdf", type=Path, help="Read an already expanded upstream URDF without editing it")
    source.add_argument("--upstream-xacro", type=Path, help="Read and expand an upstream SO-101 xacro at runtime")
    parser.add_argument("--xacro-arg", action="append", default=[], help="One xacro argument, such as use_gazebo:=true")
    parser.add_argument("--output-urdf", required=True, type=Path)
    parser.add_argument("--output-model-spec", required=True, type=Path)
    parser.add_argument("--output-controller-fragment", required=True, type=Path)
    parser.add_argument(
        "--initial-joint-state",
        type=Path,
        help=(
            "Project-owned JSON with exactly the seven SO-101 startup joint values in radians. "
            "It is used only to initialize the robot model before physics begins."
        ),
    )
    parser.add_argument("--left-joint-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--left-joint-rpy", nargs=3, type=float, metavar=("R", "P", "Y"))
    parser.add_argument("--right-pad-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--left-pad-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    return parser


def geometry_from_args(args: argparse.Namespace) -> OverlayGeometry:
    defaults = OverlayGeometry()
    return OverlayGeometry(
        left_joint_xyz=tuple(args.left_joint_xyz) if args.left_joint_xyz else defaults.left_joint_xyz,
        left_joint_rpy=tuple(args.left_joint_rpy) if args.left_joint_rpy else defaults.left_joint_rpy,
        right_pad_xyz=tuple(args.right_pad_xyz) if args.right_pad_xyz else defaults.right_pad_xyz,
        left_pad_xyz=tuple(args.left_pad_xyz) if args.left_pad_xyz else defaults.left_pad_xyz,
        left_body_xyz=defaults.left_body_xyz,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.input_urdf:
        source_xml = args.input_urdf.read_text(encoding="utf-8")
    else:
        source_xml = expand_xacro(args.upstream_xacro, args.xacro_arg)
    initial_joint_positions = (
        load_initial_joint_state(args.initial_joint_state)
        if args.initial_joint_state is not None
        else None
    )
    tree, model_spec = transform_urdf(
        source_xml,
        geometry_from_args(args),
        initial_joint_positions=initial_joint_positions,
    )
    ET.indent(tree, space="  ")
    _write_text(args.output_urdf, ET.tostring(tree.getroot(), encoding="unicode") + "\n")
    _write_text(args.output_model_spec, json.dumps(model_spec, indent=2, sort_keys=True) + "\n")
    _write_text(args.output_controller_fragment, controller_fragment())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OverlayError as error:
        print(f"overlay generation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
