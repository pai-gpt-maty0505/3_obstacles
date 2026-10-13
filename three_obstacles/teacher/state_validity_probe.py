#!/usr/bin/env python3
"""Read-only MoveIt state-validity probe for one explicit five-joint arm state.

The default constructs the request only.  Only explicit ``--check`` calls
``/check_state_validity``.  There is no controller, action, Gazebo,
PlanningScene, attachment, allowed-collision-matrix, or trajectory code here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Sequence


DEFAULT_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
EXPECTED_JOINT_COUNT = 5
GROUP_NAME = "arm"
SERVICE_NAME = "/check_state_validity"


class StateValidityProbeError(RuntimeError):
    """Invalid probe input or a bounded read-only service failure."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise StateValidityProbeError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise StateValidityProbeError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise StateValidityProbeError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class StateValidityProbeSpec:
    joint_names: tuple[str, ...]
    positions_rad: tuple[float, ...]
    service_timeout_s: float
    group_name: str = GROUP_NAME


def build_probe_spec(
    positions_rad: Sequence[Any],
    joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
    *,
    service_timeout_s: Any = 5.0,
) -> StateValidityProbeSpec:
    """Validate a five-joint query before importing or initializing ROS."""
    positions = tuple(_finite(value, f"joint[{index}]") for index, value in enumerate(positions_rad))
    names = tuple(joint_names)
    if len(positions) != EXPECTED_JOINT_COUNT:
        raise StateValidityProbeError(f"exactly {EXPECTED_JOINT_COUNT} joint positions are required")
    if len(names) != EXPECTED_JOINT_COUNT or any(not isinstance(name, str) or not name for name in names):
        raise StateValidityProbeError(f"exactly {EXPECTED_JOINT_COUNT} nonempty joint names are required")
    if len(set(names)) != len(names):
        raise StateValidityProbeError("joint names must be unique")
    timeout = _finite(service_timeout_s, "service timeout")
    if timeout <= 0.0:
        raise StateValidityProbeError("service timeout must be positive")
    return StateValidityProbeSpec(joint_names=names, positions_rad=positions, service_timeout_s=timeout)


def portable_probe_summary(spec: StateValidityProbeSpec) -> dict[str, Any]:
    """Describe the strictly read-only request without ROS messages."""
    return {
        "service": SERVICE_NAME,
        "group_name": spec.group_name,
        "joint_positions_rad": dict(zip(spec.joint_names, spec.positions_rad)),
        "robot_state_policy": "explicit_arm_joint_positions_with_current_move_group_values_for_unspecified_joints",
        "scene_mutation": "none",
        "attached_objects": "none",
        "collision_matrix_changes": "none",
        "robot_or_gazebo_motion": "none",
    }


def build_ros_state_validity_request(spec: StateValidityProbeSpec) -> Any:
    """Build ``GetStateValidity.Request`` in memory without making a call."""
    try:
        from moveit_msgs.srv import GetStateValidity
    except ImportError as error:
        raise StateValidityProbeError("moveit_msgs is required to construct the state-validity request") from error
    request = GetStateValidity.Request()
    request.group_name = spec.group_name
    request.robot_state.joint_state.name = list(spec.joint_names)
    request.robot_state.joint_state.position = list(spec.positions_rad)
    # Missing joints are resolved by MoveIt's current state. This requests no
    # controller update and does not change the PlanningScene.
    request.robot_state.is_diff = True
    return request


def _error_code_value(response: Any) -> int | None:
    error_code = getattr(response, "error_code", None)
    value = getattr(error_code, "val", None)
    return int(value) if value is not None else None


def validity_report(response: Any) -> dict[str, Any]:
    """Extract validity, MoveIt error code, and reported contact body names."""
    contacts = []
    for contact in getattr(response, "contacts", []):
        first = getattr(contact, "contact_body_1", "")
        second = getattr(contact, "contact_body_2", "")
        if first or second:
            contacts.append({"body_1": first, "body_2": second})
    return {
        "valid": bool(getattr(response, "valid", False)),
        "moveit_error_code": _error_code_value(response),
        "contact_count": len(contacts),
        "contacts": contacts,
    }


def call_state_validity_service(ros_request: Any, timeout_s: float) -> Any:
    """Perform one bounded, read-only service request; never change state."""
    if timeout_s <= 0.0:
        raise StateValidityProbeError("service timeout must be positive")
    try:
        import rclpy
        from moveit_msgs.srv import GetStateValidity
    except ImportError as error:
        raise StateValidityProbeError("rclpy and moveit_msgs are required only for --check") from error

    initialized_here = False
    if not rclpy.ok():
        rclpy.init(args=None)
        initialized_here = True
    node = None
    try:
        node = rclpy.create_node("three_obstacles_state_validity_probe")
        client = node.create_client(GetStateValidity, SERVICE_NAME)
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise StateValidityProbeError(f"{SERVICE_NAME} did not become available before timeout")
        future = client.call_async(ros_request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
        if not future.done():
            raise StateValidityProbeError(f"{SERVICE_NAME} call timed out")
        response = future.result()
        if response is None:
            raise StateValidityProbeError(f"{SERVICE_NAME} returned no response")
        return response
    finally:
        if node is not None:
            node.destroy_node()
        if initialized_here:
            rclpy.shutdown()


def _base_report(spec: StateValidityProbeSpec, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "request": portable_probe_summary(spec),
        "trajectory_execution": "not_issued",
        "controller_or_gripper_command": "not_issued",
        "gazebo_mutation": "not_issued",
        "planning_scene_mutation": "not_issued",
        "attached_objects": "not_used",
        "collision_matrix_changes": "not_used",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-rad", nargs=EXPECTED_JOINT_COUNT, required=True, type=float, metavar=("J1", "J2", "J3", "J4", "J5"), help="five explicit arm angles in radians")
    parser.add_argument("--joint-names", nargs=EXPECTED_JOINT_COUNT, default=DEFAULT_JOINT_NAMES, metavar=("J1", "J2", "J3", "J4", "J5"), help="five names matching group arm")
    parser.add_argument("--service-timeout-s", type=float, default=5.0)
    parser.add_argument("--check", action="store_true", help="explicitly call /check_state_validity once")
    parser.add_argument("--dry-run", action="store_true", help="construct request only; this is the default")
    args = parser.parse_args()
    if args.check and args.dry_run:
        parser.error("--check and --dry-run cannot be combined")
    return args


def main() -> int:
    args = parse_args()
    try:
        spec = build_probe_spec(args.joint_rad, args.joint_names, service_timeout_s=args.service_timeout_s)
        ros_request = build_ros_state_validity_request(spec)
        if not args.check:
            print(json.dumps(_base_report(spec, "dry_run_no_state_validity_service_call"), ensure_ascii=False, sort_keys=True))
            return 0
        report = _base_report(spec, "read_only_state_validity_service_call")
        report["result"] = validity_report(call_state_validity_service(ros_request, spec.service_timeout_s))
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0 if report["result"]["valid"] else 3
    except (StateValidityProbeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
