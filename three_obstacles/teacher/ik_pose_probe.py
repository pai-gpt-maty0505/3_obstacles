#!/usr/bin/env python3
"""Make one explicit-pose GetPositionIK request without planning or execution.

Default mode is a dry run: it constructs the exact ``GetPositionIK.Request``
but creates no node and calls no service.  ``--check`` is the only mode that
contacts ROS; it waits for and calls ``/compute_ik`` once with a bounded
timeout.  This module deliberately contains no controller, action, Gazebo,
planning-scene, attachment, or collision-matrix operation.
"""

from __future__ import annotations

import argparse
import json
import math
from numbers import Real
import sys
from dataclasses import dataclass
from typing import Any


DEFAULT_SERVICE = "/compute_ik"
SUCCESS_ERROR_CODE = 1  # moveit_msgs/MoveItErrorCodes.SUCCESS


class IKPoseProbeError(ValueError):
    """The explicit request cannot be constructed or checked safely."""


@dataclass(frozen=True)
class IKPoseSpec:
    group_name: str
    link_name: str
    frame_id: str
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    timeout_s: float
    service_name: str = DEFAULT_SERVICE

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        return (self.qx, self.qy, self.qz, self.qw)


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IKPoseProbeError(f"{name} must be a non-empty string")
    return value.strip()


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise IKPoseProbeError(f"{name} must be finite")
    return converted


def make_spec(
    *,
    group_name: str,
    link_name: str,
    frame_id: str,
    x: float,
    y: float,
    z: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    timeout_s: float,
    service_name: str = DEFAULT_SERVICE,
) -> IKPoseSpec:
    """Validate an explicit pose without normalizing or changing user input."""
    fields = {
        "x": _finite(x, "x"),
        "y": _finite(y, "y"),
        "z": _finite(z, "z"),
        "qx": _finite(qx, "qx"),
        "qy": _finite(qy, "qy"),
        "qz": _finite(qz, "qz"),
        "qw": _finite(qw, "qw"),
        "timeout_s": _finite(timeout_s, "timeout_s"),
    }
    if not 0.0 < fields["timeout_s"] <= 30.0:
        raise IKPoseProbeError("timeout_s must be greater than zero and no more than 30 seconds")
    norm = math.sqrt(sum(component * component for component in (fields["qx"], fields["qy"], fields["qz"], fields["qw"])))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise IKPoseProbeError("quaternion must already have unit length; it is not normalized implicitly")
    return IKPoseSpec(
        group_name=_nonempty(group_name, "group_name"),
        link_name=_nonempty(link_name, "link_name"),
        frame_id=_nonempty(frame_id, "frame_id"),
        service_name=_nonempty(service_name, "service_name"),
        **fields,
    )


def build_ros_request(spec: IKPoseSpec, get_position_ik_type: Any, pose_stamped_type: Any) -> Any:
    """Construct one service request; this function never creates a ROS node."""
    request = get_position_ik_type.Request()
    pose_stamped = pose_stamped_type()
    pose_stamped.header.frame_id = spec.frame_id
    pose_stamped.pose.position.x = spec.x
    pose_stamped.pose.position.y = spec.y
    pose_stamped.pose.position.z = spec.z
    pose_stamped.pose.orientation.x = spec.qx
    pose_stamped.pose.orientation.y = spec.qy
    pose_stamped.pose.orientation.z = spec.qz
    pose_stamped.pose.orientation.w = spec.qw
    request.ik_request.group_name = spec.group_name
    request.ik_request.ik_link_name = spec.link_name
    request.ik_request.pose_stamped = pose_stamped
    # This is a standard request flag, not a request to alter collision data.
    request.ik_request.avoid_collisions = True
    seconds = int(spec.timeout_s)
    nanoseconds = int(round((spec.timeout_s - seconds) * 1_000_000_000))
    # Rounding a value immediately below the next integer must not create the
    # invalid Duration representation ``sec=N, nanosec=1_000_000_000``.
    if nanoseconds == 1_000_000_000:
        seconds += 1
        nanoseconds = 0
    request.ik_request.timeout.sec = seconds
    request.ik_request.timeout.nanosec = nanoseconds
    return request


def _response_solution(response: Any) -> dict[str, list[float] | list[str]]:
    error_code = getattr(getattr(response, "error_code", None), "val", None)
    if error_code != SUCCESS_ERROR_CODE:
        raise IKPoseProbeError(f"/compute_ik returned MoveIt error code {error_code!r}")
    joint_state = getattr(getattr(response, "solution", None), "joint_state", None)
    names = getattr(joint_state, "name", None)
    positions = getattr(joint_state, "position", None)
    # ROS Python generators commonly expose sequence-like containers (for
    # example ``array.array``) rather than a literal list or tuple.  Materialize
    # ordinary iterables once, while rejecting string/mapping values that would
    # otherwise look iterable but cannot represent a joint-state sequence.
    try:
        if isinstance(names, (str, bytes, bytearray, dict)) or isinstance(positions, (str, bytes, bytearray, dict)):
            raise TypeError
        name_values = list(names)
        position_values = list(positions)
    except TypeError as error:
        raise IKPoseProbeError("/compute_ik response has no joint-state solution") from error
    if not name_values or len(name_values) != len(position_values) or any(
        not isinstance(name, str) or not name for name in name_values
    ):
        raise IKPoseProbeError("/compute_ik response has an invalid joint-state solution")
    if any(
        isinstance(position, bool) or not isinstance(position, Real) or not math.isfinite(float(position))
        for position in position_values
    ):
        raise IKPoseProbeError("/compute_ik response has non-finite joint positions")
    return {"joint_names": name_values, "joint_positions_rad": [float(position) for position in position_values]}


def call_service_once(spec: IKPoseSpec, rclpy_module: Any, get_position_ik_type: Any, pose_stamped_type: Any) -> dict[str, list[float] | list[str]]:
    """Create one client, issue exactly one request, and wait only `timeout_s`."""
    request = build_ros_request(spec, get_position_ik_type, pose_stamped_type)
    initialized_here = not rclpy_module.ok()
    if initialized_here:
        rclpy_module.init(args=None)
    node = rclpy_module.create_node("three_obstacles_ik_pose_probe")
    try:
        client = node.create_client(get_position_ik_type, spec.service_name)
        if not client.wait_for_service(timeout_sec=spec.timeout_s):
            raise IKPoseProbeError(f"IK service unavailable within {spec.timeout_s:g} s: {spec.service_name}")
        future = client.call_async(request)  # The sole service invocation in this module.
        rclpy_module.spin_until_future_complete(node, future, timeout_sec=spec.timeout_s)
        if not future.done():
            raise IKPoseProbeError(f"IK service did not respond within {spec.timeout_s:g} s")
        response = future.result()
        if response is None:
            raise IKPoseProbeError("IK service returned no response")
        return _response_solution(response)
    finally:
        node.destroy_node()
        if initialized_here:
            rclpy_module.shutdown()


def load_ros_types() -> tuple[Any, Any, Any]:
    """Import ROS types lazily so pure unit tests need no ROS installation."""
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.srv import GetPositionIK
    except ImportError as error:
        raise IKPoseProbeError(
            "ROS 2 Jazzy and moveit_msgs are required to construct a ROS GetPositionIK request"
        ) from error
    return rclpy, GetPositionIK, PoseStamped


def dry_run_summary(spec: IKPoseSpec) -> dict[str, Any]:
    """Human-readable evidence that no node, plan, or service call was made."""
    return {
        "mode": "dry_run_request_constructed_only",
        "service": spec.service_name,
        "group": spec.group_name,
        "link": spec.link_name,
        "frame": spec.frame_id,
        "position_m": [spec.x, spec.y, spec.z],
        "quaternion_xyzw": list(spec.quaternion),
        "timeout_s": spec.timeout_s,
        "service_calls": 0,
        "robot_or_gazebo_motion": "not_issued",
        "trajectory_execution": "not_issued",
        "controller_or_gripper_command": "not_issued",
        "gazebo_mutation": "not_issued",
        "planning_scene_mutation": "not_issued",
        "attached_objects": "not_used",
        "collision_matrix_changes": "not_used",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, help="MoveIt planning group, e.g. arm")
    parser.add_argument("--link", required=True, help="IK link name")
    parser.add_argument("--frame", required=True, help="pose frame ID, e.g. base_link")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--qx", type=float, required=True)
    parser.add_argument("--qy", type=float, required=True)
    parser.add_argument("--qz", type=float, required=True)
    parser.add_argument("--qw", type=float, required=True)
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="GetPositionIK service name")
    parser.add_argument("--timeout-s", type=float, default=3.0, help="bounded wait/call timeout in seconds")
    parser.add_argument("--check", action="store_true", help="call /compute_ik once; omitted means request construction only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        spec = make_spec(
            group_name=args.group,
            link_name=args.link,
            frame_id=args.frame,
            x=args.x,
            y=args.y,
            z=args.z,
            qx=args.qx,
            qy=args.qy,
            qz=args.qz,
            qw=args.qw,
            timeout_s=args.timeout_s,
            service_name=args.service,
        )
        rclpy_module, service_type, pose_type = load_ros_types()
        # Request construction is deliberately performed in both modes.
        build_ros_request(spec, service_type, pose_type)
        if not args.check:
            print(json.dumps(dry_run_summary(spec), ensure_ascii=False, sort_keys=True))
            return 0
        # Successful check output contains only the returned joint solution.
        print(json.dumps(call_service_once(spec, rclpy_module, service_type, pose_type), ensure_ascii=False, sort_keys=True))
        return 0
    except IKPoseProbeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
