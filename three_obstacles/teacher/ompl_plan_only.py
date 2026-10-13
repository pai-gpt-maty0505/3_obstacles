#!/usr/bin/env python3
"""Strictly plan-only offline OMPL smoke client for one arm joint goal.

This module can construct a MoveIt ``GetMotionPlan`` request for the
``/plan_kinematic_path`` service.  It has no trajectory execution, action
client, controller, Gazebo, attachment, PlanningScene mutation, or collision
matrix code.  The service is called only with explicit ``--plan``; the default
is a request-construction dry run.
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
GROUP_NAME = "arm"
PIPELINE_ID = "ompl"
PLANNER_ID = "RRTConnectkConfigDefault"
EXPECTED_JOINT_COUNT = 5


class PlanOnlyError(RuntimeError):
    """The plan-only request or bounded service call could not be completed."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PlanOnlyError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise PlanOnlyError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise PlanOnlyError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class JointSpacePlanSpec:
    """ROS-independent, five-joint request contract."""

    joint_names: tuple[str, ...]
    goal_positions_rad: tuple[float, ...]
    start_positions_rad: tuple[float, ...] | None
    joint_tolerance_rad: float
    allowed_planning_time_s: float
    service_timeout_s: float
    num_planning_attempts: int = 1
    group_name: str = GROUP_NAME
    pipeline_id: str = PIPELINE_ID
    planner_id: str = PLANNER_ID


def build_plan_spec(
    goal_positions_rad: Sequence[Any],
    joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
    *,
    start_positions_rad: Sequence[Any] | None = None,
    joint_tolerance_rad: Any = 0.01,
    allowed_planning_time_s: Any = 2.0,
    service_timeout_s: Any = 5.0,
) -> JointSpacePlanSpec:
    """Validate the explicit five-joint goal before any ROS import happens."""
    positions = tuple(_finite(value, f"goal[{index}]") for index, value in enumerate(goal_positions_rad))
    start_positions = None if start_positions_rad is None else tuple(
        _finite(value, f"start[{index}]") for index, value in enumerate(start_positions_rad)
    )
    names = tuple(joint_names)
    if len(positions) != EXPECTED_JOINT_COUNT:
        raise PlanOnlyError(f"exactly {EXPECTED_JOINT_COUNT} goal joint angles are required")
    if start_positions is not None and len(start_positions) != EXPECTED_JOINT_COUNT:
        raise PlanOnlyError(f"exactly {EXPECTED_JOINT_COUNT} start joint angles are required")
    if len(names) != EXPECTED_JOINT_COUNT or any(not isinstance(name, str) or not name for name in names):
        raise PlanOnlyError(f"exactly {EXPECTED_JOINT_COUNT} nonempty joint names are required")
    if len(set(names)) != len(names):
        raise PlanOnlyError("joint names must be unique")
    tolerance = _finite(joint_tolerance_rad, "joint tolerance")
    planning_time = _finite(allowed_planning_time_s, "planning time")
    service_timeout = _finite(service_timeout_s, "service timeout")
    if tolerance <= 0.0 or planning_time <= 0.0 or service_timeout <= 0.0:
        raise PlanOnlyError("tolerance, planning time, and service timeout must be positive")
    return JointSpacePlanSpec(
        joint_names=names,
        goal_positions_rad=positions,
        start_positions_rad=start_positions,
        joint_tolerance_rad=tolerance,
        allowed_planning_time_s=planning_time,
        service_timeout_s=service_timeout,
    )


def portable_request_summary(spec: JointSpacePlanSpec) -> dict[str, Any]:
    """Return the full plan request contract without ROS message objects."""
    return {
        "service": "/plan_kinematic_path",
        "group_name": spec.group_name,
        "pipeline_id": spec.pipeline_id,
        "planner_id": spec.planner_id,
        "num_planning_attempts": spec.num_planning_attempts,
        "allowed_planning_time_s": spec.allowed_planning_time_s,
        "joint_tolerance_rad": spec.joint_tolerance_rad,
        "goal_joint_positions_rad": dict(zip(spec.joint_names, spec.goal_positions_rad)),
        "start_joint_positions_rad": (
            dict(zip(spec.joint_names, spec.start_positions_rad)) if spec.start_positions_rad is not None else None
        ),
        "start_state_policy": (
            "explicit_arm_joint_positions_only_no_controller_command"
            if spec.start_positions_rad is not None
            else "current_move_group_state_only_no_controller_command"
        ),
        "path_constraints": "none",
        "attachments": "none",
        "collision_matrix_changes": "none",
    }


def populate_motion_plan_request(
    spec: JointSpacePlanSpec,
    request_factory: Any,
    constraints_factory: Any,
    joint_constraint_factory: Any,
) -> Any:
    """Populate a request using injectable message factories for pure tests."""
    request = request_factory()
    motion = request.motion_plan_request
    motion.group_name = spec.group_name
    motion.pipeline_id = spec.pipeline_id
    motion.planner_id = spec.planner_id
    motion.num_planning_attempts = spec.num_planning_attempts
    motion.allowed_planning_time = spec.allowed_planning_time_s
    motion.start_state.is_diff = True
    if spec.start_positions_rad is not None:
        motion.start_state.joint_state.name = list(spec.joint_names)
        motion.start_state.joint_state.position = list(spec.start_positions_rad)
    goal = constraints_factory()
    for name, position in zip(spec.joint_names, spec.goal_positions_rad):
        constraint = joint_constraint_factory()
        constraint.joint_name = name
        constraint.position = position
        constraint.tolerance_above = spec.joint_tolerance_rad
        constraint.tolerance_below = spec.joint_tolerance_rad
        constraint.weight = 1.0
        goal.joint_constraints.append(constraint)
    motion.goal_constraints.append(goal)
    return request


def build_ros_motion_plan_request(spec: JointSpacePlanSpec) -> Any:
    """Construct a ``GetMotionPlan.Request`` in memory; do not call a service."""
    try:
        from moveit_msgs.msg import Constraints, JointConstraint
        from moveit_msgs.srv import GetMotionPlan
    except ImportError as error:
        raise PlanOnlyError("moveit_msgs is required to construct the ROS planning request") from error
    return populate_motion_plan_request(spec, GetMotionPlan.Request, Constraints, JointConstraint)


def _duration_to_seconds(value: Any) -> float:
    return float(getattr(value, "sec", 0)) + float(getattr(value, "nanosec", 0)) / 1_000_000_000.0


def trajectory_report(response: Any) -> dict[str, Any]:
    """Extract only plan metadata and joint positions from a service response."""
    motion_response = response.motion_plan_response
    joint_trajectory = motion_response.trajectory.joint_trajectory
    points = list(joint_trajectory.points)
    positions = [[float(value) for value in point.positions] for point in points]
    return {
        "moveit_error_code": int(motion_response.error_code.val),
        "planning_time_s": _duration_to_seconds(motion_response.planning_time),
        "trajectory": {
            "joint_names": list(joint_trajectory.joint_names),
            "point_count": len(points),
            "duration_s": _duration_to_seconds(points[-1].time_from_start) if points else 0.0,
            "positions_rad": positions,
        },
    }


def call_plan_service(ros_request: Any, timeout_s: float) -> Any:
    """Make one bounded service call and return its response without executing it."""
    if timeout_s <= 0.0:
        raise PlanOnlyError("service timeout must be positive")
    try:
        import rclpy
        from moveit_msgs.srv import GetMotionPlan
    except ImportError as error:
        raise PlanOnlyError("rclpy and moveit_msgs are required only for --plan") from error

    initialized_here = False
    if not rclpy.ok():
        rclpy.init(args=None)
        initialized_here = True
    node = None
    try:
        node = rclpy.create_node("three_obstacles_offline_ompl_plan_only")
        client = node.create_client(GetMotionPlan, "/plan_kinematic_path")
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise PlanOnlyError("/plan_kinematic_path did not become available before timeout")
        future = client.call_async(ros_request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
        if not future.done():
            raise PlanOnlyError("/plan_kinematic_path call timed out")
        response = future.result()
        if response is None:
            raise PlanOnlyError("/plan_kinematic_path returned no response")
        return response
    finally:
        if node is not None:
            node.destroy_node()
        if initialized_here:
            rclpy.shutdown()


def _base_report(spec: JointSpacePlanSpec, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "request": portable_request_summary(spec),
        "robot_or_gazebo_motion": "not_issued",
        "trajectory_execution": "not_issued",
        "attached_objects": "not_used",
        "collision_matrix_changes": "not_used",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-rad", nargs=EXPECTED_JOINT_COUNT, required=True, type=float, metavar=("J1", "J2", "J3", "J4", "J5"), help="five explicit target joint angles in radians")
    parser.add_argument("--start-rad", nargs=EXPECTED_JOINT_COUNT, type=float, metavar=("J1", "J2", "J3", "J4", "J5"), help="optional explicit five-joint planning start state in radians")
    parser.add_argument("--joint-names", nargs=EXPECTED_JOINT_COUNT, default=DEFAULT_JOINT_NAMES, metavar=("J1", "J2", "J3", "J4", "J5"), help="five names matching the active arm group")
    parser.add_argument("--joint-tolerance-rad", type=float, default=0.01)
    parser.add_argument("--planning-time-s", type=float, default=2.0)
    parser.add_argument("--service-timeout-s", type=float, default=5.0)
    parser.add_argument("--plan", action="store_true", help="explicitly call /plan_kinematic_path once")
    parser.add_argument("--dry-run", action="store_true", help="construct the request only; this is the default")
    args = parser.parse_args()
    if args.plan and args.dry_run:
        parser.error("--plan and --dry-run cannot be combined")
    return args


def main() -> int:
    args = parse_args()
    try:
        spec = build_plan_spec(
            args.goal_rad,
            args.joint_names,
            start_positions_rad=args.start_rad,
            joint_tolerance_rad=args.joint_tolerance_rad,
            allowed_planning_time_s=args.planning_time_s,
            service_timeout_s=args.service_timeout_s,
        )
        ros_request = build_ros_motion_plan_request(spec)
        if not args.plan:
            print(json.dumps(_base_report(spec, "dry_run_no_plan_service_call"), ensure_ascii=False, sort_keys=True))
            return 0
        response = call_plan_service(ros_request, spec.service_timeout_s)
        report = _base_report(spec, "planned_offline_service_call_only")
        report["result"] = trajectory_report(response)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0 if report["result"]["moveit_error_code"] == 1 else 3
    except (PlanOnlyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
