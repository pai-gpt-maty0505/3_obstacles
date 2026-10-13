#!/usr/bin/env python3
"""Safely prepare or explicitly apply one teacher scene to MoveIt.

Default and ``--dry-run`` modes only build an in-memory
``moveit_msgs/PlanningScene``.  Only an explicit ``--apply`` opens the
``/apply_planning_scene`` service client.  This command has no trajectory,
controller, Gazebo, attachment, or collision-matrix code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .ros_adapter import RosMessagesUnavailable, to_moveit_planning_scene
from .scene_builder import (
    TeacherSceneError,
    build_finalized_teacher_scene,
    load_mission_spec,
    load_teacher_requests,
    project_root,
    select_episode,
)


SAFE_OBJECT_IDS = {
    "teacher/cylinder",
    "teacher/rectangular_prism",
    "teacher/triangular_pyramid",
    "teacher/payload",
    "teacher/dish",
}


class TeacherSceneApplyError(RuntimeError):
    """The document or optional service call failed a safety check."""


def validate_apply_document(document: dict[str, Any]) -> None:
    """Reject documents that would mutate anything beyond ADD world geometry.

    The adapter itself accepts a generic portable PlanningScene document.  This
    CLI is deliberately stricter: it permits only the five project-owned world
    collision objects and forbids attached objects or any collision-matrix
    change before a ROS service client is even constructed.
    """
    policy = document.get("teacher_policy", {})
    if policy.get("offline_teacher_only") is not True:
        raise TeacherSceneApplyError("scene is not marked offline-teacher-only")
    if policy.get("online_replanning_allowed") is not False:
        raise TeacherSceneApplyError("scene would permit online replanning")
    scene = document.get("planning_scene")
    if not isinstance(scene, dict):
        raise TeacherSceneApplyError("planning_scene is missing")
    forbidden_scene_keys = {"allowed_collision_matrix", "fixed_frame_transforms", "link_padding", "link_scale"}
    present = forbidden_scene_keys.intersection(scene)
    if present:
        raise TeacherSceneApplyError(f"forbidden planning-scene mutation: {sorted(present)}")
    robot_state = scene.get("robot_state")
    if not isinstance(robot_state, dict) or set(robot_state) != {"is_diff", "note"}:
        raise TeacherSceneApplyError("robot_state must be the no-command preparation placeholder")
    if "attached_collision_objects" in robot_state:
        raise TeacherSceneApplyError("attached collision objects are forbidden")
    world = scene.get("world")
    if not isinstance(world, dict) or set(world) != {"collision_objects"}:
        raise TeacherSceneApplyError("scene world must only contain collision_objects")
    objects = world.get("collision_objects")
    if not isinstance(objects, list) or {item.get("id") for item in objects if isinstance(item, dict)} != SAFE_OBJECT_IDS:
        raise TeacherSceneApplyError("scene must contain exactly the five project collision objects")
    for item in objects:
        if not isinstance(item, dict) or item.get("operation") != "ADD":
            raise TeacherSceneApplyError("only ADD collision-object operations are permitted")
        if item.get("role") == "dynamic_payload_start_pose" and item.get("id") != "teacher/payload":
            raise TeacherSceneApplyError("unexpected payload representation")


def prepare_scene(
    request: dict[str, Any], mission_spec: dict[str, Any], adapter: Callable[[dict[str, Any]], Any] = to_moveit_planning_scene
) -> tuple[dict[str, Any], Any]:
    """Build and adapt one scene in memory, without contacting ROS services."""
    document = build_finalized_teacher_scene(request, mission_spec)
    validate_apply_document(document)
    return document, adapter(document)


def apply_via_service(planning_scene: Any, timeout_s: float) -> None:
    """Call only MoveIt's apply-planning-scene service, with bounded waiting.

    There are intentionally no action clients, publishers, Gazebo calls, or
    attachment/collision-matrix changes in this function.
    """
    if timeout_s <= 0.0:
        raise TeacherSceneApplyError("service timeout must be positive")
    try:
        import rclpy
        from moveit_msgs.srv import ApplyPlanningScene
    except ImportError as error:
        raise TeacherSceneApplyError("rclpy and moveit_msgs/srv are required only for --apply") from error

    rclpy.init(args=None)
    node = None
    try:
        node = rclpy.create_node("three_obstacles_teacher_scene_apply")
        client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise TeacherSceneApplyError("/apply_planning_scene did not become available before timeout")
        request = ApplyPlanningScene.Request()
        request.scene = planning_scene
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
        if not future.done():
            raise TeacherSceneApplyError("/apply_planning_scene call timed out")
        response = future.result()
        if response is None or response.success is not True:
            raise TeacherSceneApplyError("/apply_planning_scene rejected the collision scene")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def summary(document: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "episode_id": document["source_request"]["episode_id"],
        "scene_hash": document["source_request"]["scene_hash"],
        "document_sha256": document["document_sha256"],
        "collision_object_ids": [item["id"] for item in document["planning_scene"]["world"]["collision_objects"]],
        "route_gate_ids": [gate["id"] for gate in document["route_plan"]["gates"]],
        "robot_or_gazebo_motion": "not_issued",
        "attached_objects": "not_used",
        "collision_matrix_changes": "not_used",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True, help="teacher request JSONL")
    parser.add_argument("--episode-id", required=True, help="one candidate scene ID")
    parser.add_argument("--mission-spec", type=Path, default=project_root() / "config/mission_spec.json")
    parser.add_argument("--apply", action="store_true", help="explicitly call /apply_planning_scene")
    parser.add_argument("--dry-run", action="store_true", help="build the ROS message only; this is the default")
    parser.add_argument("--service-timeout-s", type=float, default=5.0, help="bounded timeout used only with --apply")
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run cannot be combined")
    return args


def main() -> int:
    args = parse_args()
    try:
        request = select_episode(load_teacher_requests(args.requests), args.episode_id)
        document, planning_scene = prepare_scene(request, load_mission_spec(args.mission_spec))
        if args.apply:
            apply_via_service(planning_scene, args.service_timeout_s)
            mode = "applied_to_moveit_planning_scene_only"
        else:
            mode = "dry_run_no_ros_service_call"
        print(json.dumps(summary(document, mode), ensure_ascii=False, sort_keys=True))
    except (TeacherSceneError, TeacherSceneApplyError, RosMessagesUnavailable, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
