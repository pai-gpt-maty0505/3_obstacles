"""Deterministically make a portable MoveIt PlanningScene document.

This module intentionally has no ROS, Gazebo, controller, or OMPL runtime
dependency.  It prepares collision objects and route metadata for an *offline*
teacher worker only; the final mission must replay a validated result and must
not invoke OMPL online.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .geometry import box, compose_local_yaw_pose, cylinder, pose_from_yaw_pose, positive, triangular_pyramid_mesh


REQUEST_SCHEMA = "three_obstacles_teacher_request/v1"
MISSION_SCHEMA = "three_obstacles_mission_spec/v1"
SCENE_SCHEMA = "three_obstacles_teacher_scene/v1"
ENTITIES = ("payload", "dish", "cylinder", "rectangular_prism", "triangular_pyramid")
GATES = ("PYRAMID_OUTER", "RECTANGLE_FRONT", "CYLINDER_OUTER", "DISH_PLACE")
GATE_ENTITY = dict(zip(GATES, ("triangular_pyramid", "rectangular_prism", "cylinder", "dish")))


class TeacherSceneError(ValueError):
    """Input does not meet the offline teacher-scene contract."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_mission_spec(path: Path | None = None) -> dict[str, Any]:
    source = path or project_root() / "config/mission_spec.json"
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TeacherSceneError(f"cannot load mission specification {source}: {error}") from error
    if document.get("schema_version") != MISSION_SCHEMA:
        raise TeacherSceneError("unsupported mission specification schema")
    return document


def load_teacher_requests(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise TeacherSceneError(f"cannot read request JSONL {path}: {error}") from error
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TeacherSceneError(f"{path}:{number}: invalid JSON: {error.msg}") from error
        if not isinstance(record, dict):
            raise TeacherSceneError(f"{path}:{number}: a record must be an object")
        records.append(record)
    if not records:
        raise TeacherSceneError(f"{path}: no records")
    return records


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TeacherSceneError(f"{label} must be an object")
    return value


def _expected_scene_hash(poses: dict[str, Any]) -> str:
    return json_sha256(poses)[:16]


def validate_teacher_request(request: dict[str, Any]) -> None:
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise TeacherSceneError("unsupported teacher-request schema")
    if request.get("status") != "PENDING_OMPL":
        raise TeacherSceneError("teacher request must have status PENDING_OMPL")
    if not isinstance(request.get("episode_id"), str) or not request["episode_id"]:
        raise TeacherSceneError("episode_id must be a nonempty string")
    if not isinstance(request.get("coordinate_frame"), str) or not request["coordinate_frame"]:
        raise TeacherSceneError("coordinate_frame must be a nonempty string")
    poses = _mapping(request.get("entity_poses"), "entity_poses")
    if set(poses) != set(ENTITIES):
        raise TeacherSceneError(f"entity_poses must contain exactly {list(ENTITIES)}")
    for name in ENTITIES:
        try:
            pose_from_yaw_pose(_mapping(poses[name], f"entity_poses.{name}"))
        except ValueError as error:
            raise TeacherSceneError(str(error)) from error
    if request.get("scene_hash") != _expected_scene_hash(poses):
        raise TeacherSceneError("scene_hash does not match entity_poses")
    if _mapping(request.get("precheck"), "precheck").get("passed") is not True:
        raise TeacherSceneError("static precheck did not pass")
    gates = request.get("route_gates")
    if not isinstance(gates, list) or [gate.get("id") if isinstance(gate, dict) else None for gate in gates] != list(GATES):
        raise TeacherSceneError(f"route_gates must preserve order {list(GATES)}")
    for gate in gates:
        if gate.get("entity") != GATE_ENTITY[gate["id"]]:
            raise TeacherSceneError(f"route gate {gate['id']} has wrong entity")
        position = _mapping(gate.get("position"), f"route gate {gate['id']}.position")
        for axis in ("x", "y", "z"):
            try:
                value = float(position[axis])
            except (KeyError, TypeError, ValueError) as error:
                raise TeacherSceneError(f"route gate {gate['id']} position.{axis} is invalid") from error
            if not math.isfinite(value):
                raise TeacherSceneError(f"route gate {gate['id']} position.{axis} is invalid")
    contract = _mapping(request.get("teacher_contract"), "teacher_contract")
    if contract.get("online_replanning") is not False:
        raise TeacherSceneError("online_replanning must remain false")
    if contract.get("planner") != "OMPL RRTConnect":
        raise TeacherSceneError("only offline OMPL RRTConnect is allowed as teacher")
    if not isinstance(contract.get("planning_group"), str) or not contract["planning_group"]:
        raise TeacherSceneError("planning_group is required")


def _object(object_id: str, frame_id: str, role: str, *, primitives: list[dict[str, Any]] | None = None,
            primitive_poses: list[dict[str, Any]] | None = None, meshes: list[dict[str, Any]] | None = None,
            mesh_poses: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    primitives, primitive_poses = primitives or [], primitive_poses or []
    meshes, mesh_poses = meshes or [], mesh_poses or []
    if len(primitives) != len(primitive_poses) or len(meshes) != len(mesh_poses):
        raise AssertionError(f"shape and pose counts differ for {object_id}")
    return {"id": object_id, "frame_id": frame_id, "operation": "ADD", "role": role,
            "primitives": primitives, "primitive_poses": primitive_poses, "meshes": meshes, "mesh_poses": mesh_poses}


def _dish(spec: dict[str, Any], pose: dict[str, Any], frame_id: str) -> dict[str, Any]:
    """Use a base cylinder and 16 small boxes as a conservative dish proxy."""
    outer_radius = positive(spec["outer_diameter_m"], "dish.outer_diameter") / 2.0
    depth = positive(spec["depth_m"], "dish.depth")
    wall, base, segments = 0.002, min(0.002, depth / 3.0), 16
    rim_radius = outer_radius - wall / 2.0
    tangent_length = (2.0 * math.pi * rim_radius / segments) * 1.08
    primitives, poses = [cylinder(base, outer_radius)], [compose_local_yaw_pose(pose, 0, 0, base / 2.0, 0)]
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        primitives.append(box((wall, tangent_length, depth)))
        poses.append(compose_local_yaw_pose(pose, rim_radius * math.cos(angle), rim_radius * math.sin(angle), depth / 2.0, angle + math.pi / 2.0))
    return _object("teacher/dish", frame_id, "static_goal_receptacle", primitives=primitives, primitive_poses=poses)


def _collision_objects(spec: dict[str, Any], request: dict[str, Any]) -> list[dict[str, Any]]:
    entities, poses, frame = spec["entities"], request["entity_poses"], request["coordinate_frame"]
    cylinder_spec = _mapping(entities["cylinder"], "entities.cylinder")
    rectangle_spec = _mapping(entities["rectangular_prism"], "entities.rectangular_prism")
    pyramid_spec = _mapping(entities["triangular_pyramid"], "entities.triangular_pyramid")
    payload_spec = _mapping(entities["payload"], "entities.payload")
    return [
        _object("teacher/cylinder", frame, "static_obstacle", primitives=[cylinder(cylinder_spec["height_m"], positive(cylinder_spec["diameter_m"], "cylinder.diameter") / 2.0)], primitive_poses=[pose_from_yaw_pose(poses["cylinder"])]),
        _object("teacher/rectangular_prism", frame, "static_obstacle", primitives=[box(rectangle_spec["size_m"])], primitive_poses=[pose_from_yaw_pose(poses["rectangular_prism"])]),
        _object("teacher/triangular_pyramid", frame, "static_obstacle", meshes=[triangular_pyramid_mesh(pyramid_spec["base_side_m"], pyramid_spec["height_m"])], mesh_poses=[pose_from_yaw_pose(poses["triangular_pyramid"])]),
        _object("teacher/payload", frame, "dynamic_payload_start_pose", primitives=[box(payload_spec["size_m"])], primitive_poses=[pose_from_yaw_pose(poses["payload"])]),
        _dish(_mapping(entities["dish"], "entities.dish"), poses["dish"], frame),
    ]


def _route_plan(request: dict[str, Any]) -> dict[str, Any]:
    gates = []
    for ordinal, source in enumerate(request["route_gates"], start=1):
        gates.append({"ordinal": ordinal, "id": source["id"], "entity": source["entity"], "semantic_side": source["semantic_side"],
                      "position": {axis: float(source["position"][axis]) for axis in ("x", "y", "z")},
                      "completion_rule": "end_effector_reference_point_within_gate_tolerance"})
    endpoints = ["POST_GRASP"] + [gate["id"] for gate in gates]
    return {"frame_id": request["coordinate_frame"], "gate_tolerance_m": 0.015, "gates": gates,
            "segments": [{"ordinal": index, "from": endpoints[index - 1], "to": endpoints[index]} for index in range(1, len(endpoints))],
            "constraint": "complete gates strictly in ascending ordinal order",
            "planner_visibility": "metadata_only_not_a_collision_object"}


def build_finalized_teacher_scene(request: dict[str, Any], mission_spec: dict[str, Any]) -> dict[str, Any]:
    """Create a stable scene document without invoking any planner or robot."""
    validate_teacher_request(request)
    if mission_spec.get("schema_version") != MISSION_SCHEMA:
        raise TeacherSceneError("unsupported mission specification")
    if mission_spec.get("coordinate_frame", {}).get("frame_id") != request["coordinate_frame"]:
        raise TeacherSceneError("mission spec and request coordinate frames differ")
    if set(_mapping(mission_spec.get("entities"), "entities")) != set(ENTITIES):
        raise TeacherSceneError("mission specification entity set differs from request")
    document = {
        "schema_version": SCENE_SCHEMA,
        "source_request": {"episode_id": request["episode_id"], "scene_hash": request["scene_hash"], "scene_seed": request.get("scene_seed"), "split": request.get("split"), "request_sha256": json_sha256(request)},
        "planning_scene": {"is_diff": True, "world": {"collision_objects": _collision_objects(mission_spec, request)},
                           "robot_state": {"is_diff": True, "note": "No start state or controller command is generated here."}},
        "route_plan": _route_plan(request),
        "teacher_policy": {
            "planning_group": request["teacher_contract"]["planning_group"], "offline_planner": "OMPL RRTConnect",
            "offline_teacher_only": True, "online_replanning_allowed": False,
            "payload_policy": "Planning proxy only; Gazebo payload must remain a dynamic rigid body under gravity and friction.",
            "final_mission_policy": "Replay a validated learned/offline trajectory only; never invoke online OMPL.",
        },
    }
    document["document_sha256"] = json_sha256(document)
    return document


def select_episode(records: Iterable[dict[str, Any]], episode_id: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("episode_id") == episode_id]
    if len(matches) != 1:
        raise TeacherSceneError(f"episode_id must occur exactly once: {episode_id}")
    return matches[0]


def serialize_scene(document: dict[str, Any]) -> str:
    return canonical_json(document) + "\n"
