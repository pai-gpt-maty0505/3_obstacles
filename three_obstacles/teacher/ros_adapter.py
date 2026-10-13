"""Optional adapter from the portable document to MoveIt ROS messages.

It imports ROS only when called.  It publishes nothing, creates no client, and
cannot command the robot.  This keeps the geometry layer unit-testable on a
plain Python installation.
"""

from __future__ import annotations

from typing import Any


class RosMessagesUnavailable(RuntimeError):
    pass


def _imports() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from geometry_msgs.msg import Point, Pose
        from moveit_msgs.msg import CollisionObject, PlanningScene
        from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
    except ImportError as error:
        raise RosMessagesUnavailable("ROS 2 MoveIt message packages are not available; source the dedicated workspace before calling this adapter") from error
    return Point, Pose, CollisionObject, PlanningScene, (Mesh, MeshTriangle, SolidPrimitive)


def _pose(payload: dict[str, Any], pose_type: Any) -> Any:
    message = pose_type()
    for axis in ("x", "y", "z"):
        setattr(message.position, axis, payload["position"][axis])
    for axis in ("x", "y", "z", "w"):
        setattr(message.orientation, axis, payload["orientation"][axis])
    return message


def to_moveit_planning_scene(document: dict[str, Any]) -> Any:
    """Build an in-memory ``moveit_msgs/PlanningScene``; no service call occurs."""
    Point, Pose, CollisionObject, PlanningScene, mesh_types = _imports()
    Mesh, MeshTriangle, SolidPrimitive = mesh_types
    scene = PlanningScene()
    scene.is_diff = bool(document["planning_scene"]["is_diff"])
    for portable in document["planning_scene"]["world"]["collision_objects"]:
        collision = CollisionObject()
        collision.id = portable["id"]
        collision.header.frame_id = portable["frame_id"]
        collision.operation = CollisionObject.ADD
        for primitive, primitive_pose in zip(portable["primitives"], portable["primitive_poses"]):
            message = SolidPrimitive()
            if primitive["type"] == "BOX":
                message.type = SolidPrimitive.BOX
                dimensions = primitive["dimensions_m"]
                message.dimensions = [dimensions["x"], dimensions["y"], dimensions["z"]]
            elif primitive["type"] == "CYLINDER":
                message.type = SolidPrimitive.CYLINDER
                dimensions = primitive["dimensions_m"]
                message.dimensions = [dimensions["height"], dimensions["radius"]]
            else:
                raise ValueError(f"unsupported primitive: {primitive['type']}")
            collision.primitives.append(message)
            collision.primitive_poses.append(_pose(primitive_pose, Pose))
        for mesh, mesh_pose in zip(portable["meshes"], portable["mesh_poses"]):
            message = Mesh()
            for vertex in mesh["vertices"]:
                point = Point()
                point.x, point.y, point.z = vertex["x"], vertex["y"], vertex["z"]
                message.vertices.append(point)
            for triangle in mesh["triangles"]:
                face = MeshTriangle()
                face.vertex_indices = triangle
                message.triangles.append(face)
            collision.meshes.append(message)
            collision.mesh_poses.append(_pose(mesh_pose, Pose))
        scene.world.collision_objects.append(collision)
    return scene
