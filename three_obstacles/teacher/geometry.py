"""ROS-independent conversion of yaw-only request geometry to scene geometry."""

from __future__ import annotations

import math
from typing import Any, Iterable


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def positive(value: Any, label: str) -> float:
    number = finite(value, label)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def pose_from_yaw_pose(value: dict[str, Any]) -> dict[str, Any]:
    """Convert request ``x/y/z/yaw_deg`` to JSON equivalent of geometry_msgs/Pose."""
    if set(value) != {"x", "y", "z", "yaw_deg"}:
        raise ValueError("pose must contain exactly x, y, z, yaw_deg")
    yaw = math.radians(finite(value["yaw_deg"], "yaw_deg")) / 2.0
    return {
        "position": {axis: finite(value[axis], f"pose.{axis}") for axis in ("x", "y", "z")},
        "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(yaw), "w": math.cos(yaw)},
    }


def box(size: Iterable[Any]) -> dict[str, Any]:
    values = tuple(size)
    if len(values) != 3:
        raise ValueError("box size must have three dimensions")
    return {
        "type": "BOX",
        "dimensions_m": {axis: positive(value, f"box.{axis}") for axis, value in zip(("x", "y", "z"), values)},
    }


def cylinder(height_m: Any, radius_m: Any) -> dict[str, Any]:
    return {
        "type": "CYLINDER",
        "dimensions_m": {"height": positive(height_m, "cylinder.height"), "radius": positive(radius_m, "cylinder.radius")},
    }


def triangular_pyramid_mesh(base_side_m: Any, height_m: Any) -> dict[str, Any]:
    """Origin-centred equilateral triangular pyramid, with its base at -h/2."""
    side = positive(base_side_m, "pyramid.base_side")
    height = positive(height_m, "pyramid.height")
    radius = side / math.sqrt(3.0)
    vertices = []
    for index in range(3):
        angle = math.pi / 2.0 + index * 2.0 * math.pi / 3.0
        vertices.append({"x": radius * math.cos(angle), "y": radius * math.sin(angle), "z": -height / 2.0})
    vertices.append({"x": 0.0, "y": 0.0, "z": height / 2.0})
    return {"vertices": vertices, "triangles": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]}


def compose_local_yaw_pose(parent: dict[str, Any], local_x: float, local_y: float, local_z: float, local_yaw: float) -> dict[str, Any]:
    """Compose a yaw-only parent pose with a local yaw-only primitive pose."""
    parent_yaw = math.radians(finite(parent["yaw_deg"], "parent.yaw_deg"))
    cos_yaw, sin_yaw = math.cos(parent_yaw), math.sin(parent_yaw)
    total_half_yaw = (parent_yaw + local_yaw) / 2.0
    return {
        "position": {
            "x": finite(parent["x"], "parent.x") + cos_yaw * local_x - sin_yaw * local_y,
            "y": finite(parent["y"], "parent.y") + sin_yaw * local_x + cos_yaw * local_y,
            "z": finite(parent["z"], "parent.z") + local_z,
        },
        "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(total_half_yaw), "w": math.cos(total_half_yaw)},
    }
