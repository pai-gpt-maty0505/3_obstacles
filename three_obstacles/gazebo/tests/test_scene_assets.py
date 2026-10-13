"""Static, no-simulator checks for project-owned SDF scene assets."""

from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
WORLD = ROOT / "worlds" / "three_obstacles_baseline.world"
SPAWN = ROOT / "config" / "baseline_spawn_poses.json"


def text(node: ET.Element, path: str) -> str:
    child = node.find(path)
    if child is None or child.text is None:
        raise AssertionError(f"missing SDF element: {path}")
    return child.text.strip()


class SceneAssetTests(unittest.TestCase):
    def parse_model(self, name: str) -> ET.Element:
        return ET.parse(MODELS / name / "model.sdf").getroot().find("model")  # type: ignore[return-value]

    def test_all_sdf_files_parse(self) -> None:
        for path in sorted(ROOT.rglob("*.sdf")):
            with self.subTest(path=path):
                ET.parse(path)

    def test_payload_is_dynamic_with_specified_mass_and_inertia(self) -> None:
        model = self.parse_model("payload")
        self.assertEqual(text(model, "static"), "false")
        self.assertEqual(text(model, "link/gravity"), "true")
        self.assertEqual(float(text(model, "link/inertial/mass")), 0.036)
        self.assertEqual(float(text(model, "link/inertial/inertia/ixx")), 7.5e-6)
        self.assertEqual(float(text(model, "link/inertial/inertia/iyy")), 7.5e-6)
        self.assertEqual(float(text(model, "link/inertial/inertia/izz")), 5.4e-6)
        self.assertEqual(text(model, "link/collision/geometry/box/size"), "0.03 0.03 0.04")

    def test_obstacle_dimensions_and_dish_depth(self) -> None:
        cylinder = self.parse_model("cylinder")
        self.assertEqual(text(cylinder, "link/collision/geometry/cylinder/radius"), "0.04")
        self.assertEqual(text(cylinder, "link/collision/geometry/cylinder/length"), "0.10")
        prism = self.parse_model("tall_rectangular_prism")
        self.assertEqual(text(prism, "link/collision/geometry/box/size"), "0.06 0.06 0.20")
        pyramid_mesh = MODELS / "triangular_pyramid" / "meshes" / "triangular_pyramid.stl"
        self.assertEqual(pyramid_mesh.read_text(encoding="utf-8").count("facet normal"), 4)
        dish = self.parse_model("dish")
        self.assertEqual(len(dish.findall("link/collision")), 17)  # floor + 16 rim segments
        self.assertEqual(text(dish, "link/collision/geometry/cylinder/radius"), "0.03")
        self.assertEqual(text(dish, "link/collision/geometry/cylinder/length"), "0.002")

    def test_project_owned_contact_surfaces_use_mu_point_five(self) -> None:
        for model_path in sorted(MODELS.glob("*/model.sdf")):
            model = ET.parse(model_path).getroot().find("model")
            assert model is not None
            collisions = model.findall(".//collision")
            self.assertTrue(collisions, model_path)
            for collision in collisions:
                with self.subTest(model=model_path.parent.name, collision=collision.attrib["name"]):
                    self.assertEqual(text(collision, "surface/friction/ode/mu"), "0.5")
                    self.assertEqual(text(collision, "surface/friction/ode/mu2"), "0.5")

    def test_baseline_world_matches_spawn_contract(self) -> None:
        world = ET.parse(WORLD).getroot().find("world")
        assert world is not None
        self.assertEqual(text(world, "gravity"), "0 0 -9.81")
        expected = json.loads(SPAWN.read_text(encoding="utf-8"))["entities"]
        seen = {}
        for include in world.findall("include"):
            name = include.findtext("name")
            if name in expected:
                seen[name] = include
        self.assertEqual(set(seen), set(expected))
        for name, entity in expected.items():
            include = seen[name]
            self.assertEqual(include.findtext("uri"), entity["model_uri"])
            self.assertEqual([float(v) for v in include.findtext("pose").split()], entity["pose"])

    def test_no_attachment_or_joint_mechanism_in_project_sdfs(self) -> None:
        prohibited_tags = {"joint", "plugin"}
        for path in sorted(ROOT.rglob("*.sdf")):
            tree = ET.parse(path)
            with self.subTest(path=path):
                self.assertFalse(prohibited_tags.intersection(elem.tag for elem in tree.iter()))


if __name__ == "__main__":
    unittest.main()
