from glob import glob
from setuptools import find_packages, setup


package_name = "three_obstacles_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pai-gpt-maty0505",
    maintainer_email="maty05051960@gmail.com",
    description="Project-owned launch integration for physical three-obstacle transport.",
    license="Apache-2.0",
)
