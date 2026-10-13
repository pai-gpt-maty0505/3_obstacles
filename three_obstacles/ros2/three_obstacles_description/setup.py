from setuptools import find_packages, setup


package_name = "three_obstacles_description"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="3_obstacles project",
    maintainer_email="maty05051960@gmail.com",
    description="Non-destructive SO-101 dual-finger description overlay",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "generate_dual_finger_urdf = three_obstacles_description.overlay_generator:main",
        ],
    },
)
