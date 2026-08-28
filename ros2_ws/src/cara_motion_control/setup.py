from setuptools import setup
import os
from glob import glob

package_name = "cara_motion_control"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
	data_files=[
       	    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
       	    ('share/' + package_name, ['package.xml']),
            # This line copies your launch files into the install directory
            (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
            (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
            (os.path.join('share', package_name, 'urdf', 'meshes'), glob('urdf/meshes/*.urdf')),
    	],
             
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="elida_z",
    description="Cara motion control system.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "head_expression_node = cara_motion_control.head_expression_node:main",
            "cara_body_node = cara_motion_control.cara_body_node:main",
            "cara_actuator_node = cara_motion_control.cara_actuator_node:main",
            "cara_health_monitor = cara_motion_control.cara_health_monitor:main",
            "cara_imu_node = cara_motion_control.cara_imu_node:main",
            "cara_servo_test_node = cara_motion_control.cara_servo_test_node:main",
        ],
    },
)
