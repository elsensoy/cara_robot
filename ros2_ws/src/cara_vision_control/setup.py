from setuptools import setup
import os
from glob import glob

package_name = 'cara_vision_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # THIS NEW LINE COPIES THE MODELS:
        (os.path.join('share', package_name, 'models'), glob('models/*.onnx')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Elida',
    maintainer_email='elsensoy@umich.edu',
    description='Cara vision nodes',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'behavior_node = cara_vision_control.behavior_node:main',
            'emotion_node = cara_vision_control.cara_emotion_node:main',
            'servo_pca9685 = cara_vision_control.cara_servo_pca9685_node:main',
            'arduino_bridge_node = cara_vision_control.arduino_bridge_node:main',
            'face_yunet_node = cara_vision_control.face_yunet_node:main',
        ],
    },
)
