from setuptools import setup

package_name = 'face_id_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        # ament index marker (so ROS 2 can discover the package)
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/face_id.launch.py']),  #
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Face ID service/node for ROS 2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'enroll_server = face_id_ros.enroll_server:main',
        ],
    },
)
