import sys
from setuptools import setup
from setuptools.command.develop import develop

package_name = 'cara_gaze_control'

# ------------------------------------------------------------------
# Strip colcon/ament-specific options that confuse setuptools/distutils
# (and the paths that come right after them)
# ------------------------------------------------------------------
FORBIDDEN_FLAGS = [
    '--uninstall',
    '--editable',
    '--build-directory',
    '--record',
    '--single-version-externally-managed',
]
class DevelopCommand(develop):
    user_options = develop.user_options + [
        ('script-dir=', None, 'ignored script directory option'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.script_dir = None

    def finalize_options(self):
        super().finalize_options()
        
        
def clean_argv(argv):
    """
    Remove ament/colcon flags like:
      --uninstall
      --editable
      --build-directory /some/path
    and also drop the *next* arg when it is a value for one of these flags.
    """
    new_argv = [argv[0]]  # keep script name
    skip_next = False

    for arg in argv[1:]:
        if skip_next:
            # skip the value associated with a forbidden flag
            skip_next = False
            continue

        # Drop flags like "--build-directory", "--uninstall", etc.
        drop_this = False
        for flag in FORBIDDEN_FLAGS:
            # forms: '--flag' or '--flag=VALUE'
            if arg == flag or arg.startswith(flag + '='):
                drop_this = True
                # if it's exactly '--flag' (no '=value'), skip next arg too
                if '=' not in arg:
                    skip_next = True
                break

        if drop_this:
            continue

        # Also be defensive: drop any direct build path that slipped through
        if arg.startswith('/workspace/ros2_ws/build/'):
            continue

        new_argv.append(arg)

    return new_argv

sys.argv = clean_argv(sys.argv)
# ------------------------------------------------------------------


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        # ament index
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # launch
        ('share/' + package_name + '/launch', [
            'launch/model_gaze.launch.py',
        ]),
        # scripts installed into lib/<package_name>
        ('lib/' + package_name, [
            'scripts/model_gaze_mapper',
            'scripts/gaze_debugger',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Elida',
    maintainer_email='elida@umich.edu',
    description='Model-based gaze tracking controller using DS dynamics.',
    license='MIT',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'model_gaze_mapper = cara_gaze_control.model_gaze_mapper:main',
            'gaze_debugger = cara_gaze_control.gaze_debugger:main',
        ],
    },
    cmdclass={
        'develop': DevelopCommand,
    },
)
