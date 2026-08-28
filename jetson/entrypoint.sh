#!/usr/bin/env bash
set -e

# --- General env ---
export SETUPTOOLS_USE_DISTUTILS=stdlib
#python3 -m pip install -q "pip<24" "setuptools==65.7.0" "wheel<0.42"
export COLCON_PYTHON_SETUP_PY_STRATEGY=install
export GST_GL_PLATFORM=x11
export QT_QPA_PLATFORM=xcb
export LIBGL_ALWAYS_INDIRECT=1

# Audio
export AUDIODEV="pulse"
export ALSA_DEFAULT_PCM="pulse"
export ALSA_DEFAULT_CTL="pulse"
export DISPLAY=":0"


# python3 -m pip install -q "pip<24" "setuptools==65.7.0" "wheel<0.42"
# --- Source ROS 2 distro ---
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi

# --- Build & source workspace if present ---
WS=/workspace/ros2_ws

if [ -d "$WS/src" ]; then
  cd "$WS"

  # Only build if not already built
  if [ ! -f "$WS/install/setup.bash" ]; then
    rosdep update || true
    rosdep install --from-paths src --ignore-src -r -y || true
    colcon build --symlink-install
  fi

  if [ -f "$WS/install/setup.bash" ]; then
    source "$WS/install/setup.bash"
  fi
fi

# Hand off to whatever CMD / docker-compose says
exec "$@"

