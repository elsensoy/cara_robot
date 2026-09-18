cmake_minimum_required(VERSION 3.8)
project(cara_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

# Build the INA219 / BNO055 I2C drivers. Needs libi2c-dev in the container.
# colcon build --packages-select cara_control --cmake-args -DCARA_WITH_HARDWARE=OFF
option(CARA_WITH_HARDWARE "Build INA219/BNO055 I2C drivers" ON)

# The signal path itself has a single source of truth in jetson/control/.
# Override with -DPIPELINE_DIR=... if the repo layout differs.
if(NOT DEFINED PIPELINE_DIR)
  set(PIPELINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../jetson/control)
endif()
get_filename_component(PIPELINE_DIR ${PIPELINE_DIR} ABSOLUTE)
if(NOT EXISTS ${PIPELINE_DIR}/src/pipeline.cpp)
  message(FATAL_ERROR
    "cara_control: pipeline sources not found at ${PIPELINE_DIR}. "
    "Pass -DPIPELINE_DIR=/abs/path/to/jetson/control")
endif()
message(STATUS "cara_control: using pipeline from ${PIPELINE_DIR}")

add_executable(cara_control_node
  src/cara_control_node.cpp
  ${PIPELINE_DIR}/src/pipeline.cpp
  ${PIPELINE_DIR}/src/diagnostics.cpp
  ${PIPELINE_DIR}/src/sources_sim.cpp
  ${PIPELINE_DIR}/src/sources_hw.cpp
)
target_include_directories(cara_control_node PRIVATE ${PIPELINE_DIR}/include)
ament_target_dependencies(cara_control_node rclcpp std_msgs trajectory_msgs)

if(CARA_WITH_HARDWARE)
  target_compile_definitions(cara_control_node PRIVATE CARA_WITH_HARDWARE=1)
  target_link_libraries(cara_control_node i2c)
else()
  target_compile_definitions(cara_control_node PRIVATE CARA_WITH_HARDWARE=0)
endif()

install(TARGETS cara_control_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

# Test helper: stand-in for cara_body_node (no ONNX policy needed).
#   ros2 run cara_control fake_body_policy
install(PROGRAMS scripts/fake_body_policy.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME fake_body_policy)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
