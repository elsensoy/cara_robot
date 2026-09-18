cmake_minimum_required(VERSION 3.16)
project(car_control CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Build the INA219 / BNO055 I2C drivers. Needs libi2c-dev. Turn OFF to build the
# simulation-only binary on a machine without I2C headers.
option(CARA_WITH_HARDWARE "Build INA219/BNO055 I2C drivers" ON)

add_executable(cara_control
  src/main.cpp
  src/pipeline.cpp
  src/diagnostics.cpp
  src/sources_sim.cpp
  src/sources_hw.cpp
)

target_include_directories(cara_control PRIVATE include)
target_compile_options(cara_control PRIVATE -O2 -Wall -Wextra)

if(CARA_WITH_HARDWARE)
  target_compile_definitions(cara_control PRIVATE CARA_WITH_HARDWARE=1)
  target_link_libraries(cara_control PRIVATE i2c)
else()
  target_compile_definitions(cara_control PRIVATE CARA_WITH_HARDWARE=0)
endif()

find_package(Threads REQUIRED)
target_link_libraries(cara_control PRIVATE Threads::Threads)
