# LiDAR-Inertial Odometry CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(LidarInertialOdometry)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Find packages
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Pangolin REQUIRED)
find_package(GLEW REQUIRED)

# Include directories
include_directories(include)
include_directories(app)
include_directories(src)

# Core library (viewer + estimator)
add_library(lio_viewer
    src/LIOViewer.cpp
    src/LieUtils.cpp
    src/PointCloudUtils.cpp
    src/State.cpp
    src/Estimator.cpp
)

target_link_libraries(lio_viewer
    Eigen3::Eigen
    spdlog::spdlog
    ${Pangolin_LIBRARIES}
    GLEW::GLEW
    GL
)

# Main LIO player executable
add_executable(lio_player 
    app/lio_player.cpp
)

target_link_libraries(lio_player 
    lio_viewer
    Eigen3::Eigen
    spdlog::spdlog
    ${Pangolin_LIBRARIES}
    GLEW::GLEW
    GL
)

target_compile_options(lio_player PRIVATE -Wall -Wextra -O2)