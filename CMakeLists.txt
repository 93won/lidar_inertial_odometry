# LiDAR-Inertial Odometry CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(LidarInertialOdometry)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Find packages
find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

# Use Pangolin from thirdparty
set(Pangolin_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/pangolin)
include_directories(${Pangolin_DIR}/components/pango_core/include)
include_directories(${Pangolin_DIR}/components/pango_display/include)
include_directories(${Pangolin_DIR}/components/pango_opengl/include)
include_directories(${Pangolin_DIR}/components/pango_image/include)
include_directories(${Pangolin_DIR}/components/pango_windowing/include)
include_directories(${Pangolin_DIR}/components/pango_vars/include)
include_directories(${Pangolin_DIR}/components/pango_plot/include)
include_directories(${Pangolin_DIR}/components/pango_geometry/include)
include_directories(${Pangolin_DIR}/components/pango_glgeometry/include)
include_directories(${Pangolin_DIR}/build/include)
link_directories(${Pangolin_DIR}/build)

# Use spdlog from thirdparty (header-only)
set(spdlog_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/spdlog/include)
include_directories(${spdlog_DIR})

# Use unordered_dense from thirdparty (header-only, fast hash map)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/unordered_dense)

# LidarIris for loop closure detection
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/LidarIris)

# Include directories
include_directories(src)
include_directories(src/core)
include_directories(src/util)
include_directories(src/viewer)
include_directories(app)

# Core library (viewer + estimator)
add_library(lio_viewer
    src/viewer/LIOViewer.cpp
    src/util/LieUtils.cpp
    src/util/PointCloudUtils.cpp
    src/util/ConfigUtils.cpp
    src/core/VoxelMap.cpp
    src/core/State.cpp
    src/core/Estimator.cpp
    src/core/ProbabilisticKernelOptimizer.cpp
    src/core/LoopClosureDetector.cpp
    thirdparty/LidarIris/LidarIris.cpp
    thirdparty/LidarIris/fftm/fftm.cpp
)

target_link_libraries(lio_viewer
    Eigen3::Eigen
    yaml-cpp
    pango_display
    pango_plot
    pango_opengl
    pango_windowing
    pango_vars
    pango_core
    GLEW::GLEW
    GL
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
)

# Main LIO player executable
add_executable(lio_player 
    app/lio_player.cpp
)

target_link_libraries(lio_player 
    lio_viewer
    Eigen3::Eigen
    yaml-cpp
    pango_display
    pango_plot
    pango_opengl
    pango_windowing
    pango_vars
    pango_core
    GLEW::GLEW
    GL
    pthread
)

target_compile_options(lio_player PRIVATE -Wall -Wextra -O2)