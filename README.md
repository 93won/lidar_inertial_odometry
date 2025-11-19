# LiDAR-Inertial Odometry

Tightly-coupled LiDAR-Inertial Odometry using Iterated Extended Kalman Filter with direct point-to-plane residuals.

### MIT License

## Features

- **Iterated Extended Kalman Filter (IEKF)**: Direct LiDAR-IMU fusion with nested iteration for re-linearization and convergence
- **Incremental Voxel hashing for fast correspondence search**: Hash-based spatial indexing with O(1) lookup
- **Motion compensation**: IMU-based undistortion for moving LiDAR scans

## Demo

[![LIO Demo](https://img.youtube.com/vi/2JymC0LWDWI/0.jpg)](https://www.youtube.com/watch?v=2JymC0LWDWI)



### Installation (Ubuntu 22.04)
```bash
sudo apt update
sudo apt install cmake libeigen3-dev libglew-dev libyaml-cpp-dev
```

Pangolin and spdlog are included in `thirdparty/` directory.

## Build

```bash
cd lidar_inertial_odometry
./build.sh
```

This will:
1. Build Pangolin from `thirdparty/pangolin`
2. Build the main project with CMake

## Usage

### Quick Start



**Livox AVIA Dataset**:
- **Download**: [Google Drive Link](https://drive.google.com/file/d/1NPtqg34vdAM-BMdqQ_pfRgVvzVVRuXd6/view?usp=sharing)
- **Source**: [R3LIVE Dataset](https://github.com/hku-mars/r3live)
- **Sensor**: Livox Avia LiDAR + Built-in IMU


```bash
cd build
./lio_player ../config/avia.yaml /home/user/data/R3LIVE/hku_main_building
```

**Livox Mid-360 Dataset** :
- **Download**: [Google Drive Link](https://drive.google.com/file/d/1VuzTaSFqwiK6E8N19AOB-KV6xgEf4554/view?usp=sharing)
- **Source**: https://www.youtube.com/watch?v=u8siB0KLFLc
- **Sensor**: Livox Mid-360 LiDAR + Built-in IMU
```

```bash
cd build
./lio_player ../config/mid360.yaml /path/to/mid360_dataset
```

## Project Structure

```
lidar_inertial_odometry/
├── src/
│   ├── core/             # Core algorithm implementation
│   │   ├── Estimator.h/cpp      # IEKF-based LIO estimator
│   │   ├── State.h/cpp          # 18-dim state representation
│   │   └── VoxelMap.h/cpp       # Hash-based voxel map for fast KNN
│   │
│   ├── util/             # Utility functions
│   │   ├── LieUtils.h/cpp       # SO3/SE3 Lie group operations
│   │   ├── PointCloudUtils.h/cpp # Point cloud processing
│   │   └── ConfigUtils.h/cpp    # YAML configuration loader
│   │
│   └── viewer/           # Visualization
│       └── LIOViewer.h/cpp      # Pangolin-based 3D viewer
│
├── app/                  # Application executables
│   └── lio_player.cpp    # Dataset player with live visualization
│
├── config/               # Configuration files
│   └── avia.yaml         # Parameters for Livox Avia LiDAR
│
├── thirdparty/           # Third-party libraries
│   ├── pangolin/         # 3D visualization
│   └── spdlog/           # Logging (header-only)
│
├── CMakeLists.txt        # CMake build configuration
└── README.md             # This file
```




