# Surfel-LIO

Surfel-LIO: Efficient LiDAR-Inertial Odometry via Pre-computed Surfels in Hierarchical Voxel Hash Maps

### MIT License

## Features

- **Iterated Extended Kalman Filter (IEKF)**: Direct LiDAR-IMU fusion with nested iteration for re-linearization and convergence
- **Adaptive Robust Estimation**: Probabilistic Kernel Optimization (PKO) for automatic Huber loss scale tuning
- **Incremental Hierarchical Voxel Map**: 3-level hash-based spatial indexing (L0/L1/L2) with occupied-only tracking for fast local KNN search
- **Pre-computed Surfel Planes**: L1 voxels store fitted plane surfels (normal, centroid, covariance) computed via SVD, enabling O(1) correspondence finding without per-point KNN/SVD
- **Motion compensation**: IMU-based undistortion for moving LiDAR scans

### Probabilistic Kernel Optimization (PKO)

This project implements adaptive robust estimation using Probabilistic Kernel Optimization for automatic Huber loss scale tuning. If you use this method in your research, please cite:

```bibtex
@article{choi2025pko,
  title={Probabilistic Kernel Optimization for Robust State Estimation},
  author={Choi, Seungwon and Kim, Tae-Wan},
  journal={IEEE Robotics and Automation Letters},
  volume={10},
  number={3},
  pages={2998--3005},
  year={2025},
  publisher={IEEE}
}
```

## Demo

[![LIO Demo](https://img.youtube.com/vi/2JymC0LWDWI/0.jpg)](https://www.youtube.com/watch?v=2JymC0LWDWI)



### Installation (Ubuntu 20.04)

```bash
cd lidar_inertial_odometry
./build.sh
```

This will:
1. Build Pangolin from `thirdparty/pangolin`
2. Build the main project with CMake

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


```bash
cd build
./lio_player ../config/mid360.yaml /path/to/mid360_dataset
```

## Project Structure

```
lidar_inertial_odometry/
├── src/
│   ├── core/             # Core algorithm implementation
│   │   ├── Estimator.h/cpp                  # IEKF-based LIO estimator
│   │   ├── State.h/cpp                      # 18-dim state representation
│   │   ├── VoxelMap.h/cpp                   # Hash-based voxel map for fast KNN
│   │   └── ProbabilisticKernelOptimizer.h/cpp # PKO for adaptive robust estimation
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
│   ├── avia.yaml         # Parameters for Livox Avia LiDAR
│   └── mid360.yaml       # Parameters for Livox Mid-360 LiDAR
│
├── thirdparty/           # Third-party libraries
│   ├── pangolin/         # 3D visualization
│   └── spdlog/           # Logging (header-only)
│
├── CMakeLists.txt        # CMake build configuration
└── README.md             # This file
```



