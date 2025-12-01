# Lidar-Inertial Odometry

### MIT License


## Demo
[![LIO Demo](https://img.youtube.com/vi/difotKwX6yo/0.jpg)](https://www.youtube.com/watch?v=difotKwX6yo)

## Features

- **Iterated Extended Kalman Filter (IEKF)**: Direct LiDAR-IMU fusion with nested iteration for re-linearization and convergence
- **Adaptive Robust Estimation**: Probabilistic Kernel Optimization (PKO) for automatic Huber loss scale tuning
- **2-Level Hierarchical Voxel Map (L1→L0)**: Coarse-to-fine spatial indexing with Z-order Morton code hashing for cache-friendly O(1) lookup
- **Pre-computed Surfel Planes**: L1 voxels store fitted plane surfels (normal, centroid, planarity) via incremental covariance, enabling fast point-to-plane correspondence
- **Motion Compensation**: IMU-based undistortion for moving LiDAR scans

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


### ROS2 Wrapper: https://github.com/93won/lio_ros_wrapper

### Installation (Ubuntu 20.04)

```bash
cd lidar_inertial_odometry
./build.sh
```

This will:
1. Build Pangolin from `thirdparty/pangolin`
2. Build the main project with CMake

### Quick Start

#### NTU VIRAL Dataset

**Download Pre-processed Dataset**:
- **Google Drive**: [NTU VIRAL Parsed Dataset](https://drive.google.com/drive/folders/1FMQRJge70qzWWRuTpiXJJMa5MDoF7u4z?usp=sharing)
- **Source**: [NTU VIRAL Dataset](https://ntu-aris.github.io/ntu_viral_dataset/)
- **Sensors**: Ouster OS1-16 LiDAR + VectorNav VN100 IMU

**Running Single Sequence**:
```bash
cd build
./lio_player ../config/ntu_viral.yaml /path/to/NTU_VIRAL/eee_01
```

#### M3DGR Dataset

**Download Pre-processed Dataset**:
- **Google Drive**: [M3DGR Parsed Dataset](https://drive.google.com/drive/folders/1zOmvw3sCwRQ0LHo1b-jhY21L693GmOfW?usp=sharing)
- **Source**: [M3DGR Dataset](https://github.com/sjtuyinjie/M3DGR)
- **Sensors**: Livox Avia / Mid-360 LiDAR + Built-in IMU

**Running Single Sequence**:
```bash
cd build

# Livox Avia
./lio_player ../config/avia.yaml /path/to/M3DGR/Dynamic03/avia

# Livox Mid-360
./lio_player ../config/mid360.yaml /path/to/M3DGR/Dynamic03/mid360
```

**Dataset Structure**:
```
M3DGR/
├── Dynamic03/
│   ├── avia/
│   │   ├── imu_data.csv
│   │   ├── lidar_timestamps.txt
│   │   └── lidar/
│   │       ├── 0000000000.pcd
│   │       ├── 0000000001.pcd
│   │       └── ...
│   └── mid360/
│       └── (same structure)
├── Dynamic04/
├── Occlusion03/
├── Occlusion04/
├── Outdoor01/
└── Outdoor04/
```


## Benchmark

Evaluation on [M3DGR Dataset](https://github.com/sjtuyinjie/M3DGR) comparing with [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) and [FASTER-LIO](https://github.com/gaoxiang12/faster-lio).

### Summary

| Sensor | Metric | **Ours** | FAST-LIO2 | FASTER-LIO |
|--------|--------|----------|-----------|------------|
| **Livox AVIA** | APE RMSE (m) | 0.365 | 0.397 | 0.362 |
| **Livox AVIA** | FPS | **531** | 125 | 184 |
| **Livox Mid360** | APE RMSE (m) | 0.342 | 0.342 | 0.352 |
| **Livox Mid360** | FPS | **690** | 282 | 353 |



### Detailed Results (Livox AVIA)

| Sequence | Ours (m) | FAST-LIO2 (m) | FASTER-LIO (m) | Ours (FPS) | FL2 (FPS) | FL (FPS) |
|----------|----------|---------------|----------------|------------|-----------|----------|
| Dark01 | 0.118 | 0.258 | 0.223 | 670 | 140 | 203 |
| Dark02 | 0.692 | 0.729 | 0.645 | 488 | 120 | 195 |
| Dynamic03 | 0.266 | 0.165 | 0.151 | 606 | 145 | 225 |
| Dynamic04 | 0.392 | 0.279 | 0.261 | 603 | 138 | 201 |
| Occlusion03 | 0.271 | 0.257 | 0.283 | 561 | 124 | 179 |
| Occlusion04 | 0.295 | 0.479 | 0.337 | 501 | 130 | 203 |
| Varying-illu03 | 0.961 | 0.897 | 1.032 | 436 | 120 | 165 |
| Varying-illu04 | 0.125 | 0.102 | 0.119 | 435 | 93 | 144 |
| Varying-illu05 | 0.167 | 0.402 | 0.207 | 576 | 133 | 172 |
| **Average** | **0.365** | 0.397 | 0.362 | **531** | 125 | 184 |

### Detailed Results (Livox Mid360)

| Sequence | Ours (m) | FAST-LIO2 (m) | FASTER-LIO (m) | Ours (FPS) | FL2 (FPS) | FL (FPS) |
|----------|----------|---------------|----------------|------------|-----------|----------|
| Dark01 | 0.185 | 0.177 | 0.173 | 1044 | 589 | 925 |
| Dark02 | 0.310 | 0.239 | 0.212 | 720 | 337 | 401 |
| Dynamic03 | 0.206 | 0.178 | 0.178 | 670 | 263 | 461 |
| Dynamic04 | 0.246 | 0.216 | 0.214 | 657 | 265 | 363 |
| Occlusion03 | 0.315 | 0.423 | 0.463 | 687 | 301 | 373 |
| Occlusion04 | 0.345 | 0.216 | 0.284 | 596 | 251 | 385 |
| Varying-illu03 | 0.957 | 1.221 | 1.189 | 618 | 242 | 259 |
| Varying-illu04 | 0.206 | 0.161 | 0.163 | 664 | 205 | 170 |
| Varying-illu05 | 0.307 | 0.245 | 0.290 | 704 | 297 | 498 |
| **Average** | **0.342** | 0.342 | 0.352 | **690** | 282 | 353 |


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


