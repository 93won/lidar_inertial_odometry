# Quick Start

[Back to README](../README.md)

## Installation (Ubuntu 20.04)

```bash
cd lidar_inertial_odometry
CMAKE_BUILD_PARALLEL_LEVEL=8 MAKEFLAGS=-j8 taskset -c 0-7 ./build.sh
```

This will:

1. Build Pangolin from `thirdparty/pangolin`
2. Build the main project with CMake

## Datasets

### NTU VIRAL Dataset

**Download Pre-processed Dataset**:

- **Google Drive**: [NTU VIRAL Parsed Dataset](https://drive.google.com/drive/folders/1FMQRJge70qzWWRuTpiXJJMa5MDoF7u4z?usp=sharing)
- **Source**: [NTU VIRAL Dataset](https://ntu-aris.github.io/ntu_viral_dataset/)
- **Sensors**: Ouster OS1-16 LiDAR + VectorNav VN100 IMU

**Running Single Sequence**:

```bash
cd build
./lio_player ../config/ntu_viral.yaml /path/to/NTU_VIRAL/eee_01
```

### M3DGR Dataset

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

## ROS 2

The [ROS 2 wrapper](https://github.com/93won/lio_ros_wrapper) is a separate package. Its current revision is not compatible with the v0.1.0 core without updates; the standalone instructions above do not validate ROS integration.
