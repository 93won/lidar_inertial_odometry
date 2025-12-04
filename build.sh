#!/bin/bash

set -e  # Exit on any error

echo "=============================================="
echo "    LiDAR-Inertial Odometry Build Script"
echo "=============================================="

# Get number of CPU cores for parallel compilation (use half of available cores)
NPROC=$(($(nproc) / 2))
if [ $NPROC -lt 1 ]; then
    NPROC=1
fi
echo "Using $NPROC cores for compilation (half of available)"

# Install system dependencies
echo ""
echo "Step 0: Installing system dependencies..."
echo "========================================"

# Check if running in Docker container
if [ -n "$DOCKER_CONTAINER" ]; then
    echo "Running in Docker container - skipping apt updates (dependencies pre-installed)"
else
    sudo apt update || true  # Continue even if apt update fails
    sudo apt install -y \
        cmake \
        build-essential \
        libeigen3-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libglew-dev \
        libyaml-cpp-dev \
        libopencv-dev

    echo "System dependencies installed successfully!"
fi

# Build third-party dependencies
echo ""
echo "Step 1: Building third-party dependencies..."
echo "=============================================="

# Check if thirdparty directory exists
if [ ! -d "thirdparty" ]; then
    echo "Error: thirdparty directory not found!"
    echo "Please ensure the thirdparty directory with dependencies exists."
    exit 1
fi

# Build Pangolin
echo "Building Pangolin..."
if [ ! -d "thirdparty/pangolin/build" ]; then
    mkdir -p thirdparty/pangolin/build
fi
cd thirdparty/pangolin/build
cmake .. \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TOOLS=OFF \
    -DBUILD_PYPANGOLIN=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF
make -j$NPROC
cd ../../..

# Header-only libraries (no build needed)
echo "spdlog: header-only library (no build needed)"

# Build main project
echo ""
echo "Step 2: Building main project..."
echo "================================="

# Create build directory for main project
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure and build main project
cmake ..
make -j$NPROC

echo ""
echo "=============================================="
echo "  Build completed successfully!"
echo "=============================================="
echo ""
echo "To run the LiDAR-Inertial Odometry system:"
echo "  cd build"
echo ""
echo "For R3LIVE dataset:"
echo "  ./lio_player /path/to/dataset"
echo ""
echo "Example:"
echo "  ./lio_player /home/eugene/data/R3LIVE/hku_main_building"
echo ""
