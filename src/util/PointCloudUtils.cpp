/**
 * @file      PointCloudUtils.cpp
 * @brief     Implementation of point cloud utilities for LiDAR-Inertial Odometry
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "PointCloudUtils.h"
#include <spdlog/spdlog.h>
#include <chrono>

namespace lio {

// ===== Utility Functions =====

void TransformPointCloud(const PointCloud::ConstPtr& input,
                        PointCloud::Ptr& output,
                        const Eigen::Matrix4f& transformation) {
    if (!input) {
        output = std::make_shared<PointCloud>();
        return;
    }
    
    output = input->TransformedCopy(transformation);
}

void CopyPointCloud(const PointCloud::ConstPtr& input, PointCloud::Ptr& output) {
    if (!input) {
        output = std::make_shared<PointCloud>();
        return;
    }
    
    output = input->Copy();
}

bool SavePointCloudPly(const std::string& filename, const PointCloud::ConstPtr& cloud) {
    if (!cloud || cloud->empty()) {
        spdlog::error("Cannot save empty point cloud");
        return false;
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        spdlog::error("Failed to open PLY file for writing: {}", filename);
        return false;
    }
    
    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << cloud->size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "end_header\n";
    
    // Write points
    for (const auto& point : *cloud) {
        file << point.x << " " << point.y << " " << point.z << "\n";
    }
    
    file.close();
    spdlog::info("Saved {} points to PLY: {}", cloud->size(), filename);
    return true;
}

// ===== VoxelGrid Implementation =====

void VoxelGrid::Filter(PointCloud& output) {
    if (!m_input_cloud || m_input_cloud->empty() || m_leaf_size <= 0) {
        output.clear();
        return;
    }
    
    // L2 voxel size for noise filtering (5x5x5 L0 voxels per L2)
    // Note: This is only for downsampling, not actual VoxelMap structure
    const float l2_size = m_leaf_size * 5;
    
    // Step 1: Group points into L2 voxels first (using fast hash map)
    ankerl::unordered_dense::map<VoxelKey, std::vector<Point3D>, VoxelKeyHash> l2_voxel_map;
    
    for (size_t i = 0; i < m_input_cloud->size(); ++i) {
        const Point3D& point = m_input_cloud->at(i);
        int l2_x = static_cast<int>(std::floor(point.x / l2_size));
        int l2_y = static_cast<int>(std::floor(point.y / l2_size));
        int l2_z = static_cast<int>(std::floor(point.z / l2_size));
        VoxelKey l2_key(l2_x, l2_y, l2_z);
        l2_voxel_map[l2_key].push_back(point);
    }
    
    // Step 2: For each L2 with >1 point, subdivide into L0 voxels
    ankerl::unordered_dense::map<VoxelKey, VoxelPoints, VoxelKeyHash> l0_voxel_map;
    
    for (auto& l2_voxel : l2_voxel_map) {
        // Skip isolated L2 voxels (only 1 point = noise)
        if (l2_voxel.second.size() == 0 ) {
            continue;
        }
        
        // Subdivide into L0 voxels
        for (const Point3D& point : l2_voxel.second) {
            VoxelKey l0_key = GetVoxelKey(point);
            l0_voxel_map[l0_key].AddPoint(point);
        }
    }
    
    output.clear();
    output.reserve(l0_voxel_map.size());
    
    // Step 3: Process each L0 voxel
    for (auto& voxel : l0_voxel_map) {
        VoxelPoints& voxel_points = voxel.second;

                
     
        // Extract and add centroid to output
        output.push_back(voxel_points.GetCentroid());
    }
}

// ===== RangeFilter Implementation =====

void RangeFilter::Filter(PointCloud& output) {
    if (!m_input_cloud) {
        output.clear();
        return;
    }
    
    output.clear();
    output.reserve(m_input_cloud->size());
    
    for (const Point3D& point : *m_input_cloud) {
        float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        if (range >= m_min_range && range <= m_max_range) {
            output.push_back(point);
        }
    }
}

// ===== FrustumFilter Implementation =====

void FrustumFilter::Filter(PointCloud& output) {
    if (!m_input_cloud || m_input_cloud->empty()) {
        output.clear();
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    output.clear();
    output.reserve(m_input_cloud->size());
    
    // Convert FOV to half-angles in radians
    const float half_horizontal_fov_rad = (m_horizontal_fov * 0.5f) * M_PI / 180.0f;
    const float half_vertical_fov_rad = (m_vertical_fov * 0.5f) * M_PI / 180.0f;
    const float max_range_squared = m_max_range * m_max_range;
    
    // Pre-compute trigonometric values for FOV checks
    const float tan_half_h_fov = std::tan(half_horizontal_fov_rad);
    const float tan_half_v_fov = std::tan(half_vertical_fov_rad);
    
    int passed_range = 0;
    int passed_forward = 0;
    int passed_h_fov = 0;
    int passed_v_fov = 0;
    
    for (const Point3D& p_world : *m_input_cloud) {
        // Transform point from world to sensor frame
        // p_sensor = R_sw * p_world + t_sw
        Eigen::Vector3f p_w(p_world.x, p_world.y, p_world.z);
        Eigen::Vector3f p_sensor = m_R_sw * p_w + m_t_sw;
        
        // Check range first (cheap test)
        float range_squared = p_sensor.squaredNorm();
        if (range_squared > max_range_squared) {
            continue;
        }
        passed_range++;
        
        // For LiDAR, assume forward direction is +X in sensor frame
        // Check if point is in front of sensor
        if (p_sensor.x() <= 0.0f) {
            continue;
        }
        passed_forward++;
        
        // Check horizontal FOV: azimuth angle from atan2(y, x)
        // Point is within FOV if |y/x| <= tan(half_fov)
        float abs_y_over_x = std::abs(p_sensor.y() / p_sensor.x());
        if (abs_y_over_x > tan_half_h_fov) {
            continue;
        }
        passed_h_fov++;
        
        // Check vertical FOV: elevation angle from atan2(z, sqrt(x^2 + y^2))
        // Point is within FOV if |z / sqrt(x^2 + y^2)| <= tan(half_fov)
        float xy_norm = std::sqrt(p_sensor.x() * p_sensor.x() + p_sensor.y() * p_sensor.y());
        if (xy_norm > 1e-6f) {  // Avoid division by zero
            float abs_z_over_xy = std::abs(p_sensor.z() / xy_norm);
            if (abs_z_over_xy > tan_half_v_fov) {
                continue;
            }
        }
        passed_v_fov++;
        
        // Point passes all checks - add to output
        output.push_back(p_world);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
}

} // namespace lio