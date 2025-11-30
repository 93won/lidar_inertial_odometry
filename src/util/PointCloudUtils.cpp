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
    
    // Direct L0 voxel hashing with centroid calculation
    ankerl::unordered_dense::map<VoxelKey, VoxelPoints, VoxelKeyHash> voxel_map;
    voxel_map.reserve(10);  // Pre-allocate for ~3 pts per voxel
    
    for (size_t i = 0; i < m_input_cloud->size(); ++i) {
        const Point3D& point = m_input_cloud->at(i);
        VoxelKey key = GetVoxelKey(point);
        voxel_map[key].AddPoint(point);
    }
    
    // Extract centroids (with optional planarity filtering)
    output.clear();
    output.reserve(voxel_map.size());
    
    // int total_voxels = 0;
    // int checked_voxels = 0;  // voxels with >= 3 points
    // int passed_voxels = 0;   // passed planarity check
    // int skipped_voxels = 0;  // failed planarity check
    
    for (auto& voxel : voxel_map) {
        // total_voxels++;
        
        // // If planarity filter enabled, check planarity score (need at least 3 points)
        // if (m_enable_planarity_filter && voxel.second.GetPointCount() >= 3) {
        //     checked_voxels++;
        //     float planarity = voxel.second.CalculatePlanarity();
        //     // Keep only planar voxels (low planarity score = more planar)
        //     if (planarity > 0.01f) {
        //         skipped_voxels++;
        //         continue;  // Skip non-planar voxels
        //     }
        //     passed_voxels++;
        // }
        output.push_back(voxel.second.GetCentroid());
    }
    
    // if (m_enable_planarity_filter) {
    //     spdlog::info("[VoxelGrid] Total: {}, Checked(>=3pts): {}, Passed: {}, Skipped: {}, Output: {}",
    //                  total_voxels, checked_voxels, passed_voxels, skipped_voxels, output.size());
    // }
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