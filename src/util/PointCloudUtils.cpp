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
    
    // Use map to store all points in each voxel
    std::map<VoxelKey, VoxelPoints> voxel_map;
    
    // Process points one by one, accumulating into voxels
    for (size_t i = 0; i < m_input_cloud->size(); ++i) {
        const Point3D& point = m_input_cloud->at(i);
        VoxelKey voxel_key = GetVoxelKey(point);
        
        auto& voxel_points = voxel_map[voxel_key];
        voxel_points.AddPoint(point);
    }
    
    output.clear();
    output.reserve(voxel_map.size());
    
    size_t total_voxels = voxel_map.size();
    size_t planar_voxels = 0;
    float min_planarity = 1.0f;
    float max_planarity = 0.0f;
    float sum_planarity = 0.0f;
    
    // Process each voxel: compute planarity and filter
    for (auto& voxel : voxel_map) {
        VoxelPoints& voxel_points = voxel.second;
        
        // If planarity filtering is enabled, check planarity
        if (m_enable_planarity_filter) {
            float planarity = voxel_points.CalculatePlanarity();
            
            // Track statistics
            min_planarity = std::min(min_planarity, planarity);
            max_planarity = std::max(max_planarity, planarity);
            sum_planarity += planarity;
            
            // Only output centroid if planarity is below threshold (more planar)
            if (planarity > m_planarity_threshold) {
                // Non-planar voxel - skip it
                continue;
            }
            
            planar_voxels++;
        } else {
            planar_voxels++;
        }
        
        // Extract and add centroid to output
        output.push_back(voxel_points.GetCentroid());
    }
    
    // // Log statistics
    // if (m_enable_planarity_filter) {
    //     float avg_planarity = (total_voxels > 0) ? sum_planarity / total_voxels : 0.0f;
    //     float retention_rate = (total_voxels > 0) ? (100.0f * planar_voxels / total_voxels) : 0.0f;
        
    //     spdlog::info("[VoxelGrid] Downsampling with Planarity Filter:");
    //     spdlog::info("  Input points: {}", m_input_cloud->size());
    //     spdlog::info("  Total voxels (before filter): {}", total_voxels);
    //     spdlog::info("  Planar voxels (after filter): {} ({:.1f}%)", planar_voxels, retention_rate);
    //     spdlog::info("  Voxel planarity stats:");
    //     spdlog::info("    - Min planarity: {:.6f}", min_planarity);
    //     spdlog::info("    - Max planarity: {:.6f}", max_planarity);
    //     spdlog::info("    - Avg planarity: {:.6f}", avg_planarity);
    //     spdlog::info("    - Threshold: {:.6f}", m_planarity_threshold);
    //     spdlog::info("  Output points: {}", output.size());
    // } else {
    //     spdlog::info("[VoxelGrid] Downsampling (no planarity filter):");
    //     spdlog::info("  Input points: {}", m_input_cloud->size());
    //     spdlog::info("  Output voxels/points: {}", output.size());
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