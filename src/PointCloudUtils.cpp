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
#include <iostream>

namespace lio {

// ===== Utility Functions =====

PointCloud::Ptr LoadKittiBinary(const std::string& filename) {
    auto cloud = std::make_shared<PointCloud>();
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open KITTI binary file: " << filename << std::endl;
        return cloud;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Each point is 4 floats (x, y, z, intensity)
    size_t num_points = file_size / (4 * sizeof(float));
    cloud->reserve(num_points);
    
    // Read points
    for (size_t i = 0; i < num_points; ++i) {
        float x, y, z, intensity;
        file.read(reinterpret_cast<char*>(&x), sizeof(float));
        file.read(reinterpret_cast<char*>(&y), sizeof(float));
        file.read(reinterpret_cast<char*>(&z), sizeof(float));
        file.read(reinterpret_cast<char*>(&intensity), sizeof(float));
        
        cloud->push_back(x, y, z);
    }
    
    file.close();
    std::cout << "Loaded " << cloud->size() << " points from " << filename << std::endl;
    return cloud;
}

bool SaveKittiBinary(const PointCloud::ConstPtr& cloud, const std::string& filename) {
    if (!cloud || cloud->empty()) {
        std::cerr << "Cannot save empty point cloud" << std::endl;
        return false;
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write points with intensity = 0
    for (const auto& point : *cloud) {
        float x = point.x;
        float y = point.y;
        float z = point.z;
        float intensity = 0.0f;
        
        file.write(reinterpret_cast<const char*>(&x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&z), sizeof(float));
        file.write(reinterpret_cast<const char*>(&intensity), sizeof(float));
    }
    
    file.close();
    std::cout << "Saved " << cloud->size() << " points to " << filename << std::endl;
    return true;
}

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
        std::cerr << "Cannot save empty point cloud" << std::endl;
        return false;
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open PLY file for writing: " << filename << std::endl;
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
    std::cout << "Saved " << cloud->size() << " points to PLY: " << filename << std::endl;
    return true;
}

// ===== KdTree Implementation =====

void KdTree::SetInputCloud(const PointCloud::ConstPtr& cloud) {
    m_cloud = cloud;
    // Note: This is a simplified implementation
    // For production use, implement proper kd-tree or use nanoflann
}

int KdTree::NearestKSearch(const Point3D& query_point, int k, 
                          std::vector<int>& indices, std::vector<float>& distances) const {
    if (!m_cloud || m_cloud->empty() || k <= 0) {
        indices.clear();
        distances.clear();
        return 0;
    }
    
    // Simple brute force search (replace with proper kd-tree in production)
    std::vector<std::pair<float, int>> point_distances;
    point_distances.reserve(m_cloud->size());
    
    for (size_t i = 0; i < m_cloud->size(); ++i) {
        float dist = query_point.squared_distance_to(m_cloud->at(i));
        point_distances.emplace_back(dist, static_cast<int>(i));
    }
    
    // Sort by distance
    std::partial_sort(point_distances.begin(), 
                     point_distances.begin() + std::min(k, static_cast<int>(point_distances.size())),
                     point_distances.end());
    
    // Extract results
    int actual_k = std::min(k, static_cast<int>(point_distances.size()));
    indices.resize(actual_k);
    distances.resize(actual_k);
    
    for (int i = 0; i < actual_k; ++i) {
        distances[i] = std::sqrt(point_distances[i].first);
        indices[i] = point_distances[i].second;
    }
    
    return actual_k;
}

int KdTree::RadiusSearch(const Point3D& query_point, float radius,
                        std::vector<int>& indices, std::vector<float>& distances) const {
    if (!m_cloud || m_cloud->empty() || radius <= 0) {
        indices.clear();
        distances.clear();
        return 0;
    }
    
    indices.clear();
    distances.clear();
    
    float radius_squared = radius * radius;
    
    for (size_t i = 0; i < m_cloud->size(); ++i) {
        float dist_squared = query_point.squared_distance_to(m_cloud->at(i));
        if (dist_squared <= radius_squared) {
            distances.push_back(std::sqrt(dist_squared));
            indices.push_back(static_cast<int>(i));
        }
    }
    
    return static_cast<int>(indices.size());
}

// ===== VoxelGrid Implementation =====

void VoxelGrid::Filter(PointCloud& output) {
    if (!m_input_cloud || m_input_cloud->empty() || m_leaf_size <= 0) {
        output.clear();
        return;
    }
    
    // Use map to store weighted centroids for each voxel
    std::map<VoxelKey, WeightedCentroid> voxel_map;
    
    // Process points one by one with weighted averaging
    for (size_t i = 0; i < m_input_cloud->size(); ++i) {
        const Point3D& point = m_input_cloud->at(i);
        VoxelKey voxel_key = GetVoxelKey(point);
        
        auto& weighted_centroid = voxel_map[voxel_key];
        weighted_centroid.AddPoint(point);
    }
    
    output.clear();
    output.reserve(voxel_map.size());
    
    // Extract final centroids from each voxel
    for (const auto& voxel : voxel_map) {
        output.push_back(voxel.second.GetCentroid());
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

} // namespace lio