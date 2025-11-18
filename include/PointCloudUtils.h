/**
 * @file      PointCloudUtils.h
 * @brief     Native C++ point cloud utilities for LiDAR-Inertial Odometry
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef POINT_CLOUD_UTILS_H
#define POINT_CLOUD_UTILS_H

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <map>
#include <Eigen/Dense>

namespace lio {

// ===== Point Types =====

/**
 * @brief Basic 3D point structure
 */
struct Point3D {
    float x, y, z;
    float intensity;           // LiDAR intensity or offset_time (in seconds)
    float offset_time;
    
    Point3D() : x(0.0f), y(0.0f), z(0.0f), intensity(0.0f), offset_time(0) {}
    Point3D(float x_, float y_, float z_) 
        : x(x_), y(y_), z(z_), intensity(0.0f), offset_time(0) {}
    Point3D(float x_, float y_, float z_, float intensity_) 
        : x(x_), y(y_), z(z_), intensity(intensity_), offset_time(0) {}
    Point3D(float x_, float y_, float z_, float intensity_, float offset_time_) 
        : x(x_), y(y_), z(z_), intensity(intensity_), offset_time(offset_time_) {}
    
    // Vector operations
    Point3D operator+(const Point3D& other) const {
        return Point3D(x + other.x, y + other.y, z + other.z);
    }
    
    Point3D operator-(const Point3D& other) const {
        return Point3D(x - other.x, y - other.y, z - other.z);
    }
    
    Point3D operator*(float scalar) const {
        return Point3D(x * scalar, y * scalar, z * scalar);
    }
    
    // Distance calculations
    float distance_to(const Point3D& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    
    float squared_distance_to(const Point3D& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        return dx*dx + dy*dy + dz*dz;
    }
    
    // Norm calculations
    float norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    float squared_norm() const {
        return x*x + y*y + z*z;
    }
    
    // Conversion to/from Eigen
    Eigen::Vector3f to_eigen() const {
        return Eigen::Vector3f(x, y, z);
    }
    
    static Point3D from_eigen(const Eigen::Vector3f& vec) {
        return Point3D(vec.x(), vec.y(), vec.z());
    }
    
    static Point3D from_eigen(const Eigen::Vector3d& vec) {
        return Point3D(static_cast<float>(vec.x()), 
                      static_cast<float>(vec.y()), 
                      static_cast<float>(vec.z()));
    }
};

/**
 * @brief Point cloud container class
 */
class PointCloud {
public:
    using Point = Point3D;
    using Ptr = std::shared_ptr<PointCloud>;
    using ConstPtr = std::shared_ptr<const PointCloud>;
    
    // ===== Constructors =====
    PointCloud() = default;
    explicit PointCloud(size_t reserve_size) {
        m_points.reserve(reserve_size);
    }
    
    // ===== Basic Operations =====
    
    /**
     * @brief Add a point to the cloud
     */
    void push_back(const Point3D& point) {
        m_points.push_back(point);
    }
    
    /**
     * @brief Add a point to the cloud
     */
    void push_back(float x, float y, float z) {
        m_points.emplace_back(x, y, z);
    }
    
    /**
     * @brief Get point by index
     */
    const Point3D& operator[](size_t index) const {
        return m_points[index];
    }
    
    Point3D& operator[](size_t index) {
        return m_points[index];
    }
    
    /**
     * @brief Get point by index with bounds checking
     */
    const Point3D& at(size_t index) const {
        return m_points.at(index);
    }
    
    Point3D& at(size_t index) {
        return m_points.at(index);
    }
    
    /**
     * @brief Get number of points
     */
    size_t size() const {
        return m_points.size();
    }
    
    /**
     * @brief Check if cloud is empty
     */
    bool empty() const {
        return m_points.empty();
    }
    
    /**
     * @brief Clear all points
     */
    void clear() {
        m_points.clear();
    }
    
    /**
     * @brief Reserve memory for points
     */
    void reserve(size_t size) {
        m_points.reserve(size);
    }
    
    /**
     * @brief Resize the point cloud
     */
    void resize(size_t size) {
        m_points.resize(size);
    }
    
    // ===== Utility Methods =====
    
    /**
     * @brief Get bounding box of the point cloud
     */
    struct BoundingBox {
        Point3D min_point, max_point;
        bool is_valid = false;
    };
    
    BoundingBox GetBoundingBox() const {
        BoundingBox bbox;
        if (m_points.empty()) {
            return bbox;
        }
        
        bbox.min_point = bbox.max_point = m_points[0];
        bbox.is_valid = true;
        
        for (const auto& point : m_points) {
            bbox.min_point.x = std::min(bbox.min_point.x, point.x);
            bbox.min_point.y = std::min(bbox.min_point.y, point.y);
            bbox.min_point.z = std::min(bbox.min_point.z, point.z);
            
            bbox.max_point.x = std::max(bbox.max_point.x, point.x);
            bbox.max_point.y = std::max(bbox.max_point.y, point.y);
            bbox.max_point.z = std::max(bbox.max_point.z, point.z);
        }
        
        return bbox;
    }
    
    /**
     * @brief Get centroid of the point cloud
     */
    Point3D GetCentroid() const {
        if (m_points.empty()) {
            return Point3D();
        }
        
        Point3D centroid;
        for (const auto& point : m_points) {
            centroid.x += point.x;
            centroid.y += point.y;
            centroid.z += point.z;
        }
        
        float inv_size = 1.0f / static_cast<float>(m_points.size());
        centroid.x *= inv_size;
        centroid.y *= inv_size;
        centroid.z *= inv_size;
        
        return centroid;
    }
    
    /**
     * @brief Transform all points by a 4x4 transformation matrix
     */
    void Transform(const Eigen::Matrix4f& transformation) {
        for (auto& point : m_points) {
            Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1.0f);
            Eigen::Vector4f transformed = transformation * homogeneous_point;
            
            point.x = transformed.x();
            point.y = transformed.y();
            point.z = transformed.z();
        }
    }
    
    void Transform(const Eigen::Matrix4d& transformation) {
        Eigen::Matrix4f transform_f = transformation.cast<float>();
        Transform(transform_f);
    }
    
    /**
     * @brief Create a transformed copy of the point cloud
     */
    PointCloud::Ptr TransformedCopy(const Eigen::Matrix4f& transformation) const {
        auto result = std::make_shared<PointCloud>();
        result->m_points.reserve(m_points.size());
        
        for (const auto& point : m_points) {
            Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1.0f);
            Eigen::Vector4f transformed = transformation * homogeneous_point;
            
            result->push_back(transformed.x(), transformed.y(), transformed.z());
        }
        
        return result;
    }
    
    /**
     * @brief Copy constructor helper
     */
    PointCloud::Ptr Copy() const {
        auto result = std::make_shared<PointCloud>();
        result->m_points = this->m_points;
        return result;
    }
    
    /**
     * @brief Append another point cloud to this one
     */
    PointCloud& operator+=(const PointCloud& other) {
        m_points.reserve(m_points.size() + other.m_points.size());
        for (const auto& point : other.m_points) {
            m_points.push_back(point);
        }
        return *this;
    }
    
    /**
     * @brief Iterator access
     */
    auto begin() const { return m_points.begin(); }
    auto end() const { return m_points.end(); }
    auto begin() { return m_points.begin(); }
    auto end() { return m_points.end(); }

private:
    std::vector<Point3D> m_points;
};

// ===== Type Aliases for Compatibility =====

using PointType = Point3D;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudConstPtr = PointCloud::ConstPtr;

// ===== Utility Functions =====

/**
 * @brief Load KITTI binary point cloud file
 * @param filename Path to .bin file
 * @return Pointer to loaded point cloud
 */
PointCloud::Ptr LoadKittiBinary(const std::string& filename);

/**
 * @brief Save point cloud to KITTI binary format
 * @param cloud Point cloud to save
 * @param filename Output file path
 * @return True if successful
 */
bool SaveKittiBinary(const PointCloud::ConstPtr& cloud, const std::string& filename);

/**
 * @brief Transform point cloud with transformation matrix
 * @param input Input point cloud
 * @param output Output point cloud
 * @param transformation 4x4 transformation matrix
 */
void TransformPointCloud(const PointCloud::ConstPtr& input,
                        PointCloud::Ptr& output,
                        const Eigen::Matrix4f& transformation);

/**
 * @brief Copy point cloud
 * @param input Input point cloud
 * @param output Output point cloud
 */
void CopyPointCloud(const PointCloud::ConstPtr& input, PointCloud::Ptr& output);

/**
 * @brief Save point cloud to PLY binary format
 * @param filename Output file path
 * @param cloud Point cloud to save
 * @return True if successful
 */
bool SavePointCloudPly(const std::string& filename, const PointCloud::ConstPtr& cloud);

// ===== Spatial Data Structures and Filters =====

/**
 * @brief Simple KD-Tree for nearest neighbor search
 */
class KdTree {
public:
    KdTree() = default;
    ~KdTree() = default;
    
    /**
     * @brief Set input point cloud and build tree
     */
    void SetInputCloud(const PointCloud::ConstPtr& cloud);
    
    /**
     * @brief Find k nearest neighbors
     */
    int NearestKSearch(const Point3D& query_point, int k, 
                      std::vector<int>& indices, std::vector<float>& distances) const;
    
    /**
     * @brief Find neighbors within radius
     */
    int RadiusSearch(const Point3D& query_point, float radius,
                    std::vector<int>& indices, std::vector<float>& distances) const;

private:
    PointCloud::ConstPtr m_cloud;
    // Simple implementation - for production use nanoflann or similar
};

/**
 * @brief Voxel grid downsampling filter
 */
class VoxelGrid {
public:
    VoxelGrid() : m_leaf_size(0.01f) {}
    
    void SetLeafSize(float size) { m_leaf_size = size; }
    
    void SetInputCloud(const PointCloud::ConstPtr& cloud) { m_input_cloud = cloud; }
    
    void Filter(PointCloud& output);
    
private:
    struct WeightedCentroid {
        Point3D centroid;
        float weight;
        
        WeightedCentroid() : centroid(0, 0, 0), weight(0.0f) {}
        
        void AddPoint(const Point3D& new_point) {
            if (weight == 0.0f) {
                centroid = new_point;
                weight = 1.0f;
            } else {
                float total_weight = weight + 1.0f;
                float old_ratio = weight / total_weight;
                float new_ratio = 1.0f / total_weight;
                
                centroid.x = old_ratio * centroid.x + new_ratio * new_point.x;
                centroid.y = old_ratio * centroid.y + new_ratio * new_point.y;
                centroid.z = old_ratio * centroid.z + new_ratio * new_point.z;
                
                weight = total_weight;
            }
        }
        
        Point3D GetCentroid() const { return centroid; }
    };
    
    struct VoxelKey {
        int x, y, z;
        
        VoxelKey(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
        
        bool operator<(const VoxelKey& other) const {
            if (x != other.x) return x < other.x;
            if (y != other.y) return y < other.y;
            return z < other.z;
        }
    };
    
    VoxelKey GetVoxelKey(const Point3D& point) const {
        int vx = static_cast<int>(std::floor(point.x / m_leaf_size));
        int vy = static_cast<int>(std::floor(point.y / m_leaf_size));
        int vz = static_cast<int>(std::floor(point.z / m_leaf_size));
        return VoxelKey(vx, vy, vz);
    }
    
    float m_leaf_size;
    PointCloud::ConstPtr m_input_cloud;
};

/**
 * @brief Range filter for distance-based filtering
 */
class RangeFilter {
public:
    RangeFilter() : m_min_range(0.0f), m_max_range(100.0f) {}
    
    void SetRadiusLimits(float min_radius, float max_radius) {
        m_min_range = min_radius;
        m_max_range = max_radius;
    }
    
    void SetInputCloud(const PointCloud::ConstPtr& cloud) { m_input_cloud = cloud; }
    
    void Filter(PointCloud& output);
    
private:
    float m_min_range, m_max_range;
    PointCloud::ConstPtr m_input_cloud;
};

} // namespace lio

#endif // POINT_CLOUD_UTILS_H