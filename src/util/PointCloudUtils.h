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
#include "unordered_dense.h"  // Fast hash map

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
 * @brief Save point cloud to PLY format
 * @param filename Output file path
 * @param cloud Point cloud to save
 * @return True if successful
 */
bool SavePointCloudPly(const std::string& filename, const PointCloud::ConstPtr& cloud);

// ===== Spatial Data Structures and Filters =====

/**
 * @brief Voxel grid downsampling filter with planarity-based filtering
 * Uses relaxed planarity thresholds compared to VoxelMap surfel creation
 */
class VoxelGrid {
public:
    VoxelGrid() 
        : m_leaf_size(0.4f)
        , m_inv_leaf_size(1.0f / 0.4f)  // Pre-computed inverse for fast division
        , m_planarity_threshold(0.1f)  // Default: relaxed threshold for downsampling
                                        // Can be overridden via SetPlanarityThreshold()
                                        // Config file: scan_planarity_threshold (0.1f)
        , m_enable_planarity_filter(false)
    {}
    
    void SetLeafSize(float size) { 
        m_leaf_size = size; 
        m_inv_leaf_size = 1.0f / size;  // Update inverse when leaf size changes
    }
    
    void SetInputCloud(const PointCloud::ConstPtr& cloud) { m_input_cloud = cloud; }
    
    /**
     * @brief Enable/disable planarity-based filtering during downsampling
     * @param enable If true, only planar points are output
     */
    void SetPlanarityFilter(bool enable) { m_enable_planarity_filter = enable; }
    
    /**
     * @brief Set planarity threshold for filtering
     * @param threshold Planarity score threshold (sigma_min / sigma_max)
     *                  Points with planarity < threshold are kept (more planar)
     *                  Default: 0.1f (relaxed for input scan filtering)
     *                  VoxelMap surfel uses: 0.01f (strict for map quality)
     */
    void SetPlanarityThreshold(float threshold) { m_planarity_threshold = threshold; }
    
    /**
     * @brief Set hierarchy factor for L1 voxel size
     * @param factor L1 voxel size = L0 size × factor (e.g., 5 means 5×5×5 L0 voxels per L1)
     */
    void SetHierarchyFactor(int factor) { m_hierarchy_factor = factor; }
    
    void Filter(PointCloud& output);
    
private:
    struct VoxelPoints {
        std::vector<Point3D> points;
        
        void AddPoint(const Point3D& new_point) {
            points.push_back(new_point);
        }
        
        size_t GetPointCount() const {
            return points.size();
        }
        
        float CalculatePlanarity() const {
            if (points.size() < 3) {
                return 0.0f;  // Not enough points for planarity check
            }
            
            // Compute centroid
            Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
            for (const auto& pt : points) {
                centroid += Eigen::Vector3f(pt.x, pt.y, pt.z);
            }
            centroid /= static_cast<float>(points.size());
            
            // Compute covariance matrix
            Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
            for (const auto& pt : points) {
                Eigen::Vector3f diff = Eigen::Vector3f(pt.x, pt.y, pt.z) - centroid;
                covariance += diff * diff.transpose();
            }
            covariance /= static_cast<float>(points.size());
            
            // SVD decomposition
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Vector3f singular_values = svd.singularValues();
            
            // Planarity: sigma_min / sigma_max
            // Lower values = more planar
            float planarity = singular_values(2) / (singular_values(0) + 1e-6f);
            return planarity;
        }
        
        Point3D GetCentroid() const {
            if (points.empty()) {
                return Point3D(0, 0, 0);
            }
            
            float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
            float sum_intensity = 0.0f;
            float sum_offset_time = 0.0f;
            
            for (const auto& pt : points) {
                sum_x += pt.x;
                sum_y += pt.y;
                sum_z += pt.z;
                sum_intensity += pt.intensity;
                sum_offset_time += pt.offset_time;
            }
            
            float n = static_cast<float>(points.size());
            Point3D centroid;
            centroid.x = sum_x / n;
            centroid.y = sum_y / n;
            centroid.z = sum_z / n;
            centroid.intensity = sum_intensity / n;
            centroid.offset_time = sum_offset_time / n;  // Average offset_time for undistortion
            
            return centroid;
        }
    };
    
    struct VoxelKey {
        int x, y, z;
        
        VoxelKey(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
        
        bool operator<(const VoxelKey& other) const {
            if (x != other.x) return x < other.x;
            if (y != other.y) return y < other.y;
            return z < other.z;
        }
        
        bool operator==(const VoxelKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    
    // Z-order (Morton code) hash for VoxelKey - preserves spatial locality
    struct VoxelKeyHash {
    private:
        static inline uint64_t ExpandBits(int32_t v) {
            uint64_t x = static_cast<uint64_t>(v + (1 << 20)) & 0x1fffff;
            x = (x | (x << 32)) & 0x1f00000000ffffULL;
            x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
            x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
            x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
            x = (x | (x << 2))  & 0x1249249249249249ULL;
            return x;
        }
    public:
        std::size_t operator()(const VoxelKey& key) const {
            uint64_t morton = ExpandBits(key.x) | (ExpandBits(key.y) << 1) | (ExpandBits(key.z) << 2);
            return static_cast<std::size_t>(morton);
        }
    };
    
    VoxelKey GetVoxelKey(const Point3D& point) const {
        // Fast floor using pre-computed inverse (multiplication instead of division)
        // For negative coords: int(-0.5) = 0, but floor(-0.5) = -1, so we need adjustment
        int vx = (point.x >= 0) ? static_cast<int>(point.x * m_inv_leaf_size) 
                                : static_cast<int>(point.x * m_inv_leaf_size) - 1;
        int vy = (point.y >= 0) ? static_cast<int>(point.y * m_inv_leaf_size) 
                                : static_cast<int>(point.y * m_inv_leaf_size) - 1;
        int vz = (point.z >= 0) ? static_cast<int>(point.z * m_inv_leaf_size) 
                                : static_cast<int>(point.z * m_inv_leaf_size) - 1;
        return VoxelKey(vx, vy, vz);
    }
    
    float m_leaf_size;
    float m_inv_leaf_size;  // Pre-computed 1.0f / m_leaf_size
    float m_planarity_threshold;
    bool m_enable_planarity_filter;
    int m_hierarchy_factor = 5;  // L1 = factor × L0
    PointCloud::ConstPtr m_input_cloud;
};

/**
 * @brief Frustum culling filter for sensor FOV-based filtering
 * Filters points based on current sensor pose and field of view
 */
class FrustumFilter {
public:
    FrustumFilter() 
        : m_horizontal_fov(90.0f)  // degrees, total FOV
        , m_vertical_fov(90.0f)    // degrees, total FOV
        , m_max_range(30.0f)       // meters
        , m_R_sw(Eigen::Matrix3f::Identity())
        , m_t_sw(Eigen::Vector3f::Zero())
    {}
    
    /**
     * @brief Set sensor pose (world frame to sensor frame transformation)
     * @param R_sw Rotation matrix from world to sensor  
     * @param t_sw Translation vector from world to sensor
     * 
     * This computes: p_sensor = R_sw * p_world + t_sw
     */
    void SetSensorPose(const Eigen::Matrix3f& R_sw, const Eigen::Vector3f& t_sw) {
        m_R_sw = R_sw;
        m_t_sw = t_sw;
    }
    
    /**
     * @brief Set field of view
     * @param horizontal_fov Horizontal FOV in degrees (total, not half-angle)
     * @param vertical_fov Vertical FOV in degrees (total, not half-angle)
     */
    void SetFOV(float horizontal_fov, float vertical_fov) {
        m_horizontal_fov = horizontal_fov;
        m_vertical_fov = vertical_fov;
    }
    
    /**
     * @brief Set maximum range
     * @param max_range Maximum distance in meters
     */
    void SetMaxRange(float max_range) {
        m_max_range = max_range;
    }
    
    void SetInputCloud(const PointCloud::ConstPtr& cloud) { m_input_cloud = cloud; }
    
    /**
     * @brief Filter points within sensor frustum
     * Transforms points to sensor frame and checks FOV/range constraints
     */
    void Filter(PointCloud& output);
    
private:
    float m_horizontal_fov;  // Total horizontal FOV in degrees
    float m_vertical_fov;    // Total vertical FOV in degrees
    float m_max_range;       // Maximum range in meters
    
    Eigen::Matrix3f m_R_sw;  // World to sensor rotation
    Eigen::Vector3f m_t_sw;  // World to sensor translation
    
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