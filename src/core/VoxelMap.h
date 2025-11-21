/**
 * @file      VoxelMap.h
 * @brief     Voxel-based hash map for efficient nearest neighbor search
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef VOXEL_MAP_H
#define VOXEL_MAP_H

#include "PointCloudUtils.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <Eigen/Dense>

namespace lio {

/**
 * @brief Voxel key for spatial hashing
 * Represents a 3D grid cell by integer indices (x, y, z)
 */
struct VoxelKey {
    int x, y, z;
    
    VoxelKey() : x(0), y(0), z(0) {}
    VoxelKey(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    
    // For debugging
    std::string ToString() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
    }
};

/**
 * @brief Hash function for VoxelKey to use in unordered_map
 */
struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& key) const {
        // Cantor pairing function for combining hash values
        std::size_t h1 = std::hash<int>{}(key.x);
        std::size_t h2 = std::hash<int>{}(key.y);
        std::size_t h3 = std::hash<int>{}(key.z);
        
        // Combine hashes using XOR and bit shifting
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

/**
 * @brief Voxel-based spatial hash map for efficient nearest neighbor search
 * 
 * This data structure provides O(1) voxel lookup and fast K-nearest neighbor search
 * by checking only neighboring voxels (27 voxels in 3x3x3 grid).
 * 
 * Performance comparison vs KdTree:
 * - KdTree: O(N * log(M)) where N=query points, M=map size
 * - VoxelMap: O(N * K) where K=fixed (27 voxels * points_per_voxel)
 * 
 * Expected speedup: 3-5x faster for large maps (20k+ points)
 */
class VoxelMap {
public:
    /**
     * @brief Constructor
     * @param voxel_size Size of each voxel in meters (default: 0.5m)
     */
    explicit VoxelMap(float voxel_size = 0.5f);
    
    /**
     * @brief Set voxel size
     * @param size Voxel size in meters
     */
    void SetVoxelSize(float size);
    
    /**
     * @brief Set maximum hit count for voxel occupancy
     * @param max_count Maximum hit count (default: 10)
     */
    void SetMaxHitCount(int max_count) { m_max_hit_count = max_count; }
    
    /**
     * @brief Get maximum hit count for voxel occupancy
     */
    int GetMaxHitCount() const { return m_max_hit_count; }
    
    /**
     * @brief Set hierarchy factor (L1 = factor × L0)
     * @param factor Hierarchy factor (3 = 3×3×3, 5 = 5×5×5, 7 = 7×7×7, must be odd)
     */
    void SetHierarchyFactor(int factor);
    
    /**
     * @brief Set planarity threshold for surfel creation
     * @param threshold Planarity threshold (sigma_min / sigma_max)
     *                  Lower values = stricter planarity requirement
     *                  Default: 0.01 (very strict for map quality)
     */
    void SetPlanarityThreshold(float threshold) { m_planarity_threshold = threshold; }
    
    /**
     * @brief Get current hierarchy factor
     */
    int GetHierarchyFactor() const { return m_hierarchy_factor; }
    
    /**
     * @brief Get current voxel size
     */
    float GetVoxelSize() const { return m_voxel_size; }
    
    /**
     * @brief Add a point to the voxel map
     * @param point 3D point to add
     */
    void AddPoint(const Point3D& point);
    
    /**
     * @brief Add multiple points to the voxel map
     * @param cloud Point cloud to add
     */
    void AddPointCloud(const PointCloudPtr& cloud);
    
    /**
     * @brief Clear all points from the map
     */
    void Clear();
    
    /**
     * @brief Update voxel map: add new points and remove distant voxels
     * @param new_cloud New point cloud to add
     * @param sensor_position Current sensor position in world frame
     * @param max_distance Maximum distance to keep voxels (meters)
     * 
     * This method:
     * 1. Adds new points to the map (creates new voxels automatically)
     * 2. Removes voxels that are more than max_distance away from sensor
     * 
     * This enables incremental map maintenance without full rebuild.
     */
    void UpdateVoxelMap(const PointCloudPtr& new_cloud,
                        const Eigen::Vector3d& sensor_position,
                        double max_distance, bool is_keyframe);
    
    /**
     * @brief Get total number of points in the map (sum of all voxel point counts)
     */
    size_t GetPointCount() const {
        size_t total = 0;
        for (const auto& pair : m_voxels_L0) {
            total += pair.second.point_count;
        }
        return total;
    }
    
    /**
     * @brief Get number of occupied voxels
     */
    size_t GetVoxelCount() const { return m_voxels_L0.size(); }
    
    /**
     * @brief Get centroid of a voxel as a Point3D
     * @param key Voxel key
     * @return Centroid as Point3D (with zero intensity and offset_time)
     */
    Point3D GetCentroidPoint(const VoxelKey& key) const;
    
    /**
     * @brief Get all occupied voxel keys for visualization
     * @return Vector of all occupied voxel keys
     */
    std::vector<VoxelKey> GetOccupiedVoxels() const;
    
    /**
     * @brief Convert voxel key to center position
     * @param key Voxel key
     * @return Center position of the voxel in world coordinates
     */
    Eigen::Vector3f VoxelKeyToCenter(const VoxelKey& key) const;
    
    /**
     * @brief Get the weighted centroid of a voxel
     * @param key Voxel key
     * @return Weighted average centroid of points in the voxel
     */
    Eigen::Vector3f GetVoxelCentroid(const VoxelKey& key) const;
    
    /**
     * @brief Get the hit count of a voxel
     * @param key Voxel key
     * @return Hit count (occupancy count) of the voxel
     */
    int GetVoxelHitCount(const VoxelKey& key) const;
    
    /**
     * @brief Mark a voxel as hit by current scan
     * @param key Voxel key to mark
     */
    void MarkVoxelAsHit(const VoxelKey& key);
    
    /**
     * @brief Clear all hit markers
     */
    void ClearHitMarkers();
    
    /**
     * @brief Check if a voxel is hit by current scan
     * @param key Voxel key to check
     * @return True if voxel is hit
     */
    bool IsVoxelHit(const VoxelKey& key) const;
    
    /**
     * @brief Get all hit voxel keys
     * @return Vector of hit voxel keys
     */
    std::vector<VoxelKey> GetHitVoxels() const;
    
    /**
     * @brief Get all L1 surfels for visualization
     * @return Vector of tuples: (centroid, normal, planarity_score, L1_key)
     */
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, VoxelKey>> GetL1Surfels() const;
    
    /**
     * @brief Get surfel information for the L1 voxel containing the given point
     * @param point Query point in world frame
     * @param normal Output: surfel normal
     * @param centroid Output: surfel centroid
     * @param planarity_score Output: planarity score
     * @return True if the containing L1 voxel has a valid surfel
     */
    bool GetSurfelAtPoint(const Point3D& point,
                          Eigen::Vector3f& normal,
                          Eigen::Vector3f& centroid,
                          float& planarity_score) const;
    
private:
    /**
     * @brief Convert 3D point to voxel key at specified level
     */
    VoxelKey PointToVoxelKey(const Point3D& point, int level = 0) const;
    
    /**
     * @brief Get parent voxel key
     */
    VoxelKey GetParentKey(const VoxelKey& key) const;
    
    /**
     * @brief Register L0 voxel to parent hierarchy
     */
    void RegisterToParent(const VoxelKey& key_L0);
    
    /**
     * @brief Unregister L0 voxel from parent hierarchy
     */
    void UnregisterFromParent(const VoxelKey& key_L0);
    
    /**
     * @brief Get all neighboring voxel keys within a specified distance
     */
    std::vector<VoxelKey> GetNeighborVoxels(const VoxelKey& center, float search_distance) const;
    
    // ===== Member Variables =====
    
    float m_voxel_size;  ///< Size of each voxel in meters (Level 0: 1×1×1)
    int m_max_hit_count; ///< Maximum hit count for occupancy (default: 10)
    int m_hierarchy_factor; ///< L1 voxel factor: L1 = factor × L0 (default: 3 for 3×3×3)
    float m_planarity_threshold = 0.01f; ///< Planarity threshold for surfel creation (sigma_min/sigma_max)
    
    // ===== Hierarchical Voxel Structure (2 Levels) =====
    
    /// Level 0: Leaf nodes (1×1×1) - stores centroid only (no raw points)
    struct VoxelNode_L0 {
        Eigen::Vector3f centroid;
        int hit_count;
        int point_count;  // Number of points used to compute centroid
        
        VoxelNode_L0() : centroid(Eigen::Vector3f::Zero()), hit_count(1), point_count(0) {}
    };
    std::unordered_map<VoxelKey, VoxelNode_L0, VoxelKeyHash> m_voxels_L0;
    
    /// Level 1: Parent nodes (3×3×3) - tracks occupied L0 children
    struct VoxelNode_L1 {
        int hit_count;
        std::unordered_set<VoxelKey, VoxelKeyHash> occupied_children;  // L0 keys
        
        // Surfel data (only valid if has_surfel == true)
        bool has_surfel;
        Eigen::Vector3f surfel_normal;     // Plane normal vector
        Eigen::Vector3f surfel_centroid;   // Plane centroid
        Eigen::Matrix3f surfel_covariance; // Covariance matrix for plane fitting
        float planarity_score;             // sigma_min / sigma_max (smaller = more planar)
        int last_child_count;              // Track number of children at last surfel update
        
        VoxelNode_L1() 
            : hit_count(0)
            , has_surfel(false)
            , surfel_normal(Eigen::Vector3f::Zero())
            , surfel_centroid(Eigen::Vector3f::Zero())
            , surfel_covariance(Eigen::Matrix3f::Zero())
            , planarity_score(1.0f)
            , last_child_count(0) {}
    };
    std::unordered_map<VoxelKey, VoxelNode_L1, VoxelKeyHash> m_voxels_L1;
    
    /// Hit markers for current scan visualization
    std::unordered_map<VoxelKey, bool, VoxelKeyHash> m_hit_voxels;
    
    /// Thread synchronization recursive mutex for thread-safe access (allows re-locking)
    mutable std::recursive_mutex m_mutex;
};

} // namespace lio

#endif // VOXEL_MAP_H
