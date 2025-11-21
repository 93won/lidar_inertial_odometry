/**
 * @file      VoxelMap.cpp
 * @brief     Implementation of voxel-based hash map for efficient nearest neighbor search
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "VoxelMap.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_set>
#include <spdlog/spdlog.h>

namespace lio {

VoxelMap::VoxelMap(float voxel_size) 
    : m_voxel_size(voxel_size)
    , m_max_hit_count(10)
    , m_hierarchy_factor(3)  // Default: 3×3×3
{
}

void VoxelMap::SetVoxelSize(float size) {
    if (size <= 0.0f) {
        throw std::invalid_argument("Voxel size must be positive");
    }
    
    // If size changed, need to rebuild the map
    if (std::abs(m_voxel_size - size) > 1e-6f) {
        m_voxel_size = size;
        
        // Clear map - cannot easily rebuild without original points
        Clear();
        spdlog::warn("[VoxelMap] Voxel size changed - map cleared");
    }
}

void VoxelMap::SetHierarchyFactor(int factor) {
    if (factor <= 0 || factor % 2 == 0) {
        spdlog::error("[VoxelMap] Hierarchy factor must be positive and odd (3, 5, 7, etc.). Got: {}", factor);
        return;
    }
    
    // If factor changed, need to rebuild hierarchy
    if (m_hierarchy_factor != factor) {
        m_hierarchy_factor = factor;
        
        // Clear map - hierarchy structure is incompatible
        Clear();
        spdlog::info("[VoxelMap] Hierarchy factor changed to {} ({}×{}×{}) - map cleared", 
                     factor, factor, factor, factor);
    }
}

VoxelKey VoxelMap::PointToVoxelKey(const Point3D& point, int level) const {
    // Level 0: 1×1×1 (voxel_size)
    // Level 1: factor×factor×factor (factor * voxel_size)
    
    float scale = m_voxel_size;
    if (level == 1) scale *= static_cast<float>(m_hierarchy_factor);
    
    int vx = static_cast<int>(std::floor(point.x / scale));
    int vy = static_cast<int>(std::floor(point.y / scale));
    int vz = static_cast<int>(std::floor(point.z / scale));
    return VoxelKey(vx, vy, vz);
}

VoxelKey VoxelMap::GetParentKey(const VoxelKey& key) const {
    // Parent key: divide by hierarchy_factor (floor division)
    int f = m_hierarchy_factor;
    return VoxelKey(
        key.x >= 0 ? key.x / f : (key.x - (f - 1)) / f,
        key.y >= 0 ? key.y / f : (key.y - (f - 1)) / f,
        key.z >= 0 ? key.z / f : (key.z - (f - 1)) / f
    );
}

void VoxelMap::RegisterToParent(const VoxelKey& key_L0) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Get L1 parent and register
    VoxelKey parent_L1 = GetParentKey(key_L0);
    m_voxels_L1[parent_L1].occupied_children.insert(key_L0);
}

void VoxelMap::UnregisterFromParent(const VoxelKey& key_L0) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Get L1 parent
    VoxelKey parent_L1 = GetParentKey(key_L0);
    
    auto it_L1 = m_voxels_L1.find(parent_L1);
    if (it_L1 == m_voxels_L1.end()) return;
    
    // Unregister from L1
    it_L1->second.occupied_children.erase(key_L0);
    
    // If L1 becomes empty, clean it up
    if (it_L1->second.occupied_children.empty()) {
        m_voxels_L1.erase(it_L1);
    }
}

void VoxelMap::AddPoint(const Point3D& point) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Get L0 voxel key for this point
    VoxelKey key = PointToVoxelKey(point, 0);
    
    // Check if voxel is newly occupied
    auto it = m_voxels_L0.find(key);
    bool was_empty = (it == m_voxels_L0.end());
    
    // Get or create voxel data
    VoxelNode_L0& voxel_data = m_voxels_L0[key];
    
    // Update centroid with weighted average
    Eigen::Vector3f point_vec(point.x, point.y, point.z);
    int n = voxel_data.point_count;
    
    if (n == 0) {
        // First point in voxel - initialize with max_voxel_hit_count
        voxel_data.centroid = point_vec;
        voxel_data.hit_count = m_max_hit_count;  // Initialize with max_voxel_hit_count
        voxel_data.point_count = 1;
    } else {
        // Weighted average: new_centroid = (n * old_centroid + new_point) / (n + 1)
        voxel_data.centroid = (voxel_data.centroid * n + point_vec) / (n + 1);
        voxel_data.point_count++;
    }
    
    // Register to hierarchy if newly occupied
    if (was_empty) {
        RegisterToParent(key);
    }
}

void VoxelMap::AddPointCloud(const PointCloudPtr& cloud) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!cloud || cloud->empty()) {
        spdlog::warn("[VoxelMap] UpdateVoxelMap called with empty point cloud");
        return;
    }

    for (size_t i = 0; i < cloud->size(); ++i) {
        AddPoint(cloud->at(i));
    }
}

std::vector<VoxelKey> VoxelMap::GetNeighborVoxels(const VoxelKey& center, float search_distance) const {
    std::vector<VoxelKey> neighbors;
    
    // Calculate how many voxels to search in each direction
    // search_distance is in meters, m_voxel_size is voxel size in meters
    // Example: search_distance=100m, voxel_size=0.5m => search_range = 200 voxels
    int search_range = static_cast<int>(std::ceil(search_distance / m_voxel_size));
    
    // Reserve space for efficiency (upper bound estimate)
    int grid_size = 2 * search_range + 1;
    neighbors.reserve(grid_size * grid_size * grid_size);
    
    // Search in cubic grid around center voxel
    for (int dx = -search_range; dx <= search_range; ++dx) {
        for (int dy = -search_range; dy <= search_range; ++dy) {
            for (int dz = -search_range; dz <= search_range; ++dz) {
                neighbors.emplace_back(center.x + dx, center.y + dy, center.z + dz);
            }
        }
    }
    
    return neighbors;
}

void VoxelMap::Clear() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_voxels_L0.clear();
    m_voxels_L1.clear();
}

void VoxelMap::UpdateVoxelMap(const PointCloudPtr& new_cloud,
                               const Eigen::Vector3d& sensor_position,
                               double max_distance, bool is_keyframe) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    
    if (new_cloud->empty()) 
    {
        spdlog::warn("[VoxelMap] UpdateVoxelMap called with empty point cloud");
        return;
    }
    // Only add new points for keyframes
    if (!is_keyframe) {
        return;
    }
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Step 1: Add new points (creates new voxels OR updates existing centroid)
    auto start_step1 = std::chrono::high_resolution_clock::now();
    AddPointCloud(new_cloud);
    auto end_step1 = std::chrono::high_resolution_clock::now();
    double time_step1 = std::chrono::duration<double, std::milli>(end_step1 - start_step1).count();

    // Step 2: Mark L0 voxels - L1-based efficient marking
    auto start_step2 = std::chrono::high_resolution_clock::now();
    std::unordered_set<VoxelKey, VoxelKeyHash> hit_voxels_L0_set;
    std::unordered_set<VoxelKey, VoxelKeyHash> hit_L1_set;
    
    // First, collect all L1 voxels containing the new points
    for (const auto& pt : *new_cloud) {
        VoxelKey key_L1 = PointToVoxelKey(pt, 1);
        hit_L1_set.insert(key_L1);
    }
    
    // Then, mark all occupied L0 voxels within those L1 voxels
    for (const VoxelKey& key_L1 : hit_L1_set) {
        auto it_L1 = m_voxels_L1.find(key_L1);
        if (it_L1 == m_voxels_L1.end()) continue;
        
        // Mark all occupied L0 children
        for (const VoxelKey& key_L0 : it_L1->second.occupied_children) {
            hit_voxels_L0_set.insert(key_L0);
        }
    }
    auto end_step2 = std::chrono::high_resolution_clock::now();
    double time_step2 = std::chrono::duration<double, std::milli>(end_step2 - start_step2).count();
    
    // Step 3: Update hit counts for all existing L0 voxels
    auto start_step3 = std::chrono::high_resolution_clock::now();
    std::vector<VoxelKey> voxels_to_remove;
    
    for (auto& pair : m_voxels_L0) {
        const VoxelKey& key = pair.first;
        VoxelNode_L0& voxel_data = pair.second;
        
        // Check if this voxel was hit by the current scan
        if (hit_voxels_L0_set.find(key) != hit_voxels_L0_set.end()) {
            // Voxel is within culling distance of new scan -> increment hit count (up to max)
            if (voxel_data.hit_count < m_max_hit_count) {
                voxel_data.hit_count++;
            }
        } else {
            // Voxel is not hit -> decrement hit count by 1
            voxel_data.hit_count--;
            
            // Mark for removal if hit count drops below 1
            if (voxel_data.hit_count < 1) {
                voxels_to_remove.push_back(key);
            }
        }
    }
    auto end_step3 = std::chrono::high_resolution_clock::now();
    double time_step3 = std::chrono::duration<double, std::milli>(end_step3 - start_step3).count();
    
    // Step 4: Remove L0 voxels with hit_count < 1 and update hierarchy
    auto start_step4 = std::chrono::high_resolution_clock::now();
    for (const auto& key : voxels_to_remove) {
        UnregisterFromParent(key);  // Remove from L1 hierarchy
        m_voxels_L0.erase(key);
    }
    auto end_step4 = std::chrono::high_resolution_clock::now();
    double time_step4 = std::chrono::duration<double, std::milli>(end_step4 - start_step4).count();
    
    // Step 5: Update L1 hit counts based on affected parents
    auto start_step5 = std::chrono::high_resolution_clock::now();
    std::unordered_set<VoxelKey, VoxelKeyHash> affected_L1;
    
    for (const VoxelKey& hit_L0 : hit_voxels_L0_set) {
        VoxelKey parent_L1 = GetParentKey(hit_L0);
        affected_L1.insert(parent_L1);
    }
    
    // Update L1 hit counts
    for (const VoxelKey& key_L1 : affected_L1) {
        auto it = m_voxels_L1.find(key_L1);
        if (it != m_voxels_L1.end()) {
            if (it->second.hit_count < m_max_hit_count) {
                it->second.hit_count++;
            }
        }
    }
    auto end_step5 = std::chrono::high_resolution_clock::now();
    double time_step5 = std::chrono::duration<double, std::milli>(end_step5 - start_step5).count();
    
    // Step 6: Create/update surfels for affected L1 voxels
    auto start_step6 = std::chrono::high_resolution_clock::now();
    const int MIN_OCCUPIED_CHILDREN = 5;  // Minimum 5 occupied L0 voxels required to create surfel
    const float MAX_PLANARITY_SCORE = 0.01f;  // Consider as planar if sigma_min / sigma_max < 0.1
    const float MAX_POINT_TO_PLANE_DIST = 0.05f;  // Max distance for incremental update (5cm)
    
    int surfels_created = 0;
    int surfels_skipped = 0;
    
    for (const VoxelKey& key_L1 : affected_L1) {
        auto it_L1 = m_voxels_L1.find(key_L1);
        if (it_L1 == m_voxels_L1.end()) continue;
        
        VoxelNode_L1& node_L1 = it_L1->second;
        int current_child_count = node_L1.occupied_children.size();
        
        // Check if enough L0 voxels are occupied
        if (current_child_count < MIN_OCCUPIED_CHILDREN) {
            node_L1.has_surfel = false;
            continue;
        }
        
        // === INCREMENTAL UPDATE: Check if existing surfel is still valid ===
        if (node_L1.has_surfel) {
            // Check if geometry changed significantly
            int child_count_change = std::abs(current_child_count - node_L1.last_child_count);
            
            // If child count didn't change much, check point-to-plane distances
            if (child_count_change <= 2) {
                bool all_points_close = true;
                float max_dist = 0.0f;
                
                // Check all L0 centroids against existing surfel plane
                for (const VoxelKey& key_L0 : node_L1.occupied_children) {
                    auto it_L0 = m_voxels_L0.find(key_L0);
                    if (it_L0 == m_voxels_L0.end()) continue;
                    
                    // Point-to-plane distance: |n · (p - c)|
                    Eigen::Vector3f diff = it_L0->second.centroid - node_L1.surfel_centroid;
                    float dist = std::abs(node_L1.surfel_normal.dot(diff));
                    max_dist = std::max(max_dist, dist);
                    
                    if (dist > MAX_POINT_TO_PLANE_DIST) {
                        all_points_close = false;
                        break;
                    }
                }
                
                // If all points are close, skip SVD (surfel is still valid)
                if (all_points_close) {
                    surfels_skipped++;
                    surfels_created++;  // Count as "created" for statistics
                    continue;
                }
            }
        }
        
        // === FULL UPDATE: Recompute surfel with SVD ===
        
        // Collect centroids from occupied L0 children
        std::vector<Eigen::Vector3f> collected_centroids;
        collected_centroids.reserve(node_L1.occupied_children.size());
        
        for (const VoxelKey& key_L0 : node_L1.occupied_children) {
            auto it_L0 = m_voxels_L0.find(key_L0);
            if (it_L0 == m_voxels_L0.end()) continue;
            
            // Use voxel centroid as representative point
            collected_centroids.push_back(it_L0->second.centroid);
        }
        
        // Need at least 5 centroids for stable plane fitting
        if (collected_centroids.size() < 5) {
            node_L1.has_surfel = false;
            continue;  // Still collecting points, don't delete yet
        }
        
        // Compute centroid
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto& pt : collected_centroids) {
            centroid += pt;
        }
        centroid /= static_cast<float>(collected_centroids.size());
        
        // Compute covariance matrix
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const auto& pt : collected_centroids) {
            Eigen::Vector3f diff = pt - centroid;
            covariance += diff * diff.transpose();
        }
        covariance /= static_cast<float>(collected_centroids.size());
        
        // SVD decomposition to get eigenvalues and eigenvectors
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3f singular_values = svd.singularValues();
        
        // Planarity check: sigma_min / sigma_max should be small
        float planarity = singular_values(2) / (singular_values(0) + 1e-6f);
        
        if (planarity > MAX_PLANARITY_SCORE) {
            // Not planar enough - clear all L0 children
            node_L1.has_surfel = false;
            
            // Clear all L0 children of this L1 voxel (set hit_count to 0)
            for (const VoxelKey& key_L0 : node_L1.occupied_children) {
                auto it_L0 = m_voxels_L0.find(key_L0);
                if (it_L0 != m_voxels_L0.end()) {
                    it_L0->second.hit_count = 0;  // Mark for removal
                }
            }
            node_L1.occupied_children.clear();  // Clear children list
            
            continue;
        }
        
        // Extract plane normal (smallest eigenvector)
        Eigen::Vector3f normal = svd.matrixU().col(2);
        
        // Ensure consistent normal orientation (pointing upward if possible)
        if (normal.z() < 0) {
            normal = -normal;
        }
        
        // Create surfel and update last child count
        node_L1.has_surfel = true;
        node_L1.surfel_normal = normal;
        node_L1.surfel_centroid = centroid;
        node_L1.surfel_covariance = covariance;
        node_L1.planarity_score = planarity;
        node_L1.last_child_count = current_child_count;
        surfels_created++;
    }
    auto end_step6 = std::chrono::high_resolution_clock::now();
    double time_step6 = std::chrono::duration<double, std::milli>(end_step6 - start_step6).count();
    
    // Calculate total time
    auto end_total = std::chrono::high_resolution_clock::now();
    double time_total = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    
   
}

std::vector<VoxelKey> VoxelMap::GetOccupiedVoxels() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    std::vector<VoxelKey> occupied_voxels;
    occupied_voxels.reserve(m_voxels_L0.size());
    
    for (const auto& pair : m_voxels_L0) {
        if (pair.second.point_count > 0) {  // Only add non-empty voxels
            occupied_voxels.push_back(pair.first);
        }
    }
    
    return occupied_voxels;
}

Eigen::Vector3f VoxelMap::VoxelKeyToCenter(const VoxelKey& key) const {
    // Convert voxel key back to world coordinates (center of voxel)
    float center_x = (key.x + 0.5f) * m_voxel_size;
    float center_y = (key.y + 0.5f) * m_voxel_size;
    float center_z = (key.z + 0.5f) * m_voxel_size;
    
    return Eigen::Vector3f(center_x, center_y, center_z);
}

Eigen::Vector3f VoxelMap::GetVoxelCentroid(const VoxelKey& key) const {
    auto it = m_voxels_L0.find(key);
    if (it == m_voxels_L0.end()) {
        // Voxel not found, return geometric center as fallback
        return VoxelKeyToCenter(key);
    }
    
    // Return the weighted centroid stored in VoxelNode_L0
    return it->second.centroid;
}

Point3D VoxelMap::GetCentroidPoint(const VoxelKey& key) const {
    auto it = m_voxels_L0.find(key);
    if (it == m_voxels_L0.end()) {
        // Voxel not found, return geometric center as fallback
        Eigen::Vector3f center = VoxelKeyToCenter(key);
        return Point3D(center.x(), center.y(), center.z(), 0.0f, 0.0f);
    }
    
    // Return the weighted centroid as Point3D
    const Eigen::Vector3f& centroid = it->second.centroid;
    return Point3D(centroid.x(), centroid.y(), centroid.z(), 0.0f, 0.0f);
}

int VoxelMap::GetVoxelHitCount(const VoxelKey& key) const {
    auto it = m_voxels_L0.find(key);
    if (it == m_voxels_L0.end()) {
        return 0;  // Voxel not found
    }
    
    return it->second.hit_count;
}

void VoxelMap::MarkVoxelAsHit(const VoxelKey& key) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_hit_voxels[key] = true;
}

void VoxelMap::ClearHitMarkers() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_hit_voxels.clear();
}

bool VoxelMap::IsVoxelHit(const VoxelKey& key) const {
    return m_hit_voxels.find(key) != m_hit_voxels.end();
}

std::vector<VoxelKey> VoxelMap::GetHitVoxels() const {
    std::vector<VoxelKey> hit_voxels;
    hit_voxels.reserve(m_hit_voxels.size());
    
    for (const auto& pair : m_hit_voxels) {
        hit_voxels.push_back(pair.first);
    }
    
    return hit_voxels;
}

std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, VoxelKey>> VoxelMap::GetL1Surfels() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, VoxelKey>> surfels;
    
    for (const auto& pair : m_voxels_L1) {
        const VoxelKey& key = pair.first;
        const VoxelNode_L1& node = pair.second;
        
        if (node.has_surfel) {
            surfels.emplace_back(node.surfel_centroid, node.surfel_normal, node.planarity_score, key);
        }
    }
    
    return surfels;
}

bool VoxelMap::GetSurfelAtPoint(const Point3D& point,
                                 Eigen::Vector3f& normal,
                                 Eigen::Vector3f& centroid,
                                 float& planarity_score) const {
    // Convert point to L1 voxel key
    VoxelKey key_L1 = PointToVoxelKey(point, 1);
    
    // Find L1 voxel
    auto it = m_voxels_L1.find(key_L1);
    if (it == m_voxels_L1.end()) {
        return false;  // L1 voxel doesn't exist
    }
    
    const VoxelNode_L1& node = it->second;
    
    // Check if surfel exists
    if (!node.has_surfel) {
        return false;  // No surfel in this L1 voxel
    }
    
    // Return surfel information
    normal = node.surfel_normal;
    centroid = node.surfel_centroid;
    planarity_score = node.planarity_score;
    
    return true;
}

} // namespace lio
