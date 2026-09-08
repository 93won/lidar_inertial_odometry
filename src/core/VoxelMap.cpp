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
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace lio {

VoxelMap::VoxelMap(float voxel_size) 
    : m_voxel_size(voxel_size)
    , m_max_hit_count(10)
    , m_hierarchy_factor(3)  // Default: 3×3×3
{
    if (!std::isfinite(voxel_size) || voxel_size <= 0.0f) {
        throw std::invalid_argument("Voxel size must be finite and positive");
    }
}

void VoxelMap::SetVoxelSize(float size) {
    if (!std::isfinite(size) || size <= 0.0f) {
        throw std::invalid_argument("Voxel size must be finite and positive");
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
    
    // If L1 has fewer than 5 children, invalidate surfel
    if (it_L1->second.occupied_children.size() < 5) {
        it_L1->second.has_surfel = false;
    }
    
    // If L1 becomes empty, clean it up
    if (it_L1->second.occupied_children.empty()) {
        m_voxels_L1.erase(it_L1);
    }
}

void VoxelMap::ErasePointIndex(const VoxelKey& key) {
    const auto iterator = m_point_ids.find(key);
    if (iterator == m_point_ids.end()) {
        return;
    }
    m_point_index.Erase(iterator->second);
    m_point_ids.erase(iterator);
}

void VoxelMap::AddPointUnlocked(
    const Point3D& point, std::vector<SpatialIndex::Point>& additions) {
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
        // First point in voxel - initialize with init_hit_count
        voxel_data.centroid = point_vec;
        voxel_data.hit_count = m_init_hit_count;  // Initialize with init_hit_count
        voxel_data.point_count = 1;
    } else {
        // Weighted average: new_centroid = (n * old_centroid + new_point) / (n + 1)
        voxel_data.centroid = (voxel_data.centroid * n + point_vec) / (n + 1);
        voxel_data.point_count++;
    }
    
    // Register to hierarchy if newly occupied
    if (was_empty) {
        RegisterToParent(key);
        const std::uint64_t id = m_next_point_id++;
        m_point_ids[key] = id;
        additions.push_back({id, point_vec});
    }
}

void VoxelMap::AddPoint(const Point3D& point) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    std::vector<SpatialIndex::Point> additions;
    AddPointUnlocked(point, additions);
    m_point_index.Add(additions);
}

void VoxelMap::AddPointCloud(const PointCloudPtr& cloud) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!cloud || cloud->empty()) {
        spdlog::warn("[VoxelMap] UpdateVoxelMap called with empty point cloud");
        return;
    }

    std::vector<SpatialIndex::Point> additions;
    additions.reserve(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) {
        AddPointUnlocked(cloud->at(i), additions);
    }
    m_point_index.Add(additions);
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
    m_point_index.Clear();
    m_point_ids.clear();
    m_next_point_id = 1;
    m_hit_voxels.clear();
    m_map_center.setZero();
    m_map_initialized = false;
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
    
    // Store max_distance for box size calculation
    m_max_distance = static_cast<float>(max_distance);
    
    Eigen::Vector3f sensor_pos = sensor_position.cast<float>();
    
    // Box geometry:
    // - box_half_size = max_distance × multiplier (e.g., 100m × 2 = 200m from center)
    // - Total box size = 2 × box_half_size = 400m × 400m × 400m
    // - Re-center trigger = max_distance / multiplier (e.g., 100m / 2 = 50m)
    float box_half_size = m_max_distance * m_map_box_multiplier;
    float recenter_threshold = m_max_distance / m_map_box_multiplier;
    
    // === Step 1: Initialize map center on first frame ===
    if (!m_map_initialized) {
        m_map_center = sensor_pos;
        m_map_initialized = true;
        spdlog::info("[VoxelMap] Map initialized at center: ({:.1f}, {:.1f}, {:.1f}), box size: {:.1f}m x {:.1f}m", 
                     m_map_center.x(), m_map_center.y(), m_map_center.z(), 
                     box_half_size * 2.0f, box_half_size * 2.0f);
    }
    
    // === Step 2: Check if we need to re-center the map ===
    float dist_from_center = (sensor_pos - m_map_center).norm();
    bool need_recenter = dist_from_center > recenter_threshold;
    
    if (need_recenter) {
        // Move map center to current sensor position
        Eigen::Vector3f old_center = m_map_center;
        m_map_center = sensor_pos;
        
        // === Step 3: Remove voxels outside the new box ===
        std::vector<VoxelKey> voxels_to_remove;
        
        for (const auto& pair : m_voxels_L0) {
            const VoxelKey& key = pair.first;
            Eigen::Vector3f voxel_center = VoxelKeyToCenter(key);
            
            // Check if voxel is outside the new box (axis-aligned bounding box)
            bool outside_box = 
                std::abs(voxel_center.x() - m_map_center.x()) > box_half_size ||
                std::abs(voxel_center.y() - m_map_center.y()) > box_half_size ||
                std::abs(voxel_center.z() - m_map_center.z()) > box_half_size;
            
            if (outside_box) {
                voxels_to_remove.push_back(key);
            }
        }
        
        // Remove voxels outside the box
        for (const auto& key : voxels_to_remove) {
            UnregisterFromParent(key);
            ErasePointIndex(key);
            m_voxels_L0.erase(key);
        }
        
        // Also remove L1 voxels that have no children
        std::vector<VoxelKey> L1_to_remove;
        for (const auto& pair : m_voxels_L1) {
            if (pair.second.occupied_children.empty()) {
                L1_to_remove.push_back(pair.first);
            }
        }
        for (const auto& key : L1_to_remove) {
            m_voxels_L1.erase(key);
        }
    }
    
    // === Step 4: Add new points ===
    AddPointCloud(new_cloud);
    
    // === Step 5: Collect affected L1 voxels for surfel update ===
    ankerl::unordered_dense::set<VoxelKey, VoxelKeyHash> affected_L1;
    
    for (const auto& pt : *new_cloud) {
        VoxelKey key_L1 = PointToVoxelKey(pt, 1);
        affected_L1.insert(key_L1);
    }
    
    // === Step 6: Create/update surfels for affected L1 voxels ===
    const int MIN_OCCUPIED_CHILDREN = 5;
    
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

        // Keep a stable plane while the same leaf voxels remain occupied.
        if (node_L1.has_surfel && node_L1.last_child_count == current_child_count) {
            continue;
        }

        // Collect centroids from occupied L0 children
        std::vector<Eigen::Vector3f> collected_centroids;
        collected_centroids.reserve(node_L1.occupied_children.size());
        
        for (const VoxelKey& key_L0 : node_L1.occupied_children) {
            auto it_L0 = m_voxels_L0.find(key_L0);
            if (it_L0 == m_voxels_L0.end()) continue;
            collected_centroids.push_back(it_L0->second.centroid);
        }
        
        if (collected_centroids.size() < 3) {
            node_L1.has_surfel = false;
            continue;
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
        
        // SVD decomposition
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3f singular_values = svd.singularValues();
        Eigen::Vector3f normal = svd.matrixU().col(2);
        float planarity = singular_values(2) / (singular_values(0) + 1e-6f);

        if (planarity > m_planarity_threshold) {
            // Keep map points; query-time KNN plane fitting rejects non-planar neighborhoods.
            node_L1.has_surfel = false;
            continue;
        }
        
        // Create surfel
        node_L1.has_surfel = true;
        node_L1.surfel_normal = normal;
        node_L1.surfel_centroid = centroid;
        node_L1.surfel_covariance = covariance;
        node_L1.planarity_score = planarity;
        node_L1.last_child_count = current_child_count;
    }
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
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_voxels_L0.find(key);
    if (it == m_voxels_L0.end()) {
        // Voxel not found, return geometric center as fallback
        return VoxelKeyToCenter(key);
    }
    
    // Return the weighted centroid stored in VoxelNode_L0
    return it->second.centroid;
}

Point3D VoxelMap::GetCentroidPoint(const VoxelKey& key) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
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
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_voxels_L0.find(key);
    if (it == m_voxels_L0.end()) {
        return 0;  // Voxel not found
    }
    
    return it->second.hit_count;
}

std::vector<std::pair<Eigen::Vector3f, int>> VoxelMap::GetOccupiedVoxelsWithHitCount() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    std::vector<std::pair<Eigen::Vector3f, int>> result;
    result.reserve(m_voxels_L0.size());
    
    for (const auto& pair : m_voxels_L0) {
        if (pair.second.point_count > 0) {
            Eigen::Vector3f center = VoxelKeyToCenter(pair.first);
            result.emplace_back(center, pair.second.hit_count);
        }
    }
    
    return result;
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
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    return m_hit_voxels.find(key) != m_hit_voxels.end();
}

std::vector<VoxelKey> VoxelMap::GetHitVoxels() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    std::vector<VoxelKey> hit_voxels;
    hit_voxels.reserve(m_hit_voxels.size());
    
    for (const auto& pair : m_hit_voxels) {
        hit_voxels.push_back(pair.first);
    }
    
    return hit_voxels;
}

std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, VoxelKey, int>> VoxelMap::GetL1Surfels() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, VoxelKey, int>> surfels;
    
    for (const auto& pair : m_voxels_L1) {
        const VoxelKey& key = pair.first;
        const VoxelNode_L1& node = pair.second;
        
        if (node.has_surfel) {
            surfels.emplace_back(node.surfel_centroid, node.surfel_normal, node.planarity_score, key, node.hit_count);
        }
    }
    
    return surfels;
}

bool VoxelMap::GetSurfelAtPoint(const Point3D& point,
                                 Eigen::Vector3f& normal,
                                 Eigen::Vector3f& centroid,
                                 float& planarity_score) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
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

bool VoxelMap::GetClosestSurfel(const Point3D& point,
                                float max_centroid_distance,
                                float max_plane_distance,
                                Eigen::Vector3f& normal,
                                Eigen::Vector3f& centroid,
                                float& planarity_score,
                                MatchDiagnostics* diagnostics) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (diagnostics) {
        ++diagnostics->queries;
    }
    if (max_centroid_distance < 0.0f || max_plane_distance < 0.0f
        || m_point_index.ValidSize() < 3) {
        if (diagnostics) {
            ++diagnostics->insufficient_map;
        }
        return false;
    }
    const Eigen::Vector3f query(point.x, point.y, point.z);

    const std::size_t required_neighbors =
        static_cast<std::size_t>(m_k_nearest_neighbors);
    const auto neighbors = m_point_index.KNearest(query, required_neighbors);
    if (neighbors.size() != required_neighbors) {
        if (diagnostics) {
            ++diagnostics->insufficient_neighbors;
        }
        return false;
    }
    if ((neighbors.front().position - query).norm() > max_centroid_distance) {
        if (diagnostics) {
            ++diagnostics->centroid_gate;
        }
        return false;
    }

    centroid.setZero();
    for (const auto& neighbor : neighbors) {
        centroid += neighbor.position;
    }
    centroid /= static_cast<float>(neighbors.size());

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto& neighbor : neighbors) {
        const Eigen::Vector3f offset = neighbor.position - centroid;
        covariance.noalias() += offset * offset.transpose();
    }
    covariance /= static_cast<float>(neighbors.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
    if (eigen_solver.info() != Eigen::Success) {
        if (diagnostics) {
            ++diagnostics->eigen_failure;
        }
        return false;
    }
    const Eigen::Vector3f eigenvalues = eigen_solver.eigenvalues().cwiseMax(0.0f);
    if (!eigenvalues.allFinite() || eigenvalues.y() <= 1e-8f
        || eigenvalues.y() / (eigenvalues.z() + 1e-10f) < m_min_linearity_ratio) {
        if (diagnostics) {
            ++diagnostics->degenerate_neighbors;
        }
        return false;
    }
    const float planarity = 1.0f - eigenvalues.x() / (eigenvalues.y() + 1e-10f);
    if (planarity < m_knn_planarity_threshold) {
        if (diagnostics) {
            ++diagnostics->planarity_gate;
        }
        return false;
    }

    normal = eigen_solver.eigenvectors().col(0);
    if (normal.dot(query - centroid) < 0.0f) {
        normal = -normal;
    }
    for (const auto& neighbor : neighbors) {
        if (std::abs(normal.dot(neighbor.position - centroid))
            > m_point_to_surfel_threshold) {
            if (diagnostics) {
                ++diagnostics->neighbor_plane_gate;
            }
            return false;
        }
    }
    if (std::abs(normal.dot(query - centroid)) > max_plane_distance) {
        if (diagnostics) {
            ++diagnostics->query_plane_gate;
        }
        return false;
    }

    planarity_score = planarity;
    if (diagnostics) {
        ++diagnostics->accepted;
    }
    return true;
}

SpatialIndex::Statistics VoxelMap::GetPointIndexStatistics() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    return m_point_index.GetStatistics();
}

} // namespace lio
