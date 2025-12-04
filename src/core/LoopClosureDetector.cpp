/**
 * @file      LoopClosureDetector.cpp
 * @brief     Loop closure detection using LiDAR Iris features
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-12-04
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "LoopClosureDetector.h"
#include <algorithm>
#include <chrono>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lio {

LoopClosureDetector::LoopClosureDetector(const LoopClosureConfig& config)
    : m_config(config)
    , m_last_keyframe_position(Eigen::Vector3f::Zero())
    , m_last_keyframe_rotation(Eigen::Matrix3f::Identity())
    , m_has_keyframe(false)
{
    // Initialize LiDAR Iris detector with standard parameters
    m_iris = std::make_unique<LidarIris>(
        4,      // nscale: number of filter scales
        18,     // minWaveLength: minimum wavelength
        2.1f,   // mult: wavelength multiplier
        0.75f,  // sigmaOnf: bandwidth parameter
        2       // matchNum: both forward and reverse directions
    );
    
    spdlog::info("[LoopClosureDetector] Initialized with:");
    spdlog::info("  - similarity_threshold: {:.3f}", m_config.similarity_threshold);
    spdlog::info("  - min_keyframe_gap: {}", m_config.min_keyframe_gap);
    spdlog::info("  - max_search_distance: {:.1f}m", m_config.max_search_distance);
    spdlog::info("  - keyframe_translation_threshold: {:.2f}m", m_config.keyframe_translation_threshold);
}

LoopClosureDetector::~LoopClosureDetector() {
    spdlog::info("[LoopClosureDetector] Statistics:");
    spdlog::info("  - Total keyframes: {}", m_keyframes.size());
    spdlog::info("  - Total queries: {}", m_stats.total_queries);
    spdlog::info("  - Total loops detected: {}", m_detected_loops.size());
}

bool LoopClosureDetector::ShouldCreateKeyframe(const State& current_state) const {
    if (!m_has_keyframe) {
        return true;  // First keyframe
    }
    
    // Check translation threshold only (1m by default)
    Eigen::Vector3f translation_diff = current_state.m_position - m_last_keyframe_position;
    float translation_distance = translation_diff.norm();
    
    return translation_distance >= m_config.keyframe_translation_threshold;
}

int LoopClosureDetector::AddKeyframe(double timestamp, const State& state, PointCloudPtr cloud) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!cloud || cloud->empty()) {
        spdlog::warn("[LoopClosureDetector] Empty point cloud provided");
        return -1;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create new keyframe
        size_t new_id = m_keyframes.size();
        auto keyframe = std::make_shared<Keyframe>(new_id, timestamp, state, cloud);
        
        // Extract LiDAR Iris feature
        keyframe->iris_feature = ExtractIrisFeature(cloud);
        
        // Add to database
        m_keyframes.push_back(keyframe);
        
        // Update last keyframe state
        m_last_keyframe_position = state.m_position;
        m_last_keyframe_rotation = state.m_rotation;
        m_has_keyframe = true;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        m_total_feature_time += duration_ms;
        m_stats.total_keyframes++;
        m_stats.avg_feature_extraction_time_ms = m_total_feature_time / m_stats.total_keyframes;
        
        if (m_config.enable_debug_output) {
            spdlog::debug("[LoopClosureDetector] Added keyframe {} at position [{:.2f}, {:.2f}, {:.2f}] - {}ms",
                         new_id, state.m_position.x(), state.m_position.y(), state.m_position.z(),
                         duration_ms);
        }
        
        return static_cast<int>(new_id);
        
    } catch (const std::exception& e) {
        spdlog::error("[LoopClosureDetector] Exception adding keyframe: {}", e.what());
        return -1;
    }
}

std::vector<LoopCandidate> LoopClosureDetector::DetectLoopClosures() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_keyframes.empty()) {
        return {};
    }
    
    // Detect for the latest keyframe
    return SearchLoopCandidates(*m_keyframes.back());
}

std::vector<LoopCandidate> LoopClosureDetector::DetectLoopClosures(size_t keyframe_id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (keyframe_id >= m_keyframes.size()) {
        spdlog::warn("[LoopClosureDetector] Invalid keyframe ID: {}", keyframe_id);
        return {};
    }
    
    return SearchLoopCandidates(*m_keyframes[keyframe_id]);
}

std::vector<LoopCandidate> LoopClosureDetector::SearchLoopCandidates(const Keyframe& query_keyframe) {
    std::vector<LoopCandidate> candidates;
    
    if (!m_config.enable_loop_detection) {
        spdlog::info("[LoopClosureDetector] Loop detection disabled");
        return candidates;
    }
    
    m_stats.total_queries++;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        size_t query_id = query_keyframe.id;
        Eigen::Vector3f query_position = query_keyframe.GetPosition();
        
        float min_similarity = 999.0f;
        std::vector<std::pair<float, size_t>> similarity_scores;
        
        int skipped_gap = 0;
        int skipped_distance = 0;
        int compared = 0;
        
        // Search through all keyframes
        for (size_t i = 0; i < m_keyframes.size(); ++i) {
            const auto& candidate_kf = m_keyframes[i];
            size_t candidate_id = candidate_kf->id;
            
            // Check minimum keyframe gap
            if (static_cast<int>(query_id) - static_cast<int>(candidate_id) < m_config.min_keyframe_gap) {
                skipped_gap++;
                continue;
            }
            
            // Check distance constraint
            Eigen::Vector3f candidate_position = candidate_kf->GetPosition();
            float distance = (query_position - candidate_position).norm();
            
            if (distance > m_config.max_search_distance) {
                skipped_distance++;
                continue;
            }
            
            compared++;
            
            // Compare LiDAR Iris features
            int bias = 0;
            float similarity = m_iris->Compare(query_keyframe.iris_feature, 
                                               candidate_kf->iris_feature, &bias);
            
            similarity_scores.push_back({similarity, i});
            min_similarity = std::min(min_similarity, similarity);
        }
        
        // spdlog::info("[LoopClosureDetector] Query KF {}: total={}, skipped_gap={}, skipped_dist={}, compared={}, min_sim={:.4f}, threshold={:.4f}",
        //              query_id, m_keyframes.size(), skipped_gap, skipped_distance, compared, min_similarity, m_config.similarity_threshold);
        
        // Sort by similarity (lower is better)
        std::sort(similarity_scores.begin(), similarity_scores.end());
        
        // Select best candidate that meets threshold
        for (const auto& score_pair : similarity_scores) {
            float similarity = score_pair.first;
            size_t db_index = score_pair.second;
            
            if (similarity > m_config.similarity_threshold) {
                break;  // No more valid candidates
            }
            
            // Get rotational bias
            int bias = 0;
            m_iris->Compare(query_keyframe.iris_feature, 
                           m_keyframes[db_index]->iris_feature, &bias);
            
            LoopCandidate candidate(query_id, m_keyframes[db_index]->id, similarity, bias);
            candidates.push_back(candidate);
            
            // Store in detected loops
            m_detected_loops.push_back(candidate);
            
            break;  // Only take the best candidate
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        m_total_search_time += duration_ms;
        m_stats.avg_search_time_ms = m_total_search_time / m_stats.total_queries;
        m_stats.total_loops_detected = m_detected_loops.size();
        
        if (!candidates.empty()) {
            spdlog::info("[LoopClosureDetector] Loop detected! Keyframe {} <-> {} (score: {:.4f}, bias: {}deg)",
                        candidates[0].query_keyframe_id, candidates[0].match_keyframe_id,
                        candidates[0].similarity_score, candidates[0].bias);
        } else if (m_config.enable_debug_output) {
            spdlog::debug("[LoopClosureDetector] No loop for keyframe {} (min_score: {:.4f}, search: {:.2f}ms)",
                         query_id, min_similarity, duration_ms);
        }
        
    } catch (const std::exception& e) {
        spdlog::error("[LoopClosureDetector] Exception in loop search: {}", e.what());
    }
    
    return candidates;
}

size_t LoopClosureDetector::GetKeyframeCount() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_keyframes.size();
}

std::shared_ptr<Keyframe> LoopClosureDetector::GetKeyframe(size_t id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (id >= m_keyframes.size()) {
        return nullptr;
    }
    
    return m_keyframes[id];
}

std::vector<std::shared_ptr<Keyframe>> LoopClosureDetector::GetAllKeyframes() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_keyframes;
}

void LoopClosureDetector::UpdateConfig(const LoopClosureConfig& config) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_config = config;
    
    spdlog::info("[LoopClosureDetector] Configuration updated:");
    spdlog::info("  - enable_loop_detection: {}", m_config.enable_loop_detection);
    spdlog::info("  - similarity_threshold: {:.3f}", m_config.similarity_threshold);
    spdlog::info("  - min_keyframe_gap: {}", m_config.min_keyframe_gap);
    spdlog::info("  - max_search_distance: {:.1f}m", m_config.max_search_distance);
}

void LoopClosureDetector::Clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    m_keyframes.clear();
    m_detected_loops.clear();
    m_has_keyframe = false;
    m_last_keyframe_position = Eigen::Vector3f::Zero();
    m_last_keyframe_rotation = Eigen::Matrix3f::Identity();
    
    m_stats = Statistics();
    m_total_feature_time = 0.0;
    m_total_search_time = 0.0;
    
    spdlog::info("[LoopClosureDetector] Database cleared");
}

LoopClosureDetector::Statistics LoopClosureDetector::GetStatistics() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_stats;
}

SimplePointCloud LoopClosureDetector::ConvertToSimpleCloud(const PointCloudPtr& cloud) {
    SimplePointCloud simple_cloud;
    
    if (!cloud || cloud->empty()) {
        return simple_cloud;
    }
    
    simple_cloud.reserve(cloud->size());
    
    for (const auto& point : *cloud) {
        simple_cloud.emplace_back(point.x, point.y, point.z);
    }
    
    return simple_cloud;
}

LidarIris::FeatureDesc LoopClosureDetector::ExtractIrisFeature(const PointCloudPtr& cloud) {
    // Convert to SimplePointCloud format
    SimplePointCloud simple_cloud = ConvertToSimpleCloud(cloud);
    
    // Generate LiDAR Iris image
    cv::Mat1b iris_image = LidarIris::GetIris(simple_cloud);
    
    // Extract feature descriptor
    LidarIris::FeatureDesc feature = m_iris->GetFeature(iris_image);
    
    return feature;
}

} // namespace lio
