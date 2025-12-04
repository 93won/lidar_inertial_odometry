/**
 * @file      LoopClosureDetector.h
 * @brief     Loop closure detection using LiDAR Iris features
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-12-04
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include "State.h"
#include "PointCloudUtils.h"
#include "../../thirdparty/LidarIris/LidarIris.h"

#include <memory>
#include <vector>
#include <deque>
#include <mutex>
#include <spdlog/spdlog.h>
#include <Eigen/Dense>

namespace lio {

/**
 * @brief Keyframe structure for loop closure detection
 */
struct Keyframe {
    size_t id;                          ///< Unique keyframe ID
    double timestamp;                   ///< Timestamp of the keyframe
    State state;                        ///< State (pose) at keyframe
    PointCloudPtr cloud;                ///< Point cloud in local frame
    LidarIris::FeatureDesc iris_feature; ///< LiDAR Iris feature descriptor
    
    Keyframe() : id(0), timestamp(0.0) {}
    
    Keyframe(size_t id_, double ts, const State& s, PointCloudPtr pc)
        : id(id_), timestamp(ts), state(s), cloud(pc) {}
    
    /// Get position from state
    Eigen::Vector3f GetPosition() const {
        return state.m_position;
    }
    
    /// Get rotation from state
    Eigen::Matrix3f GetRotation() const {
        return state.m_rotation;
    }
};

/**
 * @brief Loop closure candidate structure
 */
struct LoopCandidate {
    size_t query_keyframe_id;      ///< Current keyframe ID
    size_t match_keyframe_id;      ///< Matched keyframe ID
    float similarity_score;        ///< LiDAR Iris similarity score (lower is better)
    int bias;                      ///< Rotational bias from LiDAR Iris (degrees)
    bool is_valid;                 ///< Whether this candidate passed validation
    
    // Relative pose between keyframes (optional, for pose graph)
    Eigen::Matrix3f relative_rotation;
    Eigen::Vector3f relative_translation;
    
    LoopCandidate() 
        : query_keyframe_id(0), match_keyframe_id(0), 
          similarity_score(999.0f), bias(0), is_valid(false),
          relative_rotation(Eigen::Matrix3f::Identity()),
          relative_translation(Eigen::Vector3f::Zero()) {}
    
    LoopCandidate(size_t query_id, size_t match_id, float score, int rot_bias)
        : query_keyframe_id(query_id), match_keyframe_id(match_id),
          similarity_score(score), bias(rot_bias), is_valid(true),
          relative_rotation(Eigen::Matrix3f::Identity()),
          relative_translation(Eigen::Vector3f::Zero()) {}
};

/**
 * @brief Loop closure detection configuration
 */
struct LoopClosureConfig {
    bool enable_loop_detection = true;       ///< Enable/disable loop detection
    float similarity_threshold = 0.3f;       ///< LiDAR Iris similarity threshold (lower = stricter)
    int min_keyframe_gap = 50;               ///< Minimum gap between keyframes for loop closure
    float max_search_distance = 15.0f;       ///< Maximum distance (meters) to search for loop candidates
    bool enable_debug_output = false;        ///< Enable debug logging
    
    // Keyframe selection parameters
    float keyframe_translation_threshold = 1.0f;  ///< Min translation for new keyframe (meters)
};

/**
 * @brief Loop closure detector using LiDAR Iris features
 * 
 * This class manages keyframes and detects loop closures using
 * LiDAR Iris descriptors for place recognition.
 */
class LoopClosureDetector {
public:
    /**
     * @brief Constructor
     * @param config Loop closure detection configuration
     */
    explicit LoopClosureDetector(const LoopClosureConfig& config = LoopClosureConfig());
    
    /**
     * @brief Destructor
     */
    ~LoopClosureDetector();
    
    /**
     * @brief Check if current state should be a keyframe
     * @param current_state Current state
     * @return true if should create new keyframe
     */
    bool ShouldCreateKeyframe(const State& current_state) const;
    
    /**
     * @brief Add a new keyframe
     * @param timestamp Timestamp of the keyframe
     * @param state State at keyframe
     * @param cloud Point cloud in local (sensor) frame
     * @return Keyframe ID, or -1 if failed
     */
    int AddKeyframe(double timestamp, const State& state, PointCloudPtr cloud);
    
    /**
     * @brief Detect loop closure candidates for the latest keyframe
     * @return Vector of loop closure candidates
     */
    std::vector<LoopCandidate> DetectLoopClosures();
    
    /**
     * @brief Detect loop closure candidates for a specific keyframe
     * @param keyframe_id Keyframe ID to query
     * @return Vector of loop closure candidates
     */
    std::vector<LoopCandidate> DetectLoopClosures(size_t keyframe_id);
    
    /**
     * @brief Get number of stored keyframes
     * @return Number of keyframes in database
     */
    size_t GetKeyframeCount() const;
    
    /**
     * @brief Get keyframe by ID
     * @param id Keyframe ID
     * @return Pointer to keyframe, or nullptr if not found
     */
    std::shared_ptr<Keyframe> GetKeyframe(size_t id) const;
    
    /**
     * @brief Get all keyframes
     * @return Vector of all keyframes
     */
    std::vector<std::shared_ptr<Keyframe>> GetAllKeyframes() const;
    
    /**
     * @brief Get all detected loop closures
     * @return Vector of all loop candidates
     */
    std::vector<LoopCandidate> GetAllLoopClosures() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_detected_loops;
    }
    
    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void UpdateConfig(const LoopClosureConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const LoopClosureConfig& GetConfig() const { return m_config; }
    
    /**
     * @brief Clear all stored keyframes and features
     */
    void Clear();
    
    /**
     * @brief Get statistics
     */
    struct Statistics {
        size_t total_keyframes = 0;
        size_t total_queries = 0;
        size_t total_loops_detected = 0;
        double avg_feature_extraction_time_ms = 0.0;
        double avg_search_time_ms = 0.0;
    };
    
    Statistics GetStatistics() const;

private:
    /**
     * @brief Convert PointCloud to SimplePointCloud format for LiDAR Iris
     * @param cloud Input point cloud
     * @return SimplePointCloud for LiDAR Iris
     */
    SimplePointCloud ConvertToSimpleCloud(const PointCloudPtr& cloud);
    
    /**
     * @brief Extract LiDAR Iris feature from point cloud
     * @param cloud Input point cloud
     * @return LiDAR Iris feature descriptor
     */
    LidarIris::FeatureDesc ExtractIrisFeature(const PointCloudPtr& cloud);
    
    /**
     * @brief Search for loop candidates in the database
     * @param query_keyframe Query keyframe
     * @return Vector of loop candidates
     */
    std::vector<LoopCandidate> SearchLoopCandidates(const Keyframe& query_keyframe);
    
    // Configuration
    LoopClosureConfig m_config;
    
    // LiDAR Iris detector
    std::unique_ptr<LidarIris> m_iris;
    
    // Keyframe database
    std::vector<std::shared_ptr<Keyframe>> m_keyframes;
    
    // Detected loop closures
    std::vector<LoopCandidate> m_detected_loops;
    
    // Last keyframe state for keyframe selection
    Eigen::Vector3f m_last_keyframe_position;
    Eigen::Matrix3f m_last_keyframe_rotation;
    bool m_has_keyframe;
    
    // Thread safety
    mutable std::mutex m_mutex;
    
    // Statistics
    mutable Statistics m_stats;
    double m_total_feature_time = 0.0;
    double m_total_search_time = 0.0;
};

} // namespace lio
