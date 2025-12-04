/**
 * @file      ConfigUtils.h
 * @brief     Configuration utilities for YAML config file parsing
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-19
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef CONFIG_UTILS_H
#define CONFIG_UTILS_H

#include <string>
#include <Eigen/Dense>

namespace lio {

/**
 * @brief Configuration parameters for LIO system
 */
struct LIOConfig {
    // Dataset paths
    std::string dataset_path;
    std::string lidar_folder;
    std::string imu_csv_path;
    std::string lidar_timestamps_path;
    
    // Estimator parameters
    struct EstimatorParams {
        bool enable_undistortion;
        double scan_duration;           // LiDAR scan duration (seconds)
        int init_imu_samples;           // Number of IMU samples for gravity initialization
        double voxel_size;              // Voxel size for VoxelMap (meters)
        int max_correspondences;        // Maximum number of correspondences
        double max_correspondence_distance; // Maximum distance for correspondence (meters)
        int max_iterations;             // Maximum ICP iterations
        double convergence_threshold;   // Convergence threshold for ICP
        
        // Planarity filtering parameters
        double scan_planarity_threshold;  // Planarity threshold for input scan downsampling (relaxed)
        double map_planarity_threshold;   // Planarity threshold for VoxelMap surfel creation (strict)
        double point_to_surfel_threshold; // Max distance from point to surfel plane (meters)
        int min_surfel_inliers;           // Minimum inlier count for valid surfel
        double min_linearity_ratio;       // Min σ₁/σ₀ ratio to reject edges (higher = stricter)
        
        // Local map parameters
        double min_distance;            // Minimum distance for lidar filtering (meters)
        double max_distance;            // Maximum distance for map points (meters)
        double map_box_multiplier;      // Map box size = max_distance × multiplier
        int voxel_hierarchy_factor;     // L1 voxel factor: L1 = factor × L0 (3 = 3×3×3, 5 = 5×5×5, etc.)
        double frustum_fov_horizontal;  // Frustum FOV horizontal (degrees)
        double frustum_fov_vertical;    // Frustum FOV vertical (degrees)
        double frustum_max_range;       // Maximum range for frustum culling (meters)
        
        // Keyframe parameters (distance/rotation based)
        double keyframe_translation_threshold;  // meters - triggers keyframe when moved > threshold
        double keyframe_rotation_threshold;     // degrees - triggers keyframe when rotated > threshold
        
        // Temporal bin downsampling
        int temporal_bins;                      // Number of bins for temporal downsampling (0 = use voxel)
        bool temporal_then_voxel;               // Apply voxel downsample after temporal bin (default: false)
    } estimator;
    
    // Viewer parameters
    struct ViewerParams {
        int window_width;
        int window_height;
        bool show_point_cloud;
        bool show_trajectory;
        bool show_coordinate_frame;
        bool show_imu_plots;
        bool show_map;
        bool show_voxel_cubes;
        bool follow_frame;
        bool auto_playback;
        float point_size;
        float trajectory_width;
        float coordinate_frame_size;
    } viewer;
    
    // Playback parameters
    struct PlaybackParams {
        double playback_speed;          // Playback speed multiplier
    } playback;
    
    // IMU parameters
    struct IMUParams {
        double gyr_cov;                 // Gyroscope noise covariance [rad/s]
        double acc_cov;                 // Accelerometer noise covariance [m/s^2]
        double b_gyr_cov;               // Gyroscope bias covariance [rad/s]
        double b_acc_cov;               // Accelerometer bias covariance [m/s^2]
        Eigen::Vector3d gravity;        // Gravity vector (m/s^2)
    } imu;
    
    // Extrinsics (LiDAR to IMU)
    struct ExtrinsicsParams {
        Eigen::Matrix3d R_il;           // Rotation from LiDAR to IMU
        Eigen::Vector3d t_il;           // Translation from LiDAR to IMU (meters)
    } extrinsics;
    
    // Loop closure parameters
    struct LoopClosureParams {
        bool enable;                        // Enable loop closure detection
        float similarity_threshold;         // LidarIris similarity threshold (lower = stricter)
        int min_keyframe_gap;               // Minimum keyframe gap for loop candidates
        float max_search_distance;          // Maximum distance to search for loop candidates (meters)
        float keyframe_translation_threshold; // Distance threshold for keyframe creation (meters)
    } loop_closure;
};

/**
 * @brief Load configuration from YAML file
 * @param config_path Path to YAML configuration file
 * @param config Output configuration structure
 * @return True if successful, false otherwise
 */
bool LoadConfig(const std::string& config_path, LIOConfig& config);

/**
 * @brief Set default configuration values
 * @param config Output configuration structure with default values
 */
void SetDefaultConfig(LIOConfig& config);

/**
 * @brief Print configuration to console
 * @param config Configuration to print
 */
void PrintConfig(const LIOConfig& config);

} // namespace lio

#endif // CONFIG_UTILS_H
