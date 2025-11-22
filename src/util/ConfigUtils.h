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
        
        // Local map parameters
        double max_distance;            // Maximum distance for map points (meters)
        int max_voxel_hit_count;        // Maximum hit count for voxel occupancy (1-N)
        int voxel_hierarchy_factor;     // L1 voxel factor: L1 = factor × L0 (3 = 3×3×3, 5 = 5×5×5, etc.)
        double frustum_fov_horizontal;  // Frustum FOV horizontal (degrees)
        double frustum_fov_vertical;    // Frustum FOV vertical (degrees)
        double frustum_max_range;       // Maximum range for frustum culling (meters)
        
        // Keyframe parameters (distance/rotation based)
        double keyframe_translation_threshold;  // meters - triggers keyframe when moved > threshold
        double keyframe_rotation_threshold;     // degrees - triggers keyframe when rotated > threshold
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
        double gyro_noise_density;      // rad/s/sqrt(Hz)
        double acc_noise_density;       // m/s^2/sqrt(Hz)
        double gyro_bias_random_walk;   // rad/s^2/sqrt(Hz)
        double acc_bias_random_walk;    // m/s^3/sqrt(Hz)
        Eigen::Vector3d gravity;        // Gravity vector (m/s^2)
    } imu;
    
    // Extrinsics (LiDAR to IMU)
    struct ExtrinsicsParams {
        Eigen::Matrix3d R_il;           // Rotation from LiDAR to IMU
        Eigen::Vector3d t_il;           // Translation from LiDAR to IMU (meters)
    } extrinsics;
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
