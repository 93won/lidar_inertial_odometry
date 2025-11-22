/**
 * @file      ConfigUtils.cpp
 * @brief     Implementation of configuration utilities
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-19
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "ConfigUtils.h"
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
#include <fstream>

namespace lio {

void SetDefaultConfig(LIOConfig& config) {
    // Dataset paths (will be overridden by command line or config)
    config.dataset_path = "";
    config.lidar_folder = "";
    config.imu_csv_path = "";
    config.lidar_timestamps_path = "";
    
    // Estimator parameters
    config.estimator.enable_undistortion = true;
    config.estimator.scan_duration = 0.1;  // 100ms
    config.estimator.init_imu_samples = 100;
    config.estimator.voxel_size = 0.4f;
    config.estimator.max_correspondences = 500;
    config.estimator.max_correspondence_distance = 1.0;
    config.estimator.max_iterations = 10;
    config.estimator.convergence_threshold = 1e-4;
    config.estimator.scan_planarity_threshold = 0.1;  // Relaxed threshold for input scan downsampling
    config.estimator.map_planarity_threshold = 0.01;  // Strict threshold for VoxelMap surfel
    config.estimator.max_distance = 50.0;
    config.estimator.max_voxel_hit_count = 10;
    config.estimator.voxel_hierarchy_factor = 3;  // Default: 3×3×3 (L1 = 3 × L0)
    config.estimator.frustum_fov_horizontal = 90.0;
    config.estimator.frustum_fov_vertical = 90.0;
    config.estimator.frustum_max_range = 50.0;
    config.estimator.keyframe_translation_threshold = 0.5;  // Default: 0.5 meters
    config.estimator.keyframe_rotation_threshold = 10.0;    // Default: 10 degrees
    
    // Viewer parameters
    config.viewer.window_width = 1920;
    config.viewer.window_height = 1080;
    config.viewer.show_point_cloud = true;
    config.viewer.show_trajectory = true;
    config.viewer.show_coordinate_frame = true;
    config.viewer.show_imu_plots = true;
    config.viewer.show_map = true;
    config.viewer.show_voxel_cubes = false;
    config.viewer.follow_frame = true;
    config.viewer.auto_playback = true;
    config.viewer.point_size = 2.0f;
    config.viewer.trajectory_width = 3.0f;
    config.viewer.coordinate_frame_size = 2.0f;
    
    // Playback parameters
    config.playback.playback_speed = 5.0;
    
    // IMU parameters (typical values for consumer IMU)
    config.imu.gyro_noise_density = 1.75e-4;
    config.imu.acc_noise_density = 1.86e-3;
    config.imu.gyro_bias_random_walk = 1.87e-5;
    config.imu.acc_bias_random_walk = 4.33e-4;
    config.imu.gravity << 0.0, 0.0, -9.81;
    
    // Extrinsics (R3LIVE/Avia dataset default values)
    config.extrinsics.R_il = Eigen::Matrix3d::Identity();
    config.extrinsics.t_il << 0.04165, 0.02326, -0.0284;  // R3LIVE/Avia sensor offset
}

bool LoadConfig(const std::string& config_path, LIOConfig& config) {
    // Set defaults first
    SetDefaultConfig(config);
    
    // Check if file exists
    std::ifstream file(config_path);
    if (!file.good()) {
        spdlog::warn("[ConfigUtils] Config file not found: {}", config_path);
        spdlog::info("[ConfigUtils] Using default configuration");
        return true;  // Return true with defaults
    }
    
    try {
        YAML::Node yaml_config = YAML::LoadFile(config_path);
        
        // Load dataset paths
        if (yaml_config["dataset"]) {
            auto dataset = yaml_config["dataset"];
            if (dataset["path"]) {
                config.dataset_path = dataset["path"].as<std::string>();
                config.lidar_folder = config.dataset_path + "/lidar";
                config.imu_csv_path = config.dataset_path + "/imu_data.csv";
                config.lidar_timestamps_path = config.dataset_path + "/lidar_timestamps.txt";
            }
        }
        
        // Load estimator parameters
        if (yaml_config["estimator"]) {
            auto estimator = yaml_config["estimator"];
            if (estimator["enable_undistortion"]) 
                config.estimator.enable_undistortion = estimator["enable_undistortion"].as<bool>();
            if (estimator["scan_duration"]) 
                config.estimator.scan_duration = estimator["scan_duration"].as<double>();
            if (estimator["init_imu_samples"]) 
                config.estimator.init_imu_samples = estimator["init_imu_samples"].as<int>();
            if (estimator["voxel_size"]) 
                config.estimator.voxel_size = estimator["voxel_size"].as<double>();
            if (estimator["max_correspondences"]) 
                config.estimator.max_correspondences = estimator["max_correspondences"].as<int>();
            if (estimator["max_correspondence_distance"]) 
                config.estimator.max_correspondence_distance = estimator["max_correspondence_distance"].as<double>();
            if (estimator["max_iterations"]) 
                config.estimator.max_iterations = estimator["max_iterations"].as<int>();
            if (estimator["convergence_threshold"]) 
                config.estimator.convergence_threshold = estimator["convergence_threshold"].as<double>();
            if (estimator["max_distance"]) 
                config.estimator.max_distance = estimator["max_distance"].as<double>();
            if (estimator["max_voxel_hit_count"]) 
                config.estimator.max_voxel_hit_count = estimator["max_voxel_hit_count"].as<int>();
            if (estimator["voxel_hierarchy_factor"]) {
                config.estimator.voxel_hierarchy_factor = estimator["voxel_hierarchy_factor"].as<int>();
                // Validate: must be odd number (3, 5, 7, etc.)
                if (config.estimator.voxel_hierarchy_factor % 2 == 0) {
                    spdlog::warn("[Config] voxel_hierarchy_factor must be odd (3, 5, 7, etc.). Using default: 3");
                    config.estimator.voxel_hierarchy_factor = 3;
                }
            }
            if (estimator["frustum_fov_horizontal"]) 
                config.estimator.frustum_fov_horizontal = estimator["frustum_fov_horizontal"].as<double>();
            if (estimator["frustum_fov_vertical"]) 
                config.estimator.frustum_fov_vertical = estimator["frustum_fov_vertical"].as<double>();
            if (estimator["frustum_max_range"]) 
                config.estimator.frustum_max_range = estimator["frustum_max_range"].as<double>();
            if (estimator["keyframe_translation_threshold"]) 
                config.estimator.keyframe_translation_threshold = estimator["keyframe_translation_threshold"].as<double>();
            if (estimator["keyframe_rotation_threshold"]) 
                config.estimator.keyframe_rotation_threshold = estimator["keyframe_rotation_threshold"].as<double>();
            if (estimator["scan_planarity_threshold"]) 
                config.estimator.scan_planarity_threshold = estimator["scan_planarity_threshold"].as<double>();
            if (estimator["map_planarity_threshold"]) 
                config.estimator.map_planarity_threshold = estimator["map_planarity_threshold"].as<double>();
        }
        
        // Load viewer parameters
        if (yaml_config["viewer"]) {
            auto viewer = yaml_config["viewer"];
            if (viewer["window_width"]) 
                config.viewer.window_width = viewer["window_width"].as<int>();
            if (viewer["window_height"]) 
                config.viewer.window_height = viewer["window_height"].as<int>();
            if (viewer["show_point_cloud"]) 
                config.viewer.show_point_cloud = viewer["show_point_cloud"].as<bool>();
            if (viewer["show_trajectory"]) 
                config.viewer.show_trajectory = viewer["show_trajectory"].as<bool>();
            if (viewer["show_coordinate_frame"]) 
                config.viewer.show_coordinate_frame = viewer["show_coordinate_frame"].as<bool>();
            if (viewer["show_imu_plots"]) 
                config.viewer.show_imu_plots = viewer["show_imu_plots"].as<bool>();
            if (viewer["show_map"]) 
                config.viewer.show_map = viewer["show_map"].as<bool>();
            if (viewer["show_voxel_cubes"]) 
                config.viewer.show_voxel_cubes = viewer["show_voxel_cubes"].as<bool>();
            if (viewer["follow_frame"]) 
                config.viewer.follow_frame = viewer["follow_frame"].as<bool>();
            if (viewer["auto_playback"]) 
                config.viewer.auto_playback = viewer["auto_playback"].as<bool>();
            if (viewer["point_size"]) 
                config.viewer.point_size = viewer["point_size"].as<float>();
            if (viewer["trajectory_width"]) 
                config.viewer.trajectory_width = viewer["trajectory_width"].as<float>();
            if (viewer["coordinate_frame_size"]) 
                config.viewer.coordinate_frame_size = viewer["coordinate_frame_size"].as<float>();
        }
        
        // Load playback parameters
        if (yaml_config["playback"]) {
            auto playback = yaml_config["playback"];
            if (playback["playback_speed"]) 
                config.playback.playback_speed = playback["playback_speed"].as<double>();
        }
        
        // Load IMU parameters
        if (yaml_config["imu"]) {
            auto imu = yaml_config["imu"];
            if (imu["gyro_noise_density"]) 
                config.imu.gyro_noise_density = imu["gyro_noise_density"].as<double>();
            if (imu["acc_noise_density"]) 
                config.imu.acc_noise_density = imu["acc_noise_density"].as<double>();
            if (imu["gyro_bias_random_walk"]) 
                config.imu.gyro_bias_random_walk = imu["gyro_bias_random_walk"].as<double>();
            if (imu["acc_bias_random_walk"]) 
                config.imu.acc_bias_random_walk = imu["acc_bias_random_walk"].as<double>();
            if (imu["gravity"]) {
                auto gravity = imu["gravity"];
                config.imu.gravity << gravity[0].as<double>(), 
                                      gravity[1].as<double>(), 
                                      gravity[2].as<double>();
            }
        }
        
        // Load extrinsics
        if (yaml_config["extrinsics"]) {
            auto extrinsics = yaml_config["extrinsics"];
            if (extrinsics["R_il"]) {
                auto R = extrinsics["R_il"];
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        config.extrinsics.R_il(i, j) = R[i][j].as<double>();
                    }
                }
            }
            if (extrinsics["t_il"]) {
                auto t = extrinsics["t_il"];
                config.extrinsics.t_il << t[0].as<double>(), 
                                          t[1].as<double>(), 
                                          t[2].as<double>();
            }
        }
        
        spdlog::info("[ConfigUtils] Successfully loaded config from: {}", config_path);
        return true;
        
    } catch (const YAML::Exception& e) {
        spdlog::error("[ConfigUtils] YAML parsing error: {}", e.what());
        spdlog::info("[ConfigUtils] Using default configuration");
        return true;  // Return true with defaults
    } catch (const std::exception& e) {
        spdlog::error("[ConfigUtils] Error loading config: {}", e.what());
        spdlog::info("[ConfigUtils] Using default configuration");
        return true;  // Return true with defaults
    }
}

void PrintConfig(const LIOConfig& config) {
    spdlog::info("========================================");
    spdlog::info("       LIO Configuration");
    spdlog::info("========================================");
    
    spdlog::info("Dataset:");
    spdlog::info("  Path: {}", config.dataset_path);
    
    spdlog::info("Estimator:");
    spdlog::info("  Enable Undistortion: {}", config.estimator.enable_undistortion);
    spdlog::info("  Scan Duration: {:.3f} s", config.estimator.scan_duration);
    spdlog::info("  Voxel Size: {:.2f} m", config.estimator.voxel_size);
    spdlog::info("  Max Correspondences: {}", config.estimator.max_correspondences);
    spdlog::info("  Max Correspondence Distance: {:.2f} m", config.estimator.max_correspondence_distance);
    
    spdlog::info("Viewer:");
    spdlog::info("  Window Size: {}x{}", config.viewer.window_width, config.viewer.window_height);
    spdlog::info("  Follow Frame: {}", config.viewer.follow_frame);
    spdlog::info("  Auto Playback: {}", config.viewer.auto_playback);
    
    spdlog::info("Playback:");
    spdlog::info("  Speed: {:.1f}x", config.playback.playback_speed);
    
    spdlog::info("========================================");
}

} // namespace lio
