/**
 * @file      lio_player.cpp
 * @brief     Main application for LiDAR-Inertial Odometry player
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "LIOViewer.h"
#include "Estimator.h"
#include "ConfigUtils.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <spdlog/spdlog.h>

namespace lio {

/**
 * @brief LiDAR scan data
 */
struct LiDARData {
    double timestamp;
    int scan_index;  // Index for PLY filename (000000.ply, 000001.ply, ...)
};

/**
 * @brief Load PLY point cloud file (binary little endian format)
 */
bool LoadPLYPointCloud(const std::string& ply_path, PointCloudPtr& cloud) {
    std::ifstream file(ply_path, std::ios::binary);
    if (!file.is_open()) {
        spdlog::error("[LIO Player] Cannot open PLY file: {}", ply_path);
        return false;
    }
    
    cloud = std::make_shared<PointCloud>();
    
    // Read ASCII header
    std::string line;
    int num_vertices = 0;
    bool binary_format = false;
    
    while (std::getline(file, line)) {
        if (line.find("format binary_little_endian") != std::string::npos) {
            binary_format = true;
        }
        if (line.find("element vertex") != std::string::npos) {
            std::istringstream iss(line);
            std::string dummy1, dummy2;
            iss >> dummy1 >> dummy2 >> num_vertices;
        }
        if (line == "end_header") {
            break;
        }
    }
    
    if (!binary_format) {
        spdlog::error("[LIO Player] Only binary_little_endian PLY format is supported");
        return false;
    }
    
    if (num_vertices <= 0) {
        spdlog::error("[LIO Player] Invalid number of vertices: {}", num_vertices);
        return false;
    }
    
    // Read binary data: x, y, z (float), intensity (float), offset_time (uint32 nanoseconds)
    struct PLYPoint {
        float x, y, z;
        float intensity;
        uint32_t offset_time_ns;  // Nanoseconds (0 ~ 100ms = 0 ~ 100,000,000 ns)
    };
    
    for (int i = 0; i < num_vertices; ++i) {
        PLYPoint ply_point;
        file.read(reinterpret_cast<char*>(&ply_point), sizeof(PLYPoint));
        
        if (!file) {
            spdlog::warn("[LIO Player] Failed to read point {} from PLY file", i);
            break;
        }
        
        Point3D point;
        point.x = ply_point.x;
        point.y = ply_point.y;
        point.z = ply_point.z;
        point.intensity = ply_point.intensity;
        // Convert offset_time from nanoseconds to seconds (0 ~ 0.1 sec)
        point.offset_time = static_cast<float>(static_cast<double>(ply_point.offset_time_ns) / 1e9);
        cloud->push_back(point);
    }
    
    file.close();
    
    if (cloud->empty()) {
        spdlog::error("[LIO Player] No points loaded from PLY file");
        return false;
    }
    
    return true;
}

/**
 * @brief Sensor data type enum
 */
enum class SensorType {
    IMU,
    LIDAR
};

/**
 * @brief Combined sensor event for time-ordered playback
 */
struct SensorEvent {
    SensorType type;
    double timestamp;
    size_t data_index;
};

/**
 * @brief Load IMU data from CSV file
 */
bool LoadIMUData(const std::string& csv_path, std::vector<IMUData>& imu_data) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        spdlog::error("[LIO Player] Cannot open IMU CSV file: {}", csv_path);
        return false;
    }
    
    imu_data.clear();
    
    // Skip header line
    std::string header_line;
    std::getline(file, header_line);
    
    // Read data lines
    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // Parse CSV: timestamp,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z
        std::stringstream ss(line);
        std::string token;
        std::vector<double> values;
        
        while (std::getline(ss, token, ',')) {
            try {
                values.push_back(std::stod(token));
            } catch (const std::exception& e) {
                spdlog::warn("[LIO Player] Failed to parse value '{}' at line {}", token, line_count);
                break;
            }
        }
        
        if (values.size() == 7) {
            double timestamp = values[0];
            Eigen::Vector3f gyr(static_cast<float>(values[1]),
                               static_cast<float>(values[2]),
                               static_cast<float>(values[3]));
            Eigen::Vector3f acc(static_cast<float>(values[4]),
                               static_cast<float>(values[5]),
                               static_cast<float>(values[6]));
            imu_data.emplace_back(timestamp, acc, gyr);
        }
        
        line_count++;
    }
    
    file.close();
    
    if (imu_data.empty()) {
        spdlog::error("[LIO Player] No IMU data loaded");
        return false;
    }
    
    spdlog::info("[LIO Player] Loaded {} IMU measurements", imu_data.size());
    spdlog::info("  Time range: {:.6f} - {:.6f} sec", 
                imu_data.front().timestamp, imu_data.back().timestamp);
    
    return true;
}

/**
 * @brief Load LiDAR timestamps from file
 */
bool LoadLiDARTimestamps(const std::string& timestamp_path, std::vector<LiDARData>& lidar_data) {
    std::ifstream file(timestamp_path);
    if (!file.is_open()) {
        spdlog::error("[LIO Player] Cannot open LiDAR timestamp file: {}", timestamp_path);
        return false;
    }
    
    lidar_data.clear();
    
    std::string line;
    int scan_index = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        try {
            double timestamp = std::stod(line);
            LiDARData lidar;
            lidar.timestamp = timestamp;
            lidar.scan_index = scan_index++;
            lidar_data.push_back(lidar);
        } catch (const std::exception& e) {
            spdlog::warn("[LIO Player] Failed to parse LiDAR timestamp: {}", line);
        }
    }
    
    file.close();
    
    if (lidar_data.empty()) {
        spdlog::error("[LIO Player] No LiDAR timestamps loaded");
        return false;
    }
    
    spdlog::info("[LIO Player] Loaded {} LiDAR scans", lidar_data.size());
    spdlog::info("  Time range: {:.6f} - {:.6f} sec", 
                lidar_data.front().timestamp, lidar_data.back().timestamp);
    
    return true;
}

/**
 * @brief Create time-ordered event sequence
 */
void CreateEventSequence(const std::vector<IMUData>& imu_data,
                        const std::vector<LiDARData>& lidar_data,
                        std::vector<SensorEvent>& events) {
    events.clear();
    
    // Create events for IMU data
    for (size_t i = 0; i < imu_data.size(); ++i) {
        SensorEvent event;
        event.type = SensorType::IMU;
        event.timestamp = imu_data[i].timestamp;
        event.data_index = i;
        events.push_back(event);
    }
    
    // Create events for LiDAR data
    for (size_t i = 0; i < lidar_data.size(); ++i) {
        SensorEvent event;
        event.type = SensorType::LIDAR;
        event.timestamp = lidar_data[i].timestamp;
        event.data_index = i;
        events.push_back(event);
    }
    
    // Sort by timestamp
    std::sort(events.begin(), events.end(), 
              [](const SensorEvent& a, const SensorEvent& b) {
                  return a.timestamp < b.timestamp;
              });
    
    spdlog::info("[LIO Player] Created time-ordered event sequence with {} events", events.size());
}

/**
 * @brief Print playback sequence
 */
void PrintPlaybackSequence(const std::vector<SensorEvent>& events,
                          const std::vector<IMUData>& imu_data,
                          const std::vector<LiDARData>& lidar_data,
                          size_t max_events = 100) {
    if (events.empty()) {
        spdlog::warn("[LIO Player] No events to print");
        return;
    }
    
    spdlog::info("════════════════════════════════════════════════════════════════");
    spdlog::info("                    PLAYBACK SEQUENCE                           ");
    spdlog::info("════════════════════════════════════════════════════════════════");
    
    // Print all events with simple format
    for (size_t i = 0; i < events.size(); ++i) {
        const auto& event = events[i];
        
        if (event.type == SensorType::IMU) {
            spdlog::info("[{:6d}] IMU   @ {:.6f}", i, event.timestamp);
        } else {
            spdlog::info("[{:6d}] LIDAR @ {:.6f}", i, event.timestamp);
        }
    }
    
    spdlog::info("════════════════════════════════════════════════════════════════");
    
    // Print statistics
    size_t imu_count = 0;
    size_t lidar_count = 0;
    for (const auto& event : events) {
        if (event.type == SensorType::IMU) {
            imu_count++;
        } else {
            lidar_count++;
        }
    }
    
    spdlog::info("Statistics:");
    spdlog::info("  Total events: {}", events.size());
    spdlog::info("  IMU events: {} ({:.1f}%)", imu_count, 100.0 * imu_count / events.size());
    spdlog::info("  LiDAR events: {} ({:.1f}%)", lidar_count, 100.0 * lidar_count / events.size());
    spdlog::info("  Time span: {:.3f} seconds", events.back().timestamp - events.front().timestamp);
}

} // namespace lio

int main(int argc, char** argv) {
    // Set log level
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    
    // Parse arguments: config_path is required, dataset_path is optional
    if (argc < 2) {
        spdlog::error("Usage: {} <config_path> [dataset_path] [--headless]", argv[0]);
        spdlog::info("Example: {} ../config/avia.yaml /home/eugene/data/R3LIVE/hku_main_building", argv[0]);
        spdlog::info("Example (headless): {} ../config/avia.yaml /path/to/dataset --headless", argv[0]);
        spdlog::info("");
        spdlog::info("Arguments:");
        spdlog::info("  config_path   : Path to YAML configuration file (required)");
        spdlog::info("  dataset_path  : Path to dataset directory (optional, overrides config)");
        spdlog::info("  --headless    : Run without viewer GUI (optional)");
        return 1;
    }
    
    // Check for headless mode
    bool headless_mode = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--headless") {
            headless_mode = true;
            break;
        }
    }
    
    spdlog::info("════════════════════════════════════════════════════════════════");
    spdlog::info("       LiDAR-Inertial Odometry Player - R3LIVE Dataset          ");
    spdlog::info("════════════════════════════════════════════════════════════════");
    
    // Load configuration
    lio::LIOConfig config;
    std::string config_path = argv[1];
    
    if (!lio::LoadConfig(config_path, config)) {
        spdlog::error("Failed to load configuration from: {}", config_path);
        return 1;
    }
    
    lio::PrintConfig(config);
    spdlog::info("");
    
    // Override dataset path from command line if provided
    if (argc >= 3) {
        config.dataset_path = argv[2];
        config.lidar_folder = config.dataset_path + "/lidar";
        config.imu_csv_path = config.dataset_path + "/imu_data.csv";
        config.lidar_timestamps_path = config.dataset_path + "/lidar_timestamps.txt";
        spdlog::info("Dataset path overridden from command line: {}", config.dataset_path);
        spdlog::info("");
    } else if (config.dataset_path.empty()) {
        spdlog::error("No dataset path provided!");
        spdlog::info("Please provide dataset path in config file or as command line argument");
        spdlog::info("Usage: {} <config_path> [dataset_path]", argv[0]);
        return 1;
    }
    
    std::string dataset_path = config.dataset_path;
    std::string lidar_folder = config.lidar_folder;
    
    // Construct file paths
    std::string imu_csv_path = config.imu_csv_path;
    std::string lidar_ts_path = config.lidar_timestamps_path;
    
    // Load data
    std::vector<lio::IMUData> imu_data;
    std::vector<lio::LiDARData> lidar_data;
    std::vector<lio::SensorEvent> events;
    
    if (!lio::LoadIMUData(imu_csv_path, imu_data)) {
        return 1;
    }
    
    if (!lio::LoadLiDARTimestamps(lidar_ts_path, lidar_data)) {
        return 1;
    }
    
    // Create time-ordered event sequence
    lio::CreateEventSequence(imu_data, lidar_data, events);
    
    spdlog::info("");
    spdlog::info("Statistics:");
    spdlog::info("  Total events: {}", events.size());
    spdlog::info("  IMU measurements: {}", imu_data.size());
    spdlog::info("  LiDAR scans: {}", lidar_data.size());
    spdlog::info("  Time span: {:.3f} seconds", events.back().timestamp - events.front().timestamp);
    
    // Initialize viewer (skip if headless mode)
    spdlog::info("");
    lio::LIOViewer viewer;
    
    if (!headless_mode) {
        spdlog::info("Initializing viewer...");
        if (!viewer.Initialize(config.viewer.window_width, config.viewer.window_height)) {
            spdlog::error("Failed to initialize viewer");
            return 1;
        }
        
        // Set viewer parameters from config
        viewer.SetShowPointCloud(config.viewer.show_point_cloud);
        viewer.SetShowTrajectory(config.viewer.show_trajectory);
        viewer.SetShowCoordinateFrame(config.viewer.show_coordinate_frame);
        viewer.SetShowMap(config.viewer.show_map);
        viewer.SetShowVoxelCubes(config.viewer.show_voxel_cubes);
        viewer.SetAutoPlayback(config.viewer.auto_playback);
        
        spdlog::info("Viewer initialized successfully!");
    } else {
        spdlog::info("Running in HEADLESS mode (no GUI viewer)");
    }
    spdlog::info("");
    
    // Initialize Estimator with gravity initialization
    spdlog::info("Initializing LIO Estimator...");
    lio::Estimator estimator;
    
    // Configure estimator parameters from config
    estimator.m_params.voxel_size = config.estimator.voxel_size;
    estimator.m_params.max_correspondences = config.estimator.max_correspondences;
    estimator.m_params.max_correspondence_distance = config.estimator.max_correspondence_distance;
    estimator.m_params.max_iterations = config.estimator.max_iterations;
    estimator.m_params.convergence_threshold = config.estimator.convergence_threshold;
    estimator.m_params.enable_undistortion = config.estimator.enable_undistortion;
    estimator.m_params.max_map_distance = config.estimator.max_distance;
    estimator.m_params.max_voxel_hit_count = config.estimator.max_voxel_hit_count;
    estimator.m_params.voxel_hierarchy_factor = config.estimator.voxel_hierarchy_factor;
    estimator.m_params.frustum_fov_horizontal = config.estimator.frustum_fov_horizontal;
    estimator.m_params.frustum_fov_vertical = config.estimator.frustum_fov_vertical;
    estimator.m_params.frustum_max_range = config.estimator.frustum_max_range;
    estimator.m_params.keyframe_translation_threshold = config.estimator.keyframe_translation_threshold;
    estimator.m_params.keyframe_rotation_threshold = config.estimator.keyframe_rotation_threshold;
    estimator.m_params.scan_planarity_threshold = config.estimator.scan_planarity_threshold;
    estimator.m_params.map_planarity_threshold = config.estimator.map_planarity_threshold;
    
    // Configure IMU noise parameters from config
    estimator.m_params.gyr_noise_std = config.imu.gyro_noise_density;
    estimator.m_params.acc_noise_std = config.imu.acc_noise_density;
    estimator.m_params.gyr_bias_noise_std = config.imu.gyro_bias_random_walk;
    estimator.m_params.acc_bias_noise_std = config.imu.acc_bias_random_walk;
    
    // Configure extrinsics from config
    estimator.m_params.R_il = config.extrinsics.R_il.cast<float>();
    estimator.m_params.t_il = config.extrinsics.t_il.cast<float>();
    
    // Configure gravity from config
    estimator.m_params.gravity = config.imu.gravity.cast<float>();
    
    // Update process noise matrix with new IMU parameters
    estimator.UpdateProcessNoise();
    
    spdlog::info("[Estimator] Configured from YAML:");
    spdlog::info("  Voxel size: {:.2f} m", estimator.m_params.voxel_size);
    spdlog::info("  Frustum FOV: {:.1f}° × {:.1f}°", 
                 estimator.m_params.frustum_fov_horizontal, 
                 estimator.m_params.frustum_fov_vertical);
    spdlog::info("  Frustum range: {:.1f} m", estimator.m_params.frustum_max_range);
    spdlog::info("  Keyframe thresholds: translation={:.2f}m, rotation={:.1f}°", 
                 estimator.m_params.keyframe_translation_threshold,
                 estimator.m_params.keyframe_rotation_threshold);
    spdlog::info("  IMU noise: gyr={:.4f} rad/s, acc={:.4f} m/s²",
                 estimator.m_params.gyr_noise_std,
                 estimator.m_params.acc_noise_std);
    spdlog::info("  IMU bias random walk: gyr={:.6f} rad/s², acc={:.6f} m/s³",
                 estimator.m_params.gyr_bias_noise_std,
                 estimator.m_params.acc_bias_noise_std);
    spdlog::info("  Extrinsics t_il: [{:.5f}, {:.5f}, {:.5f}]",
                 estimator.m_params.t_il.x(), 
                 estimator.m_params.t_il.y(), 
                 estimator.m_params.t_il.z());
    
    // Collect first N IMU samples for gravity initialization (from config)
    std::vector<lio::IMUData> init_imu_buffer;
    int init_samples = std::min(config.estimator.init_imu_samples, static_cast<int>(imu_data.size()));
    
    for (int i = 0; i < init_samples; ++i) {
        const auto& imu = imu_data[i];
        // IMUData already has Vector3f members, just copy directly
        init_imu_buffer.push_back(imu);
    }
    
    // Perform gravity initialization
    if (!estimator.GravityInitialization(init_imu_buffer)) {
        spdlog::error("Failed to initialize estimator with gravity alignment");
        return 1;
    }
    
    spdlog::info("Estimator initialized successfully!");
    spdlog::info("");
    
    // Skip events before gravity initialization timestamp
    double init_end_time = init_imu_buffer.back().timestamp;
    size_t start_event_idx = 0;
    for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].timestamp > init_end_time) {
            start_event_idx = i;
            break;
        }
    }
    
    spdlog::info("Skipping first {} events (before gravity initialization)", start_event_idx);
    spdlog::info("Starting playback from t={:.3f}s...", events[start_event_idx].timestamp - events[0].timestamp);
    spdlog::info("Total LiDAR scans to process: {}", 
                 std::count_if(events.begin() + start_event_idx, events.end(), 
                              [](const lio::SensorEvent& e) { return e.type == lio::SensorType::LIDAR; }));
    spdlog::info("Close viewer window to quit");
    spdlog::info("");
    
    // Playback parameters from config
    double playback_speed = config.playback.playback_speed;  // Always use config value
    double start_time = events[start_event_idx].timestamp;
    
    // Current state
    Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();
    int frame_count = 0;
    int lidar_frame_count = 0;
    int last_progress_frame = 0;
    
    // Store trajectory with timestamps
    std::vector<std::pair<double, lio::State>> trajectory_with_timestamps;
    
    // Timing measurement for processing
    auto processing_start = std::chrono::high_resolution_clock::now();
    double total_lidar_processing_time = 0.0;  // milliseconds
    
    // Playback loop
    auto playback_start = std::chrono::steady_clock::now();
    bool was_auto_playback_enabled = !headless_mode && viewer.IsAutoPlaybackEnabled();
    
    for (size_t event_idx = start_event_idx; event_idx < events.size() && (headless_mode || !viewer.ShouldClose()); ++event_idx) {
        const auto& event = events[event_idx];
        
        // Check if auto playback state changed (turned back on after being off) - only in GUI mode
        if (!headless_mode) {
            bool is_auto_playback_enabled = viewer.IsAutoPlaybackEnabled();
            if (is_auto_playback_enabled && !was_auto_playback_enabled) {
                // Reset playback timer when auto playback is re-enabled
                playback_start = std::chrono::steady_clock::now();
                start_time = event.timestamp;  // Reset start time to current event
                spdlog::info("Auto playback re-enabled, resetting timer at event {}", event_idx);
            }
            was_auto_playback_enabled = is_auto_playback_enabled;
        }
        
        // Calculate target playback time
        double event_time = event.timestamp - start_time;
        double target_time = event_time / playback_speed;
        
        // Wait until it's time to process this event (only in GUI mode with auto playback)
        // In headless mode, skip all timing delays and process as fast as possible
        if (!headless_mode && viewer.IsAutoPlaybackEnabled()) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - playback_start).count();
            double wait_time = target_time - elapsed;
            
            if (wait_time > 0) {
                std::this_thread::sleep_for(std::chrono::duration<double>(wait_time));
            }
        } else if (!headless_mode && event.type == lio::SensorType::LIDAR) {
            // In step-by-step mode (GUI only), wait for step forward request on LiDAR frames
            // Check auto playback state inside the loop to allow immediate exit
            while (!viewer.WasStepForwardRequested() && 
                   !viewer.ShouldClose() && 
                   !viewer.IsAutoPlaybackEnabled()) {  // Exit if auto playback is re-enabled
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        // In headless mode: no waiting, process as fast as possible
        
        if (!headless_mode && viewer.ShouldClose()) {
            break;
        }
        
        if (event.type == lio::SensorType::IMU) {
            // Process IMU data
            const auto& imu = imu_data[event.data_index];
            
            // Process with estimator
            estimator.ProcessIMU(imu);
            
            // Get current state to extract biases and gravity
            lio::State state = estimator.GetCurrentState();
            
            // Compute gravity-compensated acceleration (world frame)
            // Update IMU bias in viewer (only in GUI mode)
            if (!headless_mode) {
                viewer.UpdateIMUBias(state.m_gyro_bias, state.m_acc_bias);
            }
            
        } else {
            // Process LiDAR data
            const auto& lidar = lidar_data[event.data_index];
            
            // Construct PLY file path (000000.ply, 000001.ply, ...)
            char ply_filename[32];
            snprintf(ply_filename, sizeof(ply_filename), "%06d.ply", lidar.scan_index);
            std::string ply_path = lidar_folder + "/" + std::string(ply_filename);
            
            // Load point cloud
            lio::PointCloudPtr cloud;
            if (lio::LoadPLYPointCloud(ply_path, cloud)) {
                lidar_frame_count++;
                
                // Measure processing time for this frame
                auto frame_start = std::chrono::high_resolution_clock::now();
                
                // Process with LIO estimator
                estimator.ProcessLidar(lio::LidarData(lidar.timestamp, cloud));
                
                // Measure frame processing time
                auto frame_end = std::chrono::high_resolution_clock::now();
                double frame_time_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
                total_lidar_processing_time += frame_time_ms;
                
                // Get current state from estimator
                lio::State current_state = estimator.GetCurrentState();
                
                // Store trajectory with timestamp
                trajectory_with_timestamps.push_back({lidar.timestamp, current_state});
                
                // Convert state to pose matrix for visualization
                Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();
                current_pose.block<3,3>(0,0) = current_state.m_rotation;
                current_pose.block<3,1>(0,3) = current_state.m_position;
                
                // Get processed (downsampled + range filtered) cloud for visualization
                lio::PointCloudPtr processed_cloud = estimator.GetProcessedCloud();
                if (processed_cloud && !processed_cloud->empty() && !headless_mode) {
                    viewer.UpdatePointCloud(processed_cloud, current_pose);
                }
                
                if (!headless_mode) {
                    viewer.AddTrajectoryPoint(current_pose);
                    viewer.UpdateStateInfo(lidar_frame_count, 
                        processed_cloud ? processed_cloud->size() : 0);
                    
                    // Update map visualization (choose between point cloud or voxel cubes)
                    lio::PointCloudPtr map_cloud = estimator.GetMapPointCloud();
                    if (map_cloud) {
                        viewer.UpdateMapPointCloud(map_cloud);
                    }
                    
                    // Update voxel map for cube visualization
                    std::shared_ptr<lio::VoxelMap> voxel_map = estimator.GetVoxelMap();
                    if (voxel_map) {
                        viewer.UpdateVoxelMap(voxel_map);
                    }
                }
                
                // Print progress every 100 frames
                if (lidar_frame_count - last_progress_frame >= 100) {
                    auto current_time = std::chrono::high_resolution_clock::now();
                    double elapsed = std::chrono::duration<double>(current_time - processing_start).count();
                    last_progress_frame = lidar_frame_count;
                }
            } else {
                spdlog::warn("Failed to load LiDAR scan: {}", ply_path);
            }
        }
        
        frame_count++;
    }
    
    spdlog::info("");
    spdlog::info("Playback finished!");
    spdlog::info("  Total frames processed: {}", frame_count);
    spdlog::info("  LiDAR frames: {}", lidar_frame_count);
    spdlog::info("");
    
    // Calculate total processing time
    auto processing_end = std::chrono::high_resolution_clock::now();
    double total_time_s = std::chrono::duration<double>(processing_end - processing_start).count();
    
    // Print overall performance statistics
    spdlog::info("════════════════════════════════════════════════════════════════");
    spdlog::info("                  PERFORMANCE STATISTICS                        ");
    spdlog::info("════════════════════════════════════════════════════════════════");
    if (lidar_frame_count > 0) {
        double avg_frame_time_ms = total_lidar_processing_time / lidar_frame_count;
        double avg_fps = 1000.0 / avg_frame_time_ms;
        spdlog::info("  Total processing time:     {:.3f} seconds", total_time_s);
        spdlog::info("  LiDAR frames processed:    {}", lidar_frame_count);
        spdlog::info("  Average frame time:        {:.3f} ms", avg_frame_time_ms);
        spdlog::info("  Average FPS:               {:.1f} Hz", avg_fps);
        spdlog::info("  Total LiDAR proc time:     {:.3f} seconds", total_lidar_processing_time / 1000.0);
    }
    spdlog::info("════════════════════════════════════════════════════════════════");
    spdlog::info("");
    
    // Print processing time statistics
    estimator.PrintProcessingTimeStatistics();
    
    // Save trajectory to TUM format
    std::string traj_output_path = dataset_path + "/trajectory_tum.txt";
    spdlog::info("Saving trajectory to: {}", traj_output_path);
    
    std::ofstream traj_file(traj_output_path);
    if (traj_file.is_open()) {
        // Write TUM format: timestamp x y z qx qy qz qw
        for (const auto& [timestamp, state] : trajectory_with_timestamps) {
            // Extract position and rotation
            const Eigen::Vector3f& position = state.m_position;
            const Eigen::Matrix3f& rotation = state.m_rotation;
            
            // Convert rotation matrix to quaternion
            Eigen::Quaternionf q(rotation);
            
            // Write to file
            traj_file << std::fixed << std::setprecision(6) << timestamp << " "
                     << std::setprecision(9) << position.x() << " " << position.y() << " " << position.z() << " "
                     << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
        
        traj_file.close();
        spdlog::info("✓ Trajectory saved successfully ({} poses)", trajectory_with_timestamps.size());
    } else {
        spdlog::error("Failed to open trajectory file: {}", traj_output_path);
    }
    spdlog::info("");
    
    // Save processing times to file
    std::string proc_time_output_path = dataset_path + "/ours_processing_time.txt";
    spdlog::info("Saving processing times to: {}", proc_time_output_path);
    
    const auto& processing_times = estimator.GetProcessingTimes();
    std::ofstream proc_time_file(proc_time_output_path);
    if (proc_time_file.is_open()) {
        // Write each frame's processing time in milliseconds
        for (double time_ms : processing_times) {
            proc_time_file << std::fixed << std::setprecision(3) << time_ms << "\n";
        }
        
        proc_time_file.close();
        spdlog::info("✓ Processing times saved successfully ({} frames)", processing_times.size());
    } else {
        spdlog::error("Failed to open processing time file: {}", proc_time_output_path);
    }
    spdlog::info("");
    
    if (headless_mode) {
        spdlog::info("Headless playback completed!");
    } else {
        spdlog::info("Viewer will stay open. Close window to quit.");
        
        // Keep viewer open until user closes it
        while (!viewer.ShouldClose()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        viewer.Shutdown();
        spdlog::info("Viewer closed. Exiting...");
    }
    
    return 0;
}
