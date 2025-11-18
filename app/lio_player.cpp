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
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <spdlog/spdlog.h>

namespace lio {

/**
 * @brief IMU measurement data
 */
struct IMUData {
    double timestamp;
    float gyro_x, gyro_y, gyro_z;
    float acc_x, acc_y, acc_z;
};

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
            IMUData imu;
            imu.timestamp = values[0];
            imu.gyro_x = static_cast<float>(values[1]);
            imu.gyro_y = static_cast<float>(values[2]);
            imu.gyro_z = static_cast<float>(values[3]);
            imu.acc_x = static_cast<float>(values[4]);
            imu.acc_y = static_cast<float>(values[5]);
            imu.acc_z = static_cast<float>(values[6]);
            imu_data.push_back(imu);
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
    
    // Parse arguments
    if (argc < 2) {
        spdlog::error("Usage: {} <dataset_path>", argv[0]);
        spdlog::info("Example: {} /home/eugene/data/R3LIVE/hku_main_building", argv[0]);
        return 1;
    }
    
    std::string dataset_path = argv[1];
    std::string lidar_folder = dataset_path + "/lidar";
    
    spdlog::info("════════════════════════════════════════════════════════════════");
    spdlog::info("       LiDAR-Inertial Odometry Player - R3LIVE Dataset          ");
    spdlog::info("════════════════════════════════════════════════════════════════");
    
    // Construct file paths
    std::string imu_csv_path = dataset_path + "/imu_data.csv";
    std::string lidar_ts_path = dataset_path + "/lidar_timestamps.txt";
    
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
    
    // Initialize viewer
    spdlog::info("");
    spdlog::info("Initializing viewer...");
    lio::LIOViewer viewer;
    if (!viewer.Initialize(1920, 1080)) {
        spdlog::error("Failed to initialize viewer");
        return 1;
    }
    
    spdlog::info("Viewer initialized successfully!");
    spdlog::info("");
    spdlog::info("Starting playback...");
    spdlog::info("Close viewer window to quit");
    spdlog::info("");
    
    // Playback parameters
    double playback_speed = 5.0;  // 5x speed for faster playback
    double start_time = events.front().timestamp;
    
    // Current state
    Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();
    int frame_count = 0;
    int lidar_frame_count = 0;
    
    // Playback loop
    auto playback_start = std::chrono::steady_clock::now();
    
    for (size_t event_idx = 0; event_idx < events.size() && !viewer.ShouldClose(); ++event_idx) {
        const auto& event = events[event_idx];
        
        // Calculate target playback time
        double event_time = event.timestamp - start_time;
        double target_time = event_time / playback_speed;
        
        // Wait until it's time to process this event
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - playback_start).count();
        double wait_time = target_time - elapsed;
        
        if (wait_time > 0) {
            std::this_thread::sleep_for(std::chrono::duration<double>(wait_time));
        }
        
        if (event.type == lio::SensorType::IMU) {
            // Process IMU data
            const auto& imu = imu_data[event.data_index];
            
            // Convert to IMU plot data
            lio::IMUPlotData plot_data;
            plot_data.timestamp = imu.timestamp;
            plot_data.gyro_x = imu.gyro_x;
            plot_data.gyro_y = imu.gyro_y;
            plot_data.gyro_z = imu.gyro_z;
            plot_data.acc_x = imu.acc_x;
            plot_data.acc_y = imu.acc_y;
            plot_data.acc_z = imu.acc_z;
            
            viewer.AddIMUMeasurement(plot_data);
            
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
                
                // Update viewer with point cloud
                viewer.UpdatePointCloud(cloud, current_pose);
                viewer.AddTrajectoryPoint(current_pose);
                viewer.UpdateStateInfo(lidar_frame_count, cloud->size());
                
                spdlog::info("[Frame {:4d}] Loaded LiDAR scan {:06d} with {:5d} points @ {:.3f}s", 
                            lidar_frame_count, lidar.scan_index, cloud->size(), 
                            event.timestamp - start_time);
                
                // TODO: Process with LIO estimator
                // For now, keep identity pose (no motion)
                
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
    spdlog::info("Viewer will stay open. Close window to quit.");
    
    // Keep viewer open until user closes it
    while (!viewer.ShouldClose()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    viewer.Shutdown();
    spdlog::info("Viewer closed. Exiting...");
    
    return 0;
}
