/**
 * @file      LIOViewer.cpp
 * @brief     Implementation of Pangolin-based 3D viewer for LiDAR-Inertial Odometry
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "LIOViewer.h"
#include <spdlog/spdlog.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lio {

LIOViewer::LIOViewer()
    : m_should_stop(false)
    , m_thread_ready(false)
    , m_show_point_cloud("ui.1. Show Point Cloud", true, true)
    , m_show_trajectory("ui.2. Show Trajectory", true, true)
    , m_show_coordinate_frame("ui.3. Show Coordinate Frame", true, true)
    , m_show_imu_plots("ui.4. Show IMU Plots", true, true)
    , m_frame_id("info.Frame ID", 0)
    , m_total_points("info.Total Points", 0)
    , m_point_size(2.0f)
    , m_trajectory_width(3.0f)
    , m_coordinate_frame_size(2.0f)
    , m_coordinate_frame_width(3.0f)
    , m_initialized(false)
    , m_plotter_gyro(nullptr)
    , m_plotter_acc(nullptr)
{
    m_current_pose = Eigen::Matrix4f::Identity();
}

LIOViewer::~LIOViewer() {
    Shutdown();
}

bool LIOViewer::Initialize(int width, int height) {
    spdlog::info("[LIOViewer] Starting initialization with window size {}x{}", width, height);
    
    // Start render thread
    m_should_stop = false;
    m_thread_ready = false;
    
    m_render_thread = std::thread([this, width, height]() {
        try {
            // Create OpenGL window with Pangolin (must be done in render thread)
            pangolin::CreateWindowAndBind("LiDAR-Inertial Odometry - R3LIVE Dataset", width, height);
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            // Setup camera for 3D navigation
            float fx = width * 0.7f;
            float fy = height * 0.7f;
            m_cam_state = pangolin::OpenGlRenderState(
                pangolin::ProjectionMatrix(width, height, fx, fy, width/2, height/2, 0.1, 1000),
                pangolin::ModelViewLookAt(-10, -10, 10, 0, 0, 0, pangolin::AxisZ)
            );

            // Setup display panels
            SetupPanels();

            // Set clear color to dark background
            glClearColor(0.1f, 0.1f, 0.15f, 1.0f);

            m_initialized = true;
            m_thread_ready = true;
            spdlog::info("[LIOViewer] Render thread initialized successfully");

            // Run render loop
            RenderLoop();
            
        } catch (const std::exception& e) {
            spdlog::error("[LIOViewer] Exception in render thread: {}", e.what());
            m_initialized = false;
            m_thread_ready = false;
            return;
        }
        
        // Cleanup
        try {
            pangolin::DestroyWindow("LiDAR-Inertial Odometry - R3LIVE Dataset");
        } catch (const std::exception& e) {
            spdlog::warn("[LIOViewer] Exception during window cleanup: {}", e.what());
        }
        m_initialized = false;
        spdlog::info("[LIOViewer] Render thread finished");
    });
    
    // Wait for thread to be ready with timeout
    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!m_thread_ready && !m_should_stop && std::chrono::steady_clock::now() < timeout) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    if (!m_thread_ready) {
        spdlog::error("[LIOViewer] Failed to initialize render thread within timeout");
        m_should_stop = true;
        if (m_render_thread.joinable()) {
            m_render_thread.join();
        }
        return false;
    }
    
    spdlog::info("[LIOViewer] Initialized successfully with window size {}x{}", width, height);
    return m_thread_ready;
}

void LIOViewer::SetupPanels() {
    // Layout:
    // +------------------+------------------+
    // |                  |                  |
    // |   3D View        |   UI Panel       |
    // |                  |                  |
    // +------------------+------------------+
    // |   Gyro Plot      |   Acc Plot       |
    // +------------------+------------------+
    
    float ui_panel_width = 0.15f;     // UI panel 15% width
    float plot_height = 0.25f;        // Plot height 25%
    float plot_area_width = 1.0f - ui_panel_width;  // Total width for plots (85%)
    float plot_half_width = plot_area_width * 0.5f; // Half width for each plot
    
    // Create main 3D view (top-left)
    m_display_3d = pangolin::CreateDisplay()
        .SetBounds(pangolin::Attach::Frac(plot_height), 1.0, 
                   0.0, pangolin::Attach::Frac(plot_area_width))
        .SetHandler(new pangolin::Handler3D(m_cam_state));
    
    // Create UI panel (top-right)
    pangolin::View& d_panel = pangolin::CreatePanel("ui")
        .SetBounds(pangolin::Attach::Frac(plot_height), 1.0, 
                   pangolin::Attach::Frac(plot_area_width), 1.0);
    
    // Create gyroscope plot (bottom-left, exactly half width)
    m_gyro_log.SetLabels({"Gyro X", "Gyro Y", "Gyro Z"});
    m_plotter_gyro = new pangolin::Plotter(&m_gyro_log, 0.0f, 1000.0f, -10.0f, 10.0f, 100.0f, 2.0f);
    m_plotter_gyro->SetBounds(0.0, pangolin::Attach::Frac(plot_height), 
                              0.0, pangolin::Attach::Frac(plot_half_width));
    m_plotter_gyro->Track("$i");
    pangolin::DisplayBase().AddDisplay(*m_plotter_gyro);
    
    // Create accelerometer plot (bottom-right, exactly half width)
    m_acc_log.SetLabels({"Acc X", "Acc Y", "Acc Z"});
    m_plotter_acc = new pangolin::Plotter(&m_acc_log, 0.0f, 1000.0f, -20.0f, 20.0f, 100.0f, 2.0f);
    m_plotter_acc->SetBounds(0.0, pangolin::Attach::Frac(plot_height), 
                             pangolin::Attach::Frac(plot_half_width), pangolin::Attach::Frac(plot_area_width));
    m_plotter_acc->Track("$i");
    pangolin::DisplayBase().AddDisplay(*m_plotter_acc);
    
    spdlog::info("[LIOViewer] UI panel and displays created successfully");
}

void LIOViewer::Shutdown() {
    if (m_initialized || m_thread_ready) {
        spdlog::info("[LIOViewer] Shutting down viewer...");
        m_should_stop = true;
        
        if (m_render_thread.joinable()) {
            m_render_thread.join();
        }
        
        m_initialized = false;
        m_thread_ready = false;
        spdlog::info("[LIOViewer] Viewer shutdown completed");
    }
}

bool LIOViewer::ShouldClose() const {
    return m_should_stop;
}

bool LIOViewer::IsReady() const {
    return m_thread_ready;
}

void LIOViewer::ResetCamera() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_cam_state.SetModelViewMatrix(pangolin::ModelViewLookAt(-10, -10, 10, 0, 0, 0, pangolin::AxisZ));
}

void LIOViewer::UpdatePointCloud(PointCloudPtr cloud, const Eigen::Matrix4f& pose) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_current_cloud = cloud;
    m_current_pose = pose;
}

void LIOViewer::AddTrajectoryPoint(const Eigen::Matrix4f& pose) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_trajectory.push_back(pose);
    
    // Keep trajectory size limited
    if (m_trajectory.size() > MAX_TRAJECTORY_POINTS) {
        m_trajectory.erase(m_trajectory.begin());
    }
}

void LIOViewer::AddIMUMeasurement(const IMUPlotData& imu_data) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_imu_buffer.push_back(imu_data);
    
    // Keep buffer size limited
    if (m_imu_buffer.size() > MAX_IMU_BUFFER_SIZE) {
        m_imu_buffer.pop_front();
    }
}

void LIOViewer::UpdateStateInfo(int frame_id, int num_points) {
    m_frame_id = frame_id;
    m_total_points = num_points;
}

void LIOViewer::RenderLoop() {
    spdlog::info("[LIOViewer] Starting render loop");
    
    while (!m_should_stop) {
        // Check if window should quit
        if (pangolin::ShouldQuit()) {
            spdlog::info("[LIOViewer] Pangolin quit requested");
            m_should_stop = true;
            break;
        }

        // Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Activate 3D view
        m_display_3d.Activate(m_cam_state);

        // Copy data once with single lock
        PointCloudPtr current_cloud_copy;
        Eigen::Matrix4f current_pose_copy;
        std::vector<Eigen::Matrix4f> trajectory_copy;
        
        {
            std::lock_guard<std::mutex> lock(m_data_mutex);
            current_cloud_copy = m_current_cloud;
            current_pose_copy = m_current_pose;
            trajectory_copy = m_trajectory;
        }

        // Draw 3D content
        if (m_show_coordinate_frame.Get()) {
            DrawCoordinateAxes();
        }

        if (m_show_point_cloud.Get() && current_cloud_copy && !current_cloud_copy->empty()) {
            DrawPointCloud();
        }

        if (m_show_trajectory.Get() && trajectory_copy.size() > 1) {
            DrawTrajectory();
        }

        DrawCurrentPose();

        // Update IMU plots
        if (m_show_imu_plots.Get()) {
            UpdateIMUPlots();
        }

        // Swap buffers
        pangolin::FinishFrame();
        
        // Small sleep to avoid 100% CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    spdlog::info("[LIOViewer] Render loop finished");
}

void LIOViewer::DrawCoordinateAxes() {
    // Draw world coordinate frame at origin
    glLineWidth(m_coordinate_frame_width);
    
    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(m_coordinate_frame_size, 0.0f, 0.0f);
    glEnd();
    
    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, m_coordinate_frame_size, 0.0f);
    glEnd();
    
    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, m_coordinate_frame_size);
    glEnd();
    
    glLineWidth(1.0f);
}

void LIOViewer::DrawPointCloud() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    if (!m_current_cloud || m_current_cloud->empty()) {
        return;
    }
    
    glPointSize(m_point_size);
    glBegin(GL_POINTS);
    
    // Color by offset_time: blue (early) -> red (late)
    for (const auto& point : *m_current_cloud) {
        // Transform point to world coordinates
        Eigen::Vector4f local_point(point.x, point.y, point.z, 1.0f);
        Eigen::Vector4f world_point = m_current_pose * local_point;
        
        // point.offset_time stores time in seconds (0 ~ 0.1)
        // Normalize to 0~1 range
        float time_ratio = point.offset_time / 0.1f;  // 0.1 sec = max offset time
        time_ratio = std::max(0.0f, std::min(1.0f, time_ratio));
        
        // Blue to Red colormap
        float r = time_ratio;       // 0 -> 1 (blue to red)
        float g = 0.0f;             // No green
        float b = 1.0f - time_ratio; // 1 -> 0 (blue to red)
        
        glColor3f(r, g, b);
        glVertex3f(world_point.x(), world_point.y(), world_point.z());
    }
    
    glEnd();
    glPointSize(1.0f);
}

void LIOViewer::DrawTrajectory() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    if (m_trajectory.size() < 2) {
        return;
    }
    
    glLineWidth(m_trajectory_width);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow trajectory
    
    glBegin(GL_LINE_STRIP);
    for (const auto& pose : m_trajectory) {
        Eigen::Vector3f position = pose.block<3, 1>(0, 3);
        glVertex3f(position.x(), position.y(), position.z());
    }
    glEnd();
    
    glLineWidth(1.0f);
}

void LIOViewer::DrawCurrentPose() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    Eigen::Vector3f position = m_current_pose.block<3, 1>(0, 3);
    Eigen::Matrix3f rotation = m_current_pose.block<3, 3>(0, 0);
    
    // Draw coordinate frame at current pose
    float axis_length = m_coordinate_frame_size * 0.5f;
    glLineWidth(m_coordinate_frame_width);
    
    // X axis - Red
    Eigen::Vector3f x_axis = rotation.col(0);
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(position.x() + x_axis.x() * axis_length,
               position.y() + x_axis.y() * axis_length,
               position.z() + x_axis.z() * axis_length);
    glEnd();
    
    // Y axis - Green
    Eigen::Vector3f y_axis = rotation.col(1);
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(position.x() + y_axis.x() * axis_length,
               position.y() + y_axis.y() * axis_length,
               position.z() + y_axis.z() * axis_length);
    glEnd();
    
    // Z axis - Blue
    Eigen::Vector3f z_axis = rotation.col(2);
    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(position.x() + z_axis.x() * axis_length,
               position.y() + z_axis.y() * axis_length,
               position.z() + z_axis.z() * axis_length);
    glEnd();
    
    glLineWidth(1.0f);
}

void LIOViewer::UpdateIMUPlots() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    if (m_imu_buffer.empty()) {
        return;
    }
    
    // Get reference timestamp (first timestamp)
    static double ref_timestamp = m_imu_buffer.front().timestamp;
    
    // Add recent IMU data to plots
    for (const auto& imu : m_imu_buffer) {
        double relative_time = imu.timestamp - ref_timestamp;
        
        // Log gyroscope data (x, y, z only - no time in data)
        m_gyro_log.Log(imu.gyro_x, imu.gyro_y, imu.gyro_z);
        
        // Log accelerometer data (x, y, z only - no time in data)
        m_acc_log.Log(imu.acc_x, imu.acc_y, imu.acc_z);
    }
    
    // Clear buffer after logging
    m_imu_buffer.clear();
}

} // namespace lio
