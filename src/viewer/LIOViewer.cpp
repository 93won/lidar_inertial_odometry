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
    , m_show_L1_voxel("ui.4. Show L1 Voxels", false, true)  // Show valid L1 voxel cubes
    , m_show_map_points("ui.5. Show Map Points", true, true)  // Show surfel centroids as green points (default ON)
    , m_show_voxel_cubes("ui.6. Show L0 Voxels", false, true)  // Show L0 voxel cubes (heavy)
    , m_show_surfels("ui.7. Show Surfels", false, true)  // Show surfel discs with normals
    , m_auto_playback("ui.8. Auto Playback", false, true)  // Enable auto playback by default
    , m_step_forward_button("ui.9. Step Forward", false, false)
    , m_follow_mode("ui.10. Follow Mode (Top-Down)", true, true)  // Enable follow mode with mouse zoom support
    , m_frame_id("info.Frame ID", 0)
    , m_total_points("info.Total Points", 0)
    , m_point_size(2.0f)
    , m_trajectory_width(3.0f)
    , m_coordinate_frame_size(2.0f)
    , m_coordinate_frame_width(3.0f)
    , m_camera_target(0.0f, 0.0f, 0.0f)
    , m_initialized(false)
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
            pangolin::CreateWindowAndBind("Tightly-Coupled LiDAR-Inertial Odometry", width, height);
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
            pangolin::DestroyWindow("Tightly-Coupled LiDAR-Inertial Odometry");
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
    
    // Create UI panel (left side)
    pangolin::View& d_panel = pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Frac(ui_panel_width));
    
    // Create main 3D view (right side)
    m_display_3d = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 
                   pangolin::Attach::Frac(ui_panel_width), 1.0)
        .SetHandler(new pangolin::Handler3D(m_cam_state, pangolin::AxisZ, 1.0f, 0.01f));
    
    spdlog::info("[LIOViewer] UI panel and displays created successfully");
    spdlog::info("[LIOViewer] Controls:");
    spdlog::info("  Toggle 'Auto Playback' checkbox to enable/disable automatic playback");
    spdlog::info("  Click 'Step Forward' button to go to next LiDAR frame (when auto playback is OFF)");
    spdlog::info("  Use 'Camera Height' slider to zoom in/out in follow mode");
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

void LIOViewer::AddIMUBias(const IMUBiasPlotData& bias_data) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_bias_buffer.push_back(bias_data);
    
    // Keep buffer size limited
    if (m_bias_buffer.size() > MAX_IMU_BUFFER_SIZE) {
        m_bias_buffer.pop_front();
    }
}

void LIOViewer::UpdateIMUBias(const Eigen::Vector3f& gyro_bias, const Eigen::Vector3f& acc_bias) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    // Create IMU bias plot data with norm values
    static int bias_count = 0;
    
    IMUBiasPlotData bias_data;
    bias_data.timestamp = bias_count * 0.1;  // Assuming 10Hz update rate
    bias_data.gyro_bias_norm = gyro_bias.norm();  // Compute norm
    bias_data.acc_bias_norm = acc_bias.norm();    // Compute norm
    
    m_bias_buffer.push_back(bias_data);
    
    // Keep buffer size limited
    if (m_bias_buffer.size() > MAX_IMU_BUFFER_SIZE) {
        m_bias_buffer.pop_front();
    }
    
    bias_count++;
}

void LIOViewer::UpdateStateInfo(int frame_id, int num_points) {
    m_frame_id = frame_id;
    m_total_points = num_points;
}

void LIOViewer::UpdateMapPointCloud(PointCloudPtr map_cloud) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_map_cloud = map_cloud;
}

void LIOViewer::UpdateMapCentroids(const std::vector<std::pair<Eigen::Vector3f, float>>& centroids) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_map_centroids = centroids;
}

void LIOViewer::UpdateVoxelMap(std::shared_ptr<VoxelMap> voxel_map) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_voxel_map = voxel_map;
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

        // Update camera for follow mode (using Follow method for zoom support)
        if (m_follow_mode.Get()) {
            Eigen::Matrix4f current_pose_copy;
            {
                std::lock_guard<std::mutex> lock(m_data_mutex);
                current_pose_copy = m_current_pose;
            }
            
            // Extract current position
            Eigen::Vector3f current_pos = current_pose_copy.block<3, 1>(0, 3);
            
            // Smooth interpolation
            float follow_speed = 0.1f;
            m_camera_target = (1.0f - follow_speed) * m_camera_target + 
                             follow_speed * current_pos;
            
            // Create follow matrix for Pangolin Follow() method
            // This allows user to zoom/pan while following the target
            pangolin::OpenGlMatrix follow_matrix = pangolin::OpenGlMatrix::Translate(
                m_camera_target.x(), 
                m_camera_target.y(), 
                m_camera_target.z()
            );
            
            // Use Follow method - allows zoom with mouse wheel
            m_cam_state.Follow(follow_matrix, true);
            
            // Set initial top-down view on first activation
            static bool first_follow_activation = true;
            if (first_follow_activation) {
                float initial_height = 50.0f;
                pangolin::OpenGlMatrix initial_view = pangolin::ModelViewLookAt(
                    m_camera_target.x(), 
                    m_camera_target.y(), 
                    m_camera_target.z() + initial_height,
                    m_camera_target.x(), 
                    m_camera_target.y(), 
                    m_camera_target.z(),
                    0, 1, 0
                );
                m_cam_state.SetModelViewMatrix(initial_view);
                first_follow_activation = false;
            }
        } else {
            // Reset first activation flag when follow mode is disabled
            static bool first_follow_activation = true;
            first_follow_activation = true;
        }

        // Activate 3D view
        m_display_3d.Activate(m_cam_state);

        // Copy data once with single lock
        PointCloudPtr current_cloud_copy;
        PointCloudPtr map_cloud_copy;
        std::shared_ptr<VoxelMap> voxel_map_copy;
        Eigen::Matrix4f current_pose_copy;
        std::vector<Eigen::Matrix4f> trajectory_copy;
        
        {
            std::lock_guard<std::mutex> lock(m_data_mutex);
            current_cloud_copy = m_current_cloud;
            map_cloud_copy = m_map_cloud;
            voxel_map_copy = m_voxel_map;
            current_pose_copy = m_current_pose;
            trajectory_copy = m_trajectory;
        }

        // Draw 3D content
        if (m_show_coordinate_frame.Get()) {
            DrawCoordinateAxes();
        }

        if (m_show_voxel_cubes.Get() && voxel_map_copy && voxel_map_copy->GetVoxelCount() > 0) {
            DrawVoxelCubes(voxel_map_copy);  // Draw L0 voxel cubes (heavy)
        }
        
        if (m_show_L1_voxel.Get() && voxel_map_copy && voxel_map_copy->GetVoxelCount() > 0) {
            DrawL1VoxelCubes(voxel_map_copy);  // Draw L1 voxel cubes (lightweight)
        }
        
        if (m_show_map_points.Get() && voxel_map_copy && voxel_map_copy->GetVoxelCount() > 0) {
            DrawMapPoints(voxel_map_copy);  // Draw surfel centroids as green points
        }
        
        if (m_show_surfels.Get() && voxel_map_copy && voxel_map_copy->GetVoxelCount() > 0) {
            DrawSurfels(voxel_map_copy);  // Draw L1 surfels with normals
        }

        // Draw current scan point cloud
        if (m_show_point_cloud.Get() && current_cloud_copy && !current_cloud_copy->empty()) {
            DrawPointCloud();
        }

        // Draw trajectory and current pose LAST so they appear on top
        if (m_show_trajectory.Get() && trajectory_copy.size() > 1) {
            DrawTrajectory();
        }

        DrawCurrentPose();

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
    
    // Draw current scan points as thick bright red points
    glPointSize(1.0f);  // Thick points for visibility
    glBegin(GL_POINTS);
    
    // Bright red color for all scan points
    glColor3f(1.0f, 0.1f, 0.1f);  // Bright red
    
    for (const auto& point : *m_current_cloud) {
        // Transform point to world coordinates
        Eigen::Vector4f local_point(point.x, point.y, point.z, 1.0f);
        Eigen::Vector4f world_point = m_current_pose * local_point;
        
        glVertex3f(world_point.x(), world_point.y(), world_point.z());
    }
    
    glEnd();
    glPointSize(1.0f);
}

void LIOViewer::DrawMapPointCloud() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    if (m_map_centroids.empty()) {
        return;
    }
    
    // Save GL state for alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glPointSize(1.0f);
    glBegin(GL_POINTS);

    // Draw cached centroids - no VoxelMap access needed
    for (const auto& centroid_data : m_map_centroids) {
        const Eigen::Vector3f& centroid = centroid_data.first;
        float alpha = centroid_data.second;
        
        // Draw voxel centroid in green with hit-count based alpha
        glColor4f(0.0f, 1.0f, 0.0f, alpha);
        glVertex3f(centroid.x(), centroid.y(), centroid.z());
    }
    
    glEnd();
    glPointSize(1.0f);
    
    // Restore GL state
    glDisable(GL_BLEND);
}

void LIOViewer::DrawTrajectory() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    if (m_trajectory.size() < 2) {
        return;
    }
    
    // Disable depth test so trajectory is always visible on top
    glDisable(GL_DEPTH_TEST);
    
    glLineWidth(m_trajectory_width);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow trajectory
    
    // Draw only recent 3 seconds of trajectory
    // Assuming ~10Hz LiDAR rate, draw last 30 poses
    const size_t recent_count = 30;
    size_t start_idx = (m_trajectory.size() > recent_count) 
                        ? (m_trajectory.size() - recent_count) 
                        : 0;
    
    glBegin(GL_LINE_STRIP);
    for (size_t i = start_idx; i < m_trajectory.size(); ++i) {
        Eigen::Vector3f position = m_trajectory[i].block<3, 1>(0, 3);
        glVertex3f(position.x(), position.y(), position.z());
    }
    glEnd();
    
    glLineWidth(1.0f);
    
    // Re-enable depth test
    glEnable(GL_DEPTH_TEST);
}

void LIOViewer::DrawVoxelCubes(std::shared_ptr<VoxelMap> voxel_map) {
    if (!voxel_map || voxel_map->GetVoxelCount() == 0) {
        return;
    }
    
    // Get all occupied voxels with hit counts in one thread-safe call
    auto voxels_with_hitcount = voxel_map->GetOccupiedVoxelsWithHitCount();
    if (voxels_with_hitcount.empty()) {
        return;
    }
    
    float voxel_size = voxel_map->GetVoxelSize();
    
    // Fixed z range for colormap: -1.0 to 10.0
    const float z_min = -1.0f;
    const float z_max = 3.0f;
    const float z_range = z_max - z_min;  // 11.0
    
    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Draw each voxel as a colored cube based on z height
    glLineWidth(1.0f);
    
    // Get max hit count from VoxelMap config
    const int max_hit_count = voxel_map->GetMaxHitCount();
    const int hit_threshold = std::max(0, max_hit_count - 100);
    
    for (const auto& [center, hit_count] : voxels_with_hitcount) {
        // Only draw voxels with hit_count >= max_hit - 100
        if (hit_count < hit_threshold) {
            continue;
        }
        
        // Calculate alpha: linear mapping from [max-100, max] to [0.1, 0.3]
        float t = static_cast<float>(hit_count - hit_threshold) / 100.0f;
        t = std::min(1.0f, std::max(0.0f, t));
        float alpha = 0.1f + t * 0.2f;  // 0.0 to 0.2
        
        // Calculate color based on z height: red (z=-1) to blue (z=10)
        float normalized_z = (center.z() - z_min) / z_range;  // 0 to 1
        normalized_z = std::min(1.0f, std::max(0.0f, normalized_z));  // clamp
        
        // Red to Blue colormap (red at low z, blue at high z)
        float r = 0.0f;//1.0f - normalized_z;  // 1.0 at z=-1, 0.0 at z=10
        float g = 1.0f - normalized_z;
        float b = normalized_z;         // 0.0 at z=-1, 1.0 at z=10
        
        // Draw filled cube with z-based color and hit-count based alpha
        glColor4f(r, g, b, alpha);
        DrawCubeFilled(center, voxel_size);
        
        // Draw edges with slightly higher alpha
        glColor4f(r, g, b, alpha * 2.0f);
        DrawCube(center, voxel_size);
    }
    
    glDisable(GL_BLEND);
    glLineWidth(1.0f);
}

void LIOViewer::DrawCubeFilled(const Eigen::Vector3f& center, float size) {
    float half = size * 0.5f;
    
    // 8 vertices of the cube
    float v[8][3] = {
        {center.x() - half, center.y() - half, center.z() - half},  // 0: ---
        {center.x() + half, center.y() - half, center.z() - half},  // 1: +--
        {center.x() + half, center.y() + half, center.z() - half},  // 2: ++-
        {center.x() - half, center.y() + half, center.z() - half},  // 3: -+-
        {center.x() - half, center.y() - half, center.z() + half},  // 4: --+
        {center.x() + half, center.y() - half, center.z() + half},  // 5: +-+
        {center.x() + half, center.y() + half, center.z() + half},  // 6: +++
        {center.x() - half, center.y() + half, center.z() + half}   // 7: -++
    };
    
    // Draw 6 faces as quads
    glBegin(GL_QUADS);
    
    // Front face (z = +half)
    glVertex3fv(v[4]); glVertex3fv(v[5]); glVertex3fv(v[6]); glVertex3fv(v[7]);
    
    // Back face (z = -half)
    glVertex3fv(v[0]); glVertex3fv(v[3]); glVertex3fv(v[2]); glVertex3fv(v[1]);
    
    // Top face (y = +half)
    glVertex3fv(v[3]); glVertex3fv(v[7]); glVertex3fv(v[6]); glVertex3fv(v[2]);
    
    // Bottom face (y = -half)
    glVertex3fv(v[0]); glVertex3fv(v[1]); glVertex3fv(v[5]); glVertex3fv(v[4]);
    
    // Right face (x = +half)
    glVertex3fv(v[1]); glVertex3fv(v[2]); glVertex3fv(v[6]); glVertex3fv(v[5]);
    
    // Left face (x = -half)
    glVertex3fv(v[0]); glVertex3fv(v[4]); glVertex3fv(v[7]); glVertex3fv(v[3]);
    
    glEnd();
}

void LIOViewer::DrawCube(const Eigen::Vector3f& center, float size) {
    float half = size * 0.5f;
    
    // 8 vertices of the cube
    float vertices[8][3] = {
        {center.x() - half, center.y() - half, center.z() - half},  // 0: ---
        {center.x() + half, center.y() - half, center.z() - half},  // 1: +--
        {center.x() + half, center.y() + half, center.z() - half},  // 2: ++-
        {center.x() - half, center.y() + half, center.z() - half},  // 3: -+-
        {center.x() - half, center.y() - half, center.z() + half},  // 4: --+
        {center.x() + half, center.y() - half, center.z() + half},  // 5: +-+
        {center.x() + half, center.y() + half, center.z() + half},  // 6: +++
        {center.x() - half, center.y() + half, center.z() + half}   // 7: -++
    };
    
    // Draw 12 edges of the cube
    int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face
        {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face
        {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
    };
    
    glBegin(GL_LINES);
    for (int i = 0; i < 12; ++i) {
        int v1 = edges[i][0];
        int v2 = edges[i][1];
        glVertex3f(vertices[v1][0], vertices[v1][1], vertices[v1][2]);
        glVertex3f(vertices[v2][0], vertices[v2][1], vertices[v2][2]);
    }
    glEnd();
}

void LIOViewer::DrawL1VoxelCubes(std::shared_ptr<VoxelMap> voxel_map) {
    if (!voxel_map) return;
    
    // Get all L1 surfels (only valid L1 voxels have surfels)
    auto surfels = voxel_map->GetL1Surfels();
    
    if (surfels.empty()) {
        return;
    }
    
    // L1 voxel size
    float l1_voxel_size = voxel_map->GetVoxelSize() * voxel_map->GetHierarchyFactor();
    
    // Draw L1 voxel cubes as wireframes
    glColor4f(0.0f, 0.8f, 0.2f, 0.5f);  // Green wireframe
    glLineWidth(1.0f);
    
    for (const auto& surfel_data : surfels) {
        const VoxelKey& key_L1 = std::get<3>(surfel_data);
        
        // Calculate L1 voxel center
        Eigen::Vector3f center(
            (key_L1.x + 0.5f) * l1_voxel_size,
            (key_L1.y + 0.5f) * l1_voxel_size,
            (key_L1.z + 0.5f) * l1_voxel_size
        );
        
        DrawCube(center, l1_voxel_size);
    }
}

void LIOViewer::DrawMapPoints(std::shared_ptr<VoxelMap> voxel_map) {
    if (!voxel_map) return;
    
    // Get all L1 surfels
    auto surfels = voxel_map->GetL1Surfels();
    
    if (surfels.empty()) {
        return;
    }
    
    // Get max hit count for alpha normalization
    int max_hit_count = voxel_map->GetMaxHitCount();
    
    // Enable blending for alpha
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Draw surfel centroids as green points with alpha based on hit_count
    glPointSize(3.0f);
    glBegin(GL_POINTS);
    
    for (const auto& surfel_data : surfels) {
        const Eigen::Vector3f& centroid = std::get<0>(surfel_data);
        int hit_count = std::get<4>(surfel_data);
        
        // Alpha: 0.2 (min) to 1.0 (max) based on hit_count
        float alpha = 0.2f + 0.8f * (static_cast<float>(hit_count) / static_cast<float>(max_hit_count));
        alpha = std::min(1.0f, std::max(0.2f, alpha));
        
        // Green color with alpha based on hit_count
        glColor4f(0.0f, 1.0f, 0.0f, alpha);
        glVertex3f(centroid.x(), centroid.y(), centroid.z());
    }
    
    glEnd();
    glPointSize(1.0f);
}

void LIOViewer::DrawSurfels(std::shared_ptr<VoxelMap> voxel_map) {
    if (!voxel_map) return;
    
    // Get all L1 surfels
    auto surfels = voxel_map->GetL1Surfels();
    
    if (surfels.empty()) {
        return;
    }
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Surfel disc radius = L1 voxel size (hierarchy_factor × L0 voxel size)
    // Use 1/6 of L1 size for better visualization
    float l1_voxel_size = voxel_map->GetVoxelSize() * voxel_map->GetHierarchyFactor();
    float disc_radius = l1_voxel_size / 2.0f;  // 1/6 of L1 size
    const int num_segments = 20;  // Circle resolution
    
    for (const auto& surfel_data : surfels) {
        const Eigen::Vector3f& centroid = std::get<0>(surfel_data);
        const Eigen::Vector3f& normal = std::get<1>(surfel_data);
        float planarity = std::get<2>(surfel_data);
        
        // Color based on planarity: more planar = more green
        // planarity ranges from 0 (perfect plane) to 0.1 (threshold)
        float green_intensity = 1.0f - (planarity / 0.1f);  // 1.0 for perfect plane, 0.0 at threshold
        green_intensity = std::max(0.0f, std::min(1.0f, green_intensity));
        
        // Find two perpendicular vectors to the normal (local coordinate system on the plane)
        Eigen::Vector3f u_axis, v_axis;
        if (std::abs(normal.x()) < 0.9f) {
            u_axis = Eigen::Vector3f(1, 0, 0).cross(normal).normalized();
        } else {
            u_axis = Eigen::Vector3f(0, 1, 0).cross(normal).normalized();
        }
        v_axis = normal.cross(u_axis).normalized();
        
        // Draw filled disc (surfel surface)
        glColor4f(0.2f, green_intensity * 0.8f, 0.9f, 0.1f);  // Alpha 0.1
        glBegin(GL_TRIANGLE_FAN);
        glVertex3f(centroid.x(), centroid.y(), centroid.z());  // Center vertex
        
        for (int i = 0; i <= num_segments; ++i) {
            float angle = 2.0f * M_PI * i / num_segments;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            
            // Point on circle: centroid + radius * (cos(θ)*u + sin(θ)*v)
            Eigen::Vector3f circle_point = centroid + disc_radius * (cos_a * u_axis + sin_a * v_axis);
            glVertex3f(circle_point.x(), circle_point.y(), circle_point.z());
        }
        glEnd();
        
        // Draw circle outline (border)
        glLineWidth(2.0f);
        glColor4f(0.0f, green_intensity, 1.0f, 0.3f);  // Outline alpha 0.3
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < num_segments; ++i) {
            float angle = 2.0f * M_PI * i / num_segments;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            
            Eigen::Vector3f circle_point = centroid + disc_radius * (cos_a * u_axis + sin_a * v_axis);
            glVertex3f(circle_point.x(), circle_point.y(), circle_point.z());
        }
        glEnd();
    }
    
    glDisable(GL_BLEND);
    glLineWidth(1.0f);
}

void LIOViewer::DrawCurrentPose() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    // Disable depth test so pose is always visible on top
    glDisable(GL_DEPTH_TEST);
    
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
    
    // Re-enable depth test
    glEnable(GL_DEPTH_TEST);
}

} // namespace lio
