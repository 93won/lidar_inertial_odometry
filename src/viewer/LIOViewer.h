/**
 * @file      LIOViewer.h
 * @brief     Pangolin-based 3D viewer for LiDAR-Inertial Odometry visualization
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <GL/glew.h>  // Must be included before other GL headers
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/glinclude.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/var/var.h>
#include <pangolin/var/varextra.h>
#include <pangolin/display/widgets.h>
#include <pangolin/plot/plotter.h>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <deque>
#include "PointCloudUtils.h"
#include "VoxelMap.h"

namespace lio {

/**
 * @brief IMU bias data for plotting
 */
struct IMUBiasPlotData {
    double timestamp;
    float gyro_bias_norm;  // Gyroscope bias norm (rad/s)
    float acc_bias_norm;   // Accelerometer bias norm (m/s²)
};

/**
 * @brief Pangolin-based 3D viewer for LiDAR-Inertial Odometry
 * 
 */
class LIOViewer {
public:
    LIOViewer();
    ~LIOViewer();

    // ===== Initialization and Control =====
    
    /**
     * @brief Initialize the viewer window and start render thread
     * @param width Window width
     * @param height Window height
     * @return True if initialization succeeded
     */
    bool Initialize(int width = 1920, int height = 1080);
    
    /**
     * @brief Shutdown the viewer and stop render thread
     */
    void Shutdown();
    
    /**
     * @brief Check if viewer should close
     * @return True if close requested
     */
    bool ShouldClose() const;
    
    /**
     * @brief Check if viewer is ready
     * @return True if initialized and ready
     */
    bool IsReady() const;
    
    /**
     * @brief Reset camera to default view
     */
    void ResetCamera();

    // ===== Data Updates =====
    
    /**
     * @brief Update current LiDAR point cloud
     * @param cloud Point cloud in sensor frame
     * @param pose Current pose (4x4 transformation matrix)
     */
    void UpdatePointCloud(PointCloudPtr cloud, const Eigen::Matrix4f& pose);
    
    /**
     * @brief Add trajectory point
     * @param pose Pose to add to trajectory (4x4 transformation matrix)
     */
    void AddTrajectoryPoint(const Eigen::Matrix4f& pose);
    
    /**
     * @brief Add IMU bias measurement for plotting (bias norm)
     * @param bias_data IMU bias norm data
     */
    void AddIMUBias(const IMUBiasPlotData& bias_data);
    
    /**
     * @brief Update IMU bias for plotting
     * @param gyro_bias Gyroscope bias (rad/s)
     * @param acc_bias Accelerometer bias (m/s²)
     */
    void UpdateIMUBias(const Eigen::Vector3f& gyro_bias, const Eigen::Vector3f& acc_bias);
    
    /**
     * @brief Update state information
     * @param frame_id Current frame ID
     * @param num_points Number of points in current frame
     */
    void UpdateStateInfo(int frame_id, int num_points);
    
    /**
     * @brief Update map point cloud
     * @param map_cloud Map point cloud to display
     */
    void UpdateMapPointCloud(PointCloudPtr map_cloud);
    
    /**
     * @brief Update voxel map for visualization
     * @param voxel_map Voxel map to display as cubes
     */
    void UpdateVoxelMap(std::shared_ptr<VoxelMap> voxel_map);
    
    /**
     * @brief Check if auto playback is enabled
     * @return True if auto playback is on
     */
    bool IsAutoPlaybackEnabled() const { return m_auto_playback.Get(); }
    
    /**
     * @brief Check if step forward was requested
     * @return True if step forward button was pressed
     */
    bool WasStepForwardRequested() {
        if (pangolin::Pushed(m_step_forward_button)) {
            return true;
        }
        return false;
    }
    
    // ===== Setters for Configuration =====
    
    /**
     * @brief Set show point cloud flag
     */
    void SetShowPointCloud(bool show) { m_show_point_cloud = show; }
    
    /**
     * @brief Set show trajectory flag
     */
    void SetShowTrajectory(bool show) { m_show_trajectory = show; }
    
    /**
     * @brief Set show coordinate frame flag
     */
    void SetShowCoordinateFrame(bool show) { m_show_coordinate_frame = show; }
    
    /**
     * @brief Set show map flag
     */
    void SetShowMap(bool show) { m_show_map = show; }
    
    /**
     * @brief Set show voxel cubes flag
     */
    void SetShowVoxelCubes(bool show) { m_show_voxel_cubes = show; }
    
    /**
     * @brief Set show surfels flag
     */
    void SetShowSurfels(bool show) { m_show_surfels = show; }
    
    /**
     * @brief Set auto playback mode
     */
    void SetAutoPlayback(bool auto_play) { m_auto_playback = auto_play; }

private:
    // ===== Pangolin Components =====
    pangolin::OpenGlRenderState m_cam_state;
    pangolin::View m_display_3d;          ///< 3D point cloud view
    pangolin::View m_display_panel;       ///< UI panel
    
    // ===== Data Storage =====
    PointCloudPtr m_current_cloud;                     ///< Current point cloud
    PointCloudPtr m_map_cloud;                         ///< Map point cloud
    std::shared_ptr<VoxelMap> m_voxel_map;             ///< Voxel map for cube visualization
    Eigen::Matrix4f m_current_pose;                    ///< Current pose
    std::vector<Eigen::Matrix4f> m_trajectory;         ///< Trajectory
    std::deque<IMUBiasPlotData> m_bias_buffer;         ///< IMU bias data buffer for plotting
    
    // ===== Thread Safety =====
    mutable std::mutex m_data_mutex;
    
    // ===== Thread Management =====
    std::thread m_render_thread;                       ///< Render thread
    std::atomic<bool> m_should_stop;                   ///< Thread stop flag
    std::atomic<bool> m_thread_ready;                  ///< Thread ready flag
    
    // ===== UI Variables =====
    pangolin::Var<bool> m_show_point_cloud;            ///< Show point cloud checkbox
    pangolin::Var<bool> m_show_trajectory;             ///< Show trajectory checkbox
    pangolin::Var<bool> m_show_coordinate_frame;       ///< Show coordinate frame checkbox
    pangolin::Var<bool> m_show_map;                    ///< Show map checkbox
    pangolin::Var<bool> m_show_voxel_cubes;            ///< Show voxel cubes checkbox
    pangolin::Var<bool> m_show_surfels;                ///< Show L1 surfels checkbox
    pangolin::Var<bool> m_auto_playback;               ///< Auto playback mode
    pangolin::Var<bool> m_step_forward_button;         ///< Step forward button
    pangolin::Var<int> m_frame_id;                     ///< Current frame ID
    pangolin::Var<int> m_total_points;                 ///< Total points in frame
    
    // ===== Display Settings =====
    float m_point_size;                                 ///< Point cloud point size
    float m_trajectory_width;                          ///< Trajectory line width
    float m_coordinate_frame_size;                     ///< Coordinate frame axis length
    float m_coordinate_frame_width;                    ///< Coordinate frame line width
    
    // ===== Memory Management =====
    static constexpr size_t MAX_TRAJECTORY_POINTS = 10000; ///< Maximum trajectory points
    static constexpr size_t MAX_IMU_BUFFER_SIZE = 1000;    ///< Maximum IMU buffer size
    
    // ===== State =====
    bool m_initialized;                                 ///< Initialization state
    
    // ===== Internal Methods =====
    
    /**
     * @brief Setup UI panels and plots
     */
    void SetupPanels();
    
    /**
     * @brief Main render loop - runs in separate thread
     */
    void RenderLoop();
    
    /**
     * @brief Draw coordinate axes
     */
    void DrawCoordinateAxes();
    
    /**
     * @brief Draw point cloud
     */
    void DrawPointCloud();
    
    /**
     * @brief Draw map point cloud with transparency
     */
    void DrawMapPointCloud();
    
    /**
     * @brief Draw voxel map as cubes
     * @param voxel_map Voxel map to render as cubes
     */
    void DrawVoxelCubes(std::shared_ptr<VoxelMap> voxel_map);
    
    /**
     * @brief Draw L1 surfels with normal vectors
     * @param voxel_map Voxel map containing surfel data
     */
    void DrawSurfels(std::shared_ptr<VoxelMap> voxel_map);
    
    /**
     * @brief Helper function to draw a single cube (wireframe)
     * @param center Center position of the cube
     * @param size Size of the cube
     */
    void DrawCube(const Eigen::Vector3f& center, float size);
    
    /**
     * @brief Helper function to draw a single filled cube
     * @param center Center position of the cube
     * @param size Size of the cube
     */
    void DrawCubeFilled(const Eigen::Vector3f& center, float size);
    
    /**
     * @brief Draw trajectory
     */
    void DrawTrajectory();
    
    /**
     * @brief Draw current pose
     */
    void DrawCurrentPose();
};

} // namespace lio
