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

namespace lio {

/**
 * @brief IMU data for plotting
 */
struct IMUPlotData {
    double timestamp;
    float gyro_x, gyro_y, gyro_z;
    float acc_x, acc_y, acc_z;
};

/**
 * @brief Pangolin-based 3D viewer for LiDAR-Inertial Odometry
 * 
 * This viewer displays:
 * - Current LiDAR scan point cloud (colored by intensity)
 * - Camera trajectory
 * - IMU data plots (gyroscope and accelerometer)
 * - Coordinate frames and axes
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
     * @brief Add IMU measurement for plotting
     * @param imu_data IMU measurement data
     */
    void AddIMUMeasurement(const IMUPlotData& imu_data);
    
    /**
     * @brief Update state information
     * @param frame_id Current frame ID
     * @param num_points Number of points in current frame
     */
    void UpdateStateInfo(int frame_id, int num_points);

private:
    // ===== Pangolin Components =====
    pangolin::OpenGlRenderState m_cam_state;
    pangolin::View m_display_3d;          ///< 3D point cloud view
    pangolin::View m_display_panel;       ///< UI panel
    pangolin::DataLog m_gyro_log;         ///< Gyroscope data log
    pangolin::DataLog m_acc_log;          ///< Accelerometer data log
    pangolin::Plotter* m_plotter_gyro;    ///< Gyroscope plotter
    pangolin::Plotter* m_plotter_acc;     ///< Accelerometer plotter
    
    // ===== Data Storage =====
    PointCloudPtr m_current_cloud;                     ///< Current point cloud
    Eigen::Matrix4f m_current_pose;                    ///< Current pose
    std::vector<Eigen::Matrix4f> m_trajectory;         ///< Trajectory
    std::deque<IMUPlotData> m_imu_buffer;              ///< IMU data buffer for plotting
    
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
    pangolin::Var<bool> m_show_imu_plots;              ///< Show IMU plots checkbox
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
     * @brief Draw trajectory
     */
    void DrawTrajectory();
    
    /**
     * @brief Draw current pose
     */
    void DrawCurrentPose();
    
    /**
     * @brief Update IMU plots
     */
    void UpdateIMUPlots();
};

} // namespace lio
