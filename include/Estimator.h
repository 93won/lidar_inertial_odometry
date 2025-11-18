/**
 * @file      Estimator.h
 * @brief     Main estimation engine for LiDAR-Inertial Odometry system.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include "State.h"
#include "PointCloudUtils.h"
#include <vector>
#include <memory>
#include <deque>
#include <mutex>
#include <Eigen/Dense>

namespace lio {

// IMU measurement structure
struct IMUData {
    double timestamp;
    Eigen::Vector3d acc;    // acceleration [m/s²]
    Eigen::Vector3d gyr;    // angular velocity [rad/s]
    
    IMUData(double t, const Eigen::Vector3d& a, const Eigen::Vector3d& g)
        : timestamp(t), acc(a), gyr(g) {}
};

// LiDAR scan structure
struct LidarData {
    double timestamp;
    PointCloudPtr cloud;
    
    LidarData(double t, PointCloudPtr c) 
        : timestamp(t), cloud(c) {}
};

// Local map point structure
struct MapPoint {
    Eigen::Vector3d position;
    Eigen::Vector3d normal;
    double timestamp;
    int frame_id;
    
    MapPoint(const Eigen::Vector3d& pos, const Eigen::Vector3d& norm, double t, int id)
        : position(pos), normal(norm), timestamp(t), frame_id(id) {}
};

/**
 * @brief Main estimation engine using 18-dimensional direct state EKF
 * 
 * Features:
 * - IMU propagation with bias estimation
 * - LiDAR point-to-plane residuals
 * - Local map maintenance (sliding window)
 * - Direct state EKF (not error-state)
 * - Real-time processing capability
 */
class Estimator {
public:
    Estimator();
    ~Estimator();
    
    /// Initialize estimator with first IMU measurement
    void Initialize(const IMUData& first_imu);
    
    /// Process new IMU measurement
    void ProcessIMU(const IMUData& imu);
    
    /// Process new LiDAR scan
    void ProcessLidar(const LidarData& lidar);
    
    /// Get current state estimate
    State GetCurrentState() const;
    
    /// Get trajectory history
    std::vector<State> GetTrajectory() const;
    
    /// Check if system is initialized
    bool IsInitialized() const { return m_initialized; }
    
    /// Get processing statistics
    struct Statistics {
        int total_frames;
        int successful_registrations;
        double avg_processing_time_ms;
        double total_distance;
        double avg_translation_error;
        double avg_rotation_error;
    };
    
    Statistics GetStatistics() const;
    
    // Configuration parameters
    struct Parameters {
        // IMU parameters
        double acc_noise_std = 0.1;           // m/s²
        double gyr_noise_std = 0.01;          // rad/s
        double acc_bias_noise_std = 0.001;    // m/s²
        double gyr_bias_noise_std = 0.0001;   // rad/s
        double gravity_noise_std = 0.001;     // m/s²
        
        // LiDAR parameters
        double lidar_noise_std = 0.05;        // m
        double max_correspondence_distance = 1.0;  // m
        int max_iterations = 10;
        double convergence_threshold = 1e-3;
        
        // Local map parameters
        double map_voxel_size = 0.2;          // m
        int max_map_points = 100000;
        double max_map_distance = 50.0;       // m
        double min_plane_points = 5;
        
        // Processing parameters
        double min_motion_threshold = 0.1;     // m
        int imu_buffer_size = 1000;
        bool enable_undistortion = true;
    } m_params;

private:
    /// Propagate state using IMU measurements
    void PropagateState(const IMUData& imu);
    
    /// Update state using LiDAR correspondences
    void UpdateWithLidar(const LidarData& lidar);
    
    /// Find point-to-plane correspondences
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> 
    FindCorrespondences(const PointCloudPtr scan);
    
    /// Update local map with new scan
    void UpdateLocalMap(const PointCloudPtr scan);
    
    /// Remove old points from local map
    void CleanLocalMap();
    
    /// Extract planar features from point cloud
    void ExtractPlanarFeatures(const PointCloudPtr cloud,
                              std::vector<MapPoint>& features);
    
    /// Compute Jacobians for point-to-plane residuals
    void ComputeLidarJacobians(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& correspondences,
                              Eigen::MatrixXd& H,
                              Eigen::VectorXd& residual);
    
    /// Motion undistortion for point cloud
    PointCloudPtr 
    UndistortPointCloud(const PointCloudPtr cloud,
                        double scan_start_time,
                        double scan_end_time);
    
    /// Interpolate state at given timestamp
    State InterpolateState(double timestamp) const;
    
    /// Update process noise matrices
    void UpdateProcessNoise(double dt);
    
    /// Update measurement noise matrices
    void UpdateMeasurementNoise(int num_correspondences);

private:
    // State estimation
    State m_current_state;
    bool m_initialized;
    double m_last_update_time;
    int m_frame_count;
    
    // Data buffers (thread-safe)
    mutable std::mutex m_state_mutex;
    std::deque<IMUData> m_imu_buffer;
    std::deque<State> m_trajectory;
    
    // Local map storage
    std::vector<MapPoint> m_local_map;
    PointCloudPtr m_map_cloud;
    mutable std::mutex m_map_mutex;
    
    // Processing statistics
    mutable std::mutex m_stats_mutex;
    Statistics m_statistics;
    std::vector<double> m_processing_times;
    
    // Kalman filter matrices
    Eigen::Matrix<double, 18, 18> m_process_noise;      // Q
    Eigen::MatrixXd m_measurement_noise;                // R (dynamic size)
    Eigen::Matrix<double, 18, 18> m_state_transition;   // F
    
    // Temporary computation storage
    Eigen::MatrixXd m_jacobian;
    Eigen::VectorXd m_residual_vector;
    Eigen::MatrixXd m_kalman_gain;
    
    // Configuration
    bool m_first_lidar_frame;
    double m_last_lidar_time;
    State m_last_lidar_state;
};

} // namespace lio

#endif // ESTIMATOR_H