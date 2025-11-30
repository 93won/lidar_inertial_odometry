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
#include "VoxelMap.h"
#include "ProbabilisticKernelOptimizer.h"
#include <vector>
#include <memory>
#include <deque>
#include <mutex>
#include <Eigen/Dense>

namespace lio {

// IMU measurement structure
struct IMUData {
    double timestamp;           // Keep double for timestamp precision
    Eigen::Vector3f acc;        // acceleration [m/s²] - float for performance
    Eigen::Vector3f gyr;        // angular velocity [rad/s] - float for performance
    
    IMUData(double t, const Eigen::Vector3f& a, const Eigen::Vector3f& g)
        : timestamp(t), acc(a), gyr(g) {}
};

// State with timestamp for undistortion
struct StateWithTimestamp {
    State state;
    double timestamp;
    
    StateWithTimestamp() : timestamp(0.0) {}
    StateWithTimestamp(const State& s, double t) : state(s), timestamp(t) {}
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
    
    /// Update process noise matrix after parameter changes
    void UpdateProcessNoise();
    
    /// Initialize estimator with first IMU measurement
    void Initialize(const IMUData& first_imu);
    
    /// Initialize gravity direction with multiple IMU samples
    bool GravityInitialization(const std::vector<IMUData>& imu_buffer);
    
    /// Process new IMU measurement
    void ProcessIMU(const IMUData& imu);
    
    /// Process new LiDAR scan
    void ProcessLidar(const LidarData& lidar);
    
    /// Get current state estimate
    State GetCurrentState() const;
    
    /// Get trajectory history
    std::vector<State> GetTrajectory() const;
    
    /// Get local map point cloud
    PointCloudPtr GetMapPointCloud() const {
        std::lock_guard<std::mutex> lock(m_map_mutex);
        return m_map_cloud;
    }
    
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
    
    /// Get current voxel map for visualization
    std::shared_ptr<VoxelMap> GetVoxelMap() const;
    
    /// Get processed (downsampled + range filtered) point cloud for visualization
    PointCloudPtr GetProcessedCloud() const {
        std::lock_guard<std::mutex> lock(m_map_mutex);
        return m_processed_cloud;
    }
    
    /// Print processing time statistics
    void PrintProcessingTimeStatistics() const;
    
    /// Get processing times for each frame (in milliseconds)
    const std::vector<double>& GetProcessingTimes() const {
        return m_processing_times;
    }
    
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
        int max_correspondences = 1000;
        double max_correspondence_distance = 1.0;  // m
        int max_iterations = 10;
        double convergence_threshold = 1e-3;
        
        // Planarity filtering parameters
        double scan_planarity_threshold = 0.1;    // Threshold for input scan downsampling (relaxed)
                                                   // Filters non-planar voxels during downsampling (~50% reduction)
        double map_planarity_threshold = 0.01;    // Threshold for VoxelMap surfel creation (strict)
                                                   // Must be stricter than scan_planarity_threshold
        
        // Local map parameters
        double voxel_size = 0.4;              // m (voxel size for downsampling and VoxelMap)
        double map_voxel_size = 0.2;          // m (deprecated, use voxel_size)
        int max_map_points = 100000;
        double min_range = 0.5;               // m (minimum range for point filtering)
        double max_map_distance = 50.0;       // m
        int max_voxel_hit_count = 10;         // Maximum hit count for voxel occupancy
        int init_hit_count = 1;               // Initial hit count when adding new points
        int voxel_hierarchy_factor = 3;       // L1 voxel factor: L1 = factor × L0 (3, 5, 7, etc.)
        double min_plane_points = 5;
        
        // Frustum culling parameters
        double frustum_fov_horizontal = 90.0; // degrees
        double frustum_fov_vertical = 90.0;   // degrees
        double frustum_max_range = 50.0;      // meters
        
        // Keyframe parameters (distance/rotation based)
        double keyframe_translation_threshold = 0.5;  // meters - triggers keyframe when moved > threshold
        double keyframe_rotation_threshold = 10.0;    // degrees - triggers keyframe when rotated > threshold
        
        // Extrinsics (LiDAR to IMU)
        Eigen::Matrix3f R_il = Eigen::Matrix3f::Identity();
        Eigen::Vector3f t_il = Eigen::Vector3f::Zero();
        
        // Gravity vector (world frame)
        Eigen::Vector3f gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);  // m/s²
        
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
    /// Returns: vector of (p_lidar, plane_normal, plane_d, scan_index)
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, size_t>> 
    FindCorrespondences(const PointCloudPtr scan);
    
    /// Update local map with new scan
    void UpdateLocalMap(const PointCloudPtr scan);
    
    /// Remove old points from local map
    void CleanLocalMap();
    
    /// Extract planar features from point cloud
    void ExtractPlanarFeatures(const PointCloudPtr cloud,
                              std::vector<MapPoint>& features);
    
    /// Compute Jacobians for point-to-plane residuals
    void ComputeLidarJacobians(const std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, size_t>>& correspondences,
                              Eigen::MatrixXf& H,
                              Eigen::VectorXf& residual);
    
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
    
    /// Apply state correction (manifold update for IEKF)
    void ApplyStateCorrection(const Eigen::VectorXf& dx);

private:
    // State estimation
    State m_current_state;
    bool m_initialized;
    double m_last_update_time;
    int m_frame_count;

    unsigned int m_num_valid_correspondences = 0;  // Number of valid correspondences in current scan
    
    // Data buffers (thread-safe)
    mutable std::mutex m_state_mutex;
    std::deque<StateWithTimestamp> m_state_history;  // For undistortion (propagated states)
    std::deque<State> m_trajectory;                  // For trajectory visualization
    
    // Local map storage
    std::vector<MapPoint> m_local_map;
    PointCloudPtr m_map_cloud;
    PointCloudPtr m_processed_cloud;  // Processed (downsampled + range filtered) scan for visualization
    std::shared_ptr<VoxelMap> m_voxel_map;  // Voxel hash map for fast neighbor search
    mutable std::mutex m_map_mutex;
    
    // Last correspondences (for map update after initialization)
    // tuple: (p_lidar, plane_normal, plane_d, scan_index)
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, size_t>> m_last_correspondences;
    
    // Processing statistics
    mutable std::mutex m_stats_mutex;
    Statistics m_statistics;
    std::vector<double> m_processing_times;
    
    // Kalman filter matrices (float for performance, timestamp still double)
    Eigen::Matrix<float, 18, 18> m_process_noise;      // Q
    Eigen::MatrixXf m_measurement_noise;                // R (dynamic size)
    Eigen::Matrix<float, 18, 18> m_state_transition;   // F
    
    // Temporary computation storage
    Eigen::MatrixXf m_jacobian;
    Eigen::VectorXf m_residual_vector;
    Eigen::MatrixXf m_kalman_gain;
    
    // Configuration
    bool m_first_lidar_frame;
    double m_last_lidar_time;
    State m_last_lidar_state;
    
    // Keyframe management
    Eigen::Vector3f m_last_keyframe_position;
    Eigen::Matrix3f m_last_keyframe_rotation;
    bool m_first_keyframe;
    
    // Probabilistic Kernel Optimization
    std::shared_ptr<ProbabilisticKernelOptimizer> m_pko;
};

} // namespace lio

#endif // ESTIMATOR_H