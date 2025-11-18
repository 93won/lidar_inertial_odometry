/**
 * @file      Estimator.cpp
 * @brief     Implementation of tightly-coupled LiDAR-Inertial Odometry Estimator
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025    // --- Forward Propagation (Euler integration) ---
    // dR/dt = R * [omega]_x  =>  R(t+dt) = R(t) * Exp(omega * dt)
    Eigen::Vector3f omega_f = (omega * dt).cast<float>();
    Eigen::Matrix3f R_delta_f = SO3::Exp(omega_f).Matrix();
    Eigen::Matrix3d R_delta = R_delta_f.cast<double>();
    Eigen::Matrix3d R_new = R * R_delta;18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 * 
 * @par Algorithm Overview
 * Tightly-coupled LIO using Iterated Extended Kalman Filter
 * - IMU forward propagation with bias estimation
 * - Point-to-plane residuals as measurements
 * - Iterated Kalman update for state correction
 * - KdTree-based correspondence search
 */

#include "Estimator.h"
#include "LieUtils.h"
#include "PointCloudUtils.h"
#include <chrono>
#include <iostream>
#include <cmath>

namespace lio {

// ============================================================================
// HARDCODED EXTRINSICS FOR R3LIVE DATASET (Livox Avia sensor)
// Frame convention: T_il = Transform from LiDAR to IMU
// ============================================================================
namespace Extrinsics {
    // Translation from LiDAR to IMU (in meters)
    const Eigen::Vector3d t_il(0.04165, 0.02326, -0.0284);
    
    // Rotation from LiDAR to IMU (identity - sensors are aligned)
    const Eigen::Matrix3d R_il = Eigen::Matrix3d::Identity();
}

// ============================================================================
// Constructor & Destructor
// ============================================================================

Estimator::Estimator()
    : m_current_state()
    , m_initialized(false)
    , m_last_update_time(0.0)
    , m_frame_count(0)
    , m_first_lidar_frame(true)
    , m_last_lidar_time(0.0)
{
    // Initialize process noise matrix (Q)
    m_process_noise = Eigen::Matrix<double, 18, 18>::Identity();
    m_process_noise.block<3,3>(0,0) *= m_params.gyr_noise_std * m_params.gyr_noise_std;
    m_process_noise.block<3,3>(3,3) *= m_params.acc_noise_std * m_params.acc_noise_std;
    m_process_noise.block<3,3>(6,6) *= m_params.acc_noise_std * m_params.acc_noise_std;
    m_process_noise.block<3,3>(9,9) *= m_params.gyr_bias_noise_std * m_params.gyr_bias_noise_std;
    m_process_noise.block<3,3>(12,12) *= m_params.acc_bias_noise_std * m_params.acc_bias_noise_std;
    m_process_noise.block<3,3>(15,15) *= m_params.gravity_noise_std * m_params.gravity_noise_std;
    
    // Initialize state transition matrix
    m_state_transition = Eigen::Matrix<double, 18, 18>::Identity();
    
    // Initialize local map
    m_map_cloud = std::make_shared<PointCloud>();
    
    // Initialize statistics
    m_statistics = Statistics();
    m_statistics.total_frames = 0;
    m_statistics.successful_registrations = 0;
    m_statistics.avg_processing_time_ms = 0.0;
    m_statistics.total_distance = 0.0;
    m_statistics.avg_translation_error = 0.0;
    m_statistics.avg_rotation_error = 0.0;
    
    std::cout << "[Estimator] Initialized with hardcoded extrinsics (R3LIVE/Avia dataset)" << std::endl;
    std::cout << "[Estimator] t_il = [" << Extrinsics::t_il.transpose() << "]" << std::endl;
    std::cout << "[Estimator] R_il = Identity" << std::endl;
}

Estimator::~Estimator() {
    std::lock_guard<std::mutex> lock_state(m_state_mutex);
    std::lock_guard<std::mutex> lock_map(m_map_mutex);
    std::lock_guard<std::mutex> lock_stats(m_stats_mutex);
}

// ============================================================================
// Initialization
// ============================================================================

void Estimator::Initialize(const IMUData& first_imu) {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    
    if (m_initialized) {
        std::cerr << "[Estimator] Already initialized!" << std::endl;
        return;
    }
    
    // Initialize state with first IMU measurement
    m_current_state.Reset();
    
    // Initial gravity alignment (assume stationary)
    // Gravity points downward in world frame: [0, 0, -9.81]
    Eigen::Vector3d acc_world = first_imu.acc;
    double acc_norm = acc_world.norm();
    
    if (std::abs(acc_norm - 9.81) < 1.0) {
        // Use accelerometer to initialize gravity direction
        m_current_state.m_gravity = -acc_world.normalized() * 9.81;
        
        // Initialize rotation to align with gravity
        // For now, assume level initial pose (can be improved)
        m_current_state.m_rotation = Eigen::Matrix3d::Identity();
        
        std::cout << "[Estimator] Gravity initialized: " 
                  << m_current_state.m_gravity.transpose() << std::endl;
    } else {
        std::cerr << "[Estimator] Warning: Accelerometer norm = " << acc_norm 
                  << " (expected ~9.81). Using default gravity." << std::endl;
        m_current_state.m_gravity = Eigen::Vector3d(0, 0, -9.81);
    }
    
    // Initialize biases to zero (will be estimated)
    m_current_state.m_gyro_bias.setZero();
    m_current_state.m_acc_bias.setZero();
    
    // Initialize position and velocity
    m_current_state.m_position.setZero();
    m_current_state.m_velocity.setZero();
    
    // Initialize covariance with large uncertainty
    m_current_state.m_covariance = Eigen::Matrix<double, 18, 18>::Identity();
    m_current_state.m_covariance.block<3,3>(0,0) *= 0.1;    // rotation
    m_current_state.m_covariance.block<3,3>(3,3) *= 1.0;    // position
    m_current_state.m_covariance.block<3,3>(6,6) *= 0.5;    // velocity
    m_current_state.m_covariance.block<3,3>(9,9) *= 0.01;   // gyro bias
    m_current_state.m_covariance.block<3,3>(12,12) *= 0.1;  // acc bias
    m_current_state.m_covariance.block<3,3>(15,15) *= 0.01; // gravity
    
    // Add first IMU to buffer
    m_imu_buffer.push_back(first_imu);
    m_last_update_time = first_imu.timestamp;
    
    m_initialized = true;
    std::cout << "[Estimator] Initialization complete at t=" << first_imu.timestamp << std::endl;
}

// ============================================================================
// IMU Processing (Forward Propagation)
// ============================================================================

void Estimator::ProcessIMU(const IMUData& imu) {
    if (!m_initialized) {
        std::cerr << "[Estimator] Not initialized! Call Initialize() first." << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_state_mutex);
    
    // Add to buffer
    m_imu_buffer.push_back(imu);
    
    // Maintain buffer size
    while (m_imu_buffer.size() > static_cast<size_t>(m_params.imu_buffer_size)) {
        m_imu_buffer.pop_front();
    }
    
    // Propagate state
    PropagateState(imu);
}

void Estimator::PropagateState(const IMUData& imu) {
    // Time step
    double dt = imu.timestamp - m_last_update_time;
    if (dt <= 0.0 || dt > 1.0) {
        std::cerr << "[Estimator] Invalid dt: " << dt << std::endl;
        return;
    }
    
    // Get current state
    Eigen::Matrix3d R = m_current_state.m_rotation;
    Eigen::Vector3d p = m_current_state.m_position;
    Eigen::Vector3d v = m_current_state.m_velocity;
    Eigen::Vector3d bg = m_current_state.m_gyro_bias;
    Eigen::Vector3d ba = m_current_state.m_acc_bias;
    Eigen::Vector3d g = m_current_state.m_gravity;
    
    // Corrected measurements
    Eigen::Vector3d omega = imu.gyr - bg;  // angular velocity
    Eigen::Vector3d acc = imu.acc - ba;    // linear acceleration
    
    // --- Forward Propagation (Euler integration) ---
    // dR/dt = R * [omega]_x  =>  R(t+dt) = R(t) * Exp(omega * dt)
    Eigen::Vector3f omega_f = (omega * dt).cast<float>();
    Eigen::Matrix3f R_delta_f = SO3::Exp(omega_f).Matrix();
    Eigen::Matrix3d R_delta = R_delta_f.cast<double>();
    Eigen::Matrix3d R_new = R * R_delta;
    
    // dv/dt = R * acc + g  =>  v(t+dt) = v(t) + (R * acc + g) * dt
    Eigen::Vector3d v_new = v + (R * acc + g) * dt;
    
    // dp/dt = v  =>  p(t+dt) = p(t) + v * dt + 0.5 * (R * acc + g) * dt²
    Eigen::Vector3d p_new = p + v * dt + 0.5 * (R * acc + g) * dt * dt;
    
    // Biases: random walk (no change in mean)
    Eigen::Vector3d bg_new = bg;
    Eigen::Vector3d ba_new = ba;
    Eigen::Vector3d g_new = g;
    
    // --- Covariance Propagation ---
    // P(t+dt) = F * P(t) * F^T + Q * dt
    UpdateProcessNoise(dt);
    
    // Build state transition matrix F (18x18)
    // Simplified linearization around current state
    m_state_transition.setIdentity();
    
    // dR depends on omega (rotation dynamics)
    Eigen::Vector3f omega_skew_f = omega.cast<float>();
    Eigen::Matrix3d omega_skew = Hat(omega_skew_f).cast<double>();
    m_state_transition.block<3,3>(0,0) = Eigen::Matrix3d::Identity() - omega_skew * dt;
    m_state_transition.block<3,3>(0,9) = -R * dt;  // rotation vs gyro bias
    
    // dv depends on R and acc (velocity dynamics)
    Eigen::Vector3f acc_skew_f = acc.cast<float>();
    Eigen::Matrix3d acc_skew = Hat(acc_skew_f).cast<double>();
    m_state_transition.block<3,3>(6,0) = -R * acc_skew * dt;  // velocity vs rotation
    m_state_transition.block<3,3>(6,6) = Eigen::Matrix3d::Identity();
    m_state_transition.block<3,3>(6,12) = -R * dt;  // velocity vs acc bias
    m_state_transition.block<3,3>(6,15) = Eigen::Matrix3d::Identity() * dt;  // velocity vs gravity
    
    // dp depends on v (position dynamics)
    m_state_transition.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
    m_state_transition.block<3,3>(3,6) = Eigen::Matrix3d::Identity() * dt;  // position vs velocity
    
    // Propagate covariance
    Eigen::Matrix<double, 18, 18> P = m_current_state.m_covariance;
    m_current_state.m_covariance = m_state_transition * P * m_state_transition.transpose() 
                                   + m_process_noise * dt;
    
    // Update state
    m_current_state.m_rotation = R_new;
    m_current_state.m_position = p_new;
    m_current_state.m_velocity = v_new;
    m_current_state.m_gyro_bias = bg_new;
    m_current_state.m_acc_bias = ba_new;
    m_current_state.m_gravity = g_new;
    
    m_last_update_time = imu.timestamp;
}

// ============================================================================
// LiDAR Processing (Iterated Kalman Update)
// ============================================================================

void Estimator::ProcessLidar(const LidarData& lidar) {
    if (!m_initialized) {
        std::cerr << "[Estimator] Not initialized! Cannot process LiDAR." << std::endl;
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock_state(m_state_mutex);
    std::lock_guard<std::mutex> lock_map(m_map_mutex);
    
    // First frame: initialize map
    if (m_first_lidar_frame) {
        std::cout << "[Estimator] First LiDAR frame - initializing map" << std::endl;
        UpdateLocalMap(lidar.cloud);
        m_first_lidar_frame = false;
        m_last_lidar_time = lidar.timestamp;
        m_last_lidar_state = m_current_state;
        m_frame_count++;
        return;
    }
    
    // Motion check: skip if not enough motion
    double distance = (m_current_state.m_position - m_last_lidar_state.m_position).norm();
    if (distance < m_params.min_motion_threshold) {
        std::cout << "[Estimator] Skipping frame (insufficient motion: " << distance << " m)" << std::endl;
        return;
    }
    
    // Undistort point cloud using IMU integration
    PointCloudPtr undistorted_cloud = lidar.cloud;
    if (m_params.enable_undistortion) {
        // TODO: Implement motion undistortion using IMU buffer
        // For now, skip undistortion
    }
    
    // Downsample point cloud
    // TODO: Implement voxel downsampling
    PointCloudPtr scan = undistorted_cloud;
    
    // Iterated Kalman Filter Update
    UpdateWithLidar(lidar);
    
    // Update local map with new scan
    UpdateLocalMap(scan);
    
    // Clean old points from map
    CleanLocalMap();
    
    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    double processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    {
        std::lock_guard<std::mutex> lock_stats(m_stats_mutex);
        m_processing_times.push_back(processing_time);
        m_statistics.total_frames++;
        m_statistics.avg_processing_time_ms = 
            (m_statistics.avg_processing_time_ms * (m_statistics.total_frames - 1) + processing_time) 
            / m_statistics.total_frames;
    }
    
    // Store trajectory
    m_trajectory.push_back(m_current_state);
    if (m_trajectory.size() > 10000) {
        m_trajectory.pop_front();
    }
    
    // Update tracking
    m_last_lidar_time = lidar.timestamp;
    m_last_lidar_state = m_current_state;
    m_frame_count++;
    
    std::cout << "[Estimator] Frame " << m_frame_count 
              << " processed in " << processing_time << " ms" << std::endl;
}

void Estimator::UpdateWithLidar(const LidarData& lidar) {
    // TODO: Implement Iterated Kalman Filter update
    // 1. Transform scan to world frame using current state
    // 2. Find correspondences (5 nearest neighbors per point)
    // 3. Fit planes to neighbors
    // 4. Compute point-to-plane residuals
    // 5. Compute Jacobians H
    // 6. Kalman update: K = P*H^T*(H*P*H^T + R)^-1, x = x + K*r
    // 7. Iterate 4-5 times
    
    std::cout << "[Estimator] LiDAR update (TODO: implement Iterated Kalman)" << std::endl;
}

// ============================================================================
// Correspondence Finding
// ============================================================================

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> 
Estimator::FindCorrespondences(const PointCloudPtr scan) {
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> correspondences;
    
    // TODO: Implement correspondence search
    // 1. Build KdTree from map
    // 2. For each point in scan:
    //    - Transform to world frame
    //    - Find 5 nearest neighbors
    //    - Fit plane to neighbors
    //    - Add (point, plane_normal) to correspondences
    
    return correspondences;
}

// ============================================================================
// Local Map Management
// ============================================================================

void Estimator::UpdateLocalMap(const PointCloudPtr scan) {
    // Transform scan to world frame
    Eigen::Matrix3d R = m_current_state.m_rotation;
    Eigen::Vector3d t = m_current_state.m_position;
    
    int added_count = 0;
    for (const auto& pt : *scan) {
        // LiDAR point in sensor frame
        Eigen::Vector3d p_lidar(pt.x, pt.y, pt.z);
        
        // Transform: p_world = R_wb * (R_il * p_lidar + t_il) + t_wb
        Eigen::Vector3d p_imu = Extrinsics::R_il * p_lidar + Extrinsics::t_il;
        Eigen::Vector3d p_world = R * p_imu + t;
        
        // Add to map cloud
        Point3D map_pt;
        map_pt.x = static_cast<float>(p_world.x());
        map_pt.y = static_cast<float>(p_world.y());
        map_pt.z = static_cast<float>(p_world.z());
        map_pt.intensity = pt.intensity;
        map_pt.offset_time = pt.offset_time;
        m_map_cloud->push_back(map_pt);
        added_count++;
    }
    
    std::cout << "[Estimator] Map updated: " << m_map_cloud->size() 
              << " points (added " << added_count << ")" << std::endl;
}

void Estimator::CleanLocalMap() {
    // TODO: Implement proper map cleaning
    // For now, just limit the size by removing oldest points
    
    size_t map_size = m_map_cloud->size();
    size_t max_size = static_cast<size_t>(m_params.max_map_points);
    
    if (map_size > max_size) {
        // Create new cloud with recent points
        auto new_cloud = std::make_shared<PointCloud>();
        int start_idx = map_size - max_size;
        int idx = 0;
        
        for (const auto& pt : *m_map_cloud) {
            if (idx >= start_idx) {
                new_cloud->push_back(pt);
            }
            idx++;
        }
        
        m_map_cloud = new_cloud;
        std::cout << "[Estimator] Map cleaned: " << m_map_cloud->size() << " points" << std::endl;
    }
}

// ============================================================================
// Jacobian Computation
// ============================================================================

void Estimator::ComputeLidarJacobians(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& correspondences,
    Eigen::MatrixXd& H,
    Eigen::VectorXd& residual) 
{
    // TODO: Implement Jacobian computation
    // For each correspondence (point, plane_normal):
    //   residual = n^T * (R * (R_il * p_lidar + t_il) + t - plane_point)
    //   dR/dtheta = [point]_x (rotation perturbation)
    //   dt/dt = I (translation perturbation)
    
    int num_corr = correspondences.size();
    H.resize(num_corr, 18);
    residual.resize(num_corr);
    
    H.setZero();
    residual.setZero();
}

// ============================================================================
// Noise Updates
// ============================================================================

void Estimator::UpdateProcessNoise(double dt) {
    // Scale noise by time step (already set in constructor)
    // Q matrix is used as Q * dt in propagation
}

void Estimator::UpdateMeasurementNoise(int num_correspondences) {
    // Measurement noise R is diagonal (independent residuals)
    m_measurement_noise = Eigen::MatrixXd::Identity(num_correspondences, num_correspondences);
    m_measurement_noise *= m_params.lidar_noise_std * m_params.lidar_noise_std;
}

// ============================================================================
// State Getters
// ============================================================================

State Estimator::GetCurrentState() const {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    return m_current_state;
}

std::vector<State> Estimator::GetTrajectory() const {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    return std::vector<State>(m_trajectory.begin(), m_trajectory.end());
}

Estimator::Statistics Estimator::GetStatistics() const {
    std::lock_guard<std::mutex> lock(m_stats_mutex);
    return m_statistics;
}

// ============================================================================
// Undistortion & Interpolation (Placeholder)
// ============================================================================

PointCloudPtr Estimator::UndistortPointCloud(
    const PointCloudPtr cloud,
    double scan_start_time,
    double scan_end_time) 
{
    // TODO: Implement motion undistortion
    // 1. For each point with offset_time:
    //    - Interpolate state at (scan_start_time + offset_time)
    //    - Transform point using interpolated state
    
    return cloud;
}

State Estimator::InterpolateState(double timestamp) const {
    // TODO: Implement state interpolation using IMU buffer
    // Linear interpolation between nearest IMU measurements
    
    return m_current_state;
}

// ============================================================================
// Feature Extraction (Placeholder)
// ============================================================================

void Estimator::ExtractPlanarFeatures(
    const PointCloudPtr cloud,
    std::vector<MapPoint>& features) 
{
    // TODO: Implement planar feature extraction
    // For each point, check local neighborhood planarity
    
    features.clear();
}

} // namespace lio
