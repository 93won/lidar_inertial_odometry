/**
 * @file      Estimator.cpp
 * @brief     Implementation of tightly-coupled LiDAR-Inertial Odometry Estimator
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright  Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Estimator.h"
#include "LieUtils.h"
#include "PointCloudUtils.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lio {

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
    , m_first_keyframe(true)
    , m_last_keyframe_position(Eigen::Vector3f::Zero())
    , m_last_keyframe_rotation(Eigen::Matrix3f::Identity())
{
    // Initialize extrinsics with default R3LIVE/Avia values
    m_params.R_il = Eigen::Matrix3f::Identity();
    m_params.t_il = Eigen::Vector3f(0.04165f, 0.02326f, -0.0284f);
    
    // Initialize process noise matrix (Q)
    m_process_noise = Eigen::Matrix<float, 18, 18>::Identity();
    m_process_noise.block<3,3>(0,0) *= m_params.gyr_noise_std * m_params.gyr_noise_std;
    m_process_noise.block<3,3>(3,3) *= m_params.acc_noise_std * m_params.acc_noise_std;
    m_process_noise.block<3,3>(6,6) *= m_params.acc_noise_std * m_params.acc_noise_std;
    m_process_noise.block<3,3>(9,9) *= m_params.gyr_bias_noise_std * m_params.gyr_bias_noise_std;
    m_process_noise.block<3,3>(12,12) *= m_params.acc_bias_noise_std * m_params.acc_bias_noise_std;
    m_process_noise.block<3,3>(15,15) *= m_params.gravity_noise_std * m_params.gravity_noise_std;
    
    // Initialize state transition matrix
    m_state_transition = Eigen::Matrix<float, 18, 18>::Identity();
    
    // Initialize local map
    m_map_cloud = std::make_shared<PointCloud>();
    m_processed_cloud = std::make_shared<PointCloud>();
    
    // Initialize statistics
    m_statistics = Statistics();
    m_statistics.total_frames = 0;
    m_statistics.successful_registrations = 0;
    m_statistics.avg_processing_time_ms = 0.0;
    m_statistics.total_distance = 0.0;
    m_statistics.avg_translation_error = 0.0;
    m_statistics.avg_rotation_error = 0.0;
    
    // Initialize Probabilistic Kernel Optimizer
    PKOConfig pko_config;
    pko_config.use_adaptive = true;
    pko_config.min_scale_factor = 0.001;
    pko_config.max_scale_factor = 10.0;
    pko_config.num_alpha_segments = 25;
    pko_config.truncated_threshold = 10.0;
    pko_config.gmm_components = 2;
    pko_config.gmm_sample_size = 100;
    m_pko = std::make_shared<ProbabilisticKernelOptimizer>(pko_config);
    
    spdlog::info("[Estimator] Initialized with default extrinsics (R3LIVE/Avia dataset)");
    spdlog::info("[Estimator] t_il = [{:.5f}, {:.5f}, {:.5f}]", 
                 m_params.t_il.x(), m_params.t_il.y(), m_params.t_il.z());
    spdlog::info("[Estimator] R_il = Identity");
}

Estimator::~Estimator() {
    std::lock_guard<std::mutex> lock_state(m_state_mutex);
    std::lock_guard<std::mutex> lock_map(m_map_mutex);
    std::lock_guard<std::mutex> lock_stats(m_stats_mutex);
}

void Estimator::UpdateProcessNoise() {
    // Update process noise matrix (Q) with current parameters
    m_process_noise = Eigen::Matrix<float, 18, 18>::Identity();
    m_process_noise.block<3,3>(0,0) *= m_params.gyr_noise_std * m_params.gyr_noise_std;
    m_process_noise.block<3,3>(3,3) *= m_params.acc_noise_std * m_params.acc_noise_std;
    m_process_noise.block<3,3>(6,6) *= m_params.acc_noise_std * m_params.acc_noise_std;
    m_process_noise.block<3,3>(9,9) *= m_params.gyr_bias_noise_std * m_params.gyr_bias_noise_std;
    m_process_noise.block<3,3>(12,12) *= m_params.acc_bias_noise_std * m_params.acc_bias_noise_std;
    m_process_noise.block<3,3>(15,15) *= m_params.gravity_noise_std * m_params.gravity_noise_std;
    
    spdlog::debug("[Estimator] Process noise matrix updated with current IMU parameters");
}

// ============================================================================
// Initialization
// ============================================================================

bool Estimator::GravityInitialization(const std::vector<IMUData>& imu_buffer) {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    
    if (m_initialized) {
        spdlog::warn("[Estimator] Already initialized!");
        return false;
    }
    
    // 1. Check minimum number of samples (need at least 20 for good statistics)
    if (imu_buffer.size() < 20) {
        spdlog::error("[Estimator] Not enough IMU data for initialization (need >= 20 samples, got {})", 
                     imu_buffer.size());
        return false;
    }
    
    spdlog::info("[Estimator] Starting gravity initialization with {} IMU samples", imu_buffer.size());
    
    // 2. Compute mean acceleration and gyroscope (running average)
    Eigen::Vector3f mean_acc = Eigen::Vector3f::Zero();
    Eigen::Vector3f mean_gyr = Eigen::Vector3f::Zero();
    
    for (const auto& imu : imu_buffer) {
        mean_acc += imu.acc;
        mean_gyr += imu.gyr;
    }
    mean_acc /= static_cast<float>(imu_buffer.size());
    mean_gyr /= static_cast<float>(imu_buffer.size());
    
    // 3. Compute variance to check if robot is stationary
    float acc_variance = 0.0f;
    float gyr_variance = 0.0f;
    
    for (const auto& imu : imu_buffer) {
        acc_variance += (imu.acc - mean_acc).squaredNorm();
        gyr_variance += (imu.gyr - mean_gyr).squaredNorm();
    }
    acc_variance /= static_cast<float>(imu_buffer.size());
    gyr_variance /= static_cast<float>(imu_buffer.size());
    
    // 4. Check if robot is stationary (low variance)
    if (acc_variance > 0.5f) {
        spdlog::warn("[Estimator] High accelerometer variance ({:.3f}), robot may be moving!", acc_variance);
        spdlog::warn("[Estimator] Initialization may be inaccurate. Please keep robot stationary.");
    }
    
    if (gyr_variance > 0.01f) {
        spdlog::warn("[Estimator] High gyroscope variance ({:.3f}), robot may be rotating!", gyr_variance);
    }
    
    // 5. Initialize state
    m_current_state.Reset();
    
    // 6. Check accelerometer norm (should be ~g if stationary)
    float acc_norm = mean_acc.norm();
    float gravity_magnitude = m_params.gravity.norm();
    
    if (std::abs(acc_norm - gravity_magnitude) > 1.5f) {
        spdlog::error("[Estimator] Accelerometer norm = {:.3f} m/s² (expected ~{:.3f})", acc_norm, gravity_magnitude);
        spdlog::error("[Estimator] Sensor may be moving or miscalibrated. Initialization failed.");
        return false;
    }
    
    // 7. Initialize gravity vector (measured acceleration = -gravity in sensor frame)
    Eigen::Vector3f gravity_measured = -mean_acc.normalized() * gravity_magnitude;
    
    // 8. Set initial gravity (not yet aligned)
    m_current_state.m_gravity = gravity_measured;
    
    // 9. Initialize rotation to identity (will be aligned after)
    m_current_state.m_rotation = Eigen::Matrix3f::Identity();
    
    spdlog::info("[Estimator] Initial gravity (sensor frame): [{:.3f}, {:.3f}, {:.3f}]", 
                 gravity_measured.x(), gravity_measured.y(), gravity_measured.z());
    
    // 10. Gravity alignment: align world frame so gravity points to configured gravity direction
    // This rotates all states to make gravity vertical
    Eigen::Vector3f gravity_target = m_params.gravity;
    Eigen::Quaternionf q_align = Eigen::Quaternionf::FromTwoVectors(
        m_current_state.m_gravity.normalized(),
        gravity_target.normalized()
    );
    Eigen::Matrix3f R_align = q_align.toRotationMatrix();
    
   
    
    // Apply alignment rotation to all states
    m_current_state.m_rotation = R_align * m_current_state.m_rotation;  // Rotate orientation
    m_current_state.m_position = R_align * m_current_state.m_position;  // Rotate position (zero)
    m_current_state.m_velocity = R_align * m_current_state.m_velocity;  // Rotate velocity (zero)
    m_current_state.m_gravity = R_align * m_current_state.m_gravity;    // Rotate gravity -> aligned direction
    
  
    // 11. Initialize gyroscope bias (stationary gyro reading = bias)
    m_current_state.m_gyro_bias = mean_gyr;
    
    // 12. Initialize accelerometer bias from stationary measurements
    // Stationary condition: acc_measured = -g + bias
    // After gravity alignment: mean_acc ≈ -R_align^T * g_world + bias
    // Therefore: bias = mean_acc + R_align^T * g_world
    //                 = mean_acc + R_align^T * configured_gravity
    Eigen::Vector3f g_aligned = m_params.gravity;

    // Correct formula: bias = mean_acc + R^T * g
    Eigen::Vector3f acc_bias_estimate = mean_acc + m_current_state.m_rotation.transpose() * g_aligned;
    m_current_state.m_acc_bias = acc_bias_estimate;
    
    // 13. Initialize position and velocity to zero
    m_current_state.m_position.setZero();
    m_current_state.m_velocity.setZero();
    
    // 14. Initialize covariance with appropriate uncertainty
    m_current_state.m_covariance = Eigen::Matrix<float, 18, 18>::Identity();
    m_current_state.m_covariance.block<3,3>(0,0) *= 0.01f;   // rotation (small, well aligned)
    m_current_state.m_covariance.block<3,3>(3,3) *= 1.0f;    // position (unknown)
    m_current_state.m_covariance.block<3,3>(6,6) *= 0.1f;    // velocity (should be zero)
    m_current_state.m_covariance.block<3,3>(9,9) *= 0.001f;  // gyro bias (estimated from data)
    m_current_state.m_covariance.block<3,3>(12,12) *= 0.01f; // acc bias (estimated from data)
    m_current_state.m_covariance.block<3,3>(15,15) *= 0.001f; // gravity (well aligned)
    
    // 15. Set timestamp
    m_last_update_time = imu_buffer.back().timestamp;
    
    // 16. Mark as initialized
    m_initialized = true;
    
    spdlog::info("[Estimator] ===============================================================");
    spdlog::info("[Estimator] Gravity initialization SUCCESSFUL at t={:.6f}", m_last_update_time);
    spdlog::info("[Estimator] Statistics:");
    spdlog::info("  - IMU samples: {}", imu_buffer.size());
    spdlog::info("  - Acc variance: {:.6f} m^2/s^4", acc_variance);
    spdlog::info("  - Gyr variance: {:.6f} rad^2/s^2", gyr_variance);
    spdlog::info("  - Acc norm: {:.3f} m/s^2 (expected: {:.3f})", acc_norm, gravity_magnitude);
    spdlog::info("[Estimator] ===============================================================");
    
    return true;
}

void Estimator::Initialize(const IMUData& first_imu) {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    
    if (m_initialized) {
        spdlog::warn("[Estimator] Already initialized!");
        return;
    }
    
    spdlog::warn("[Estimator] Simple initialization with single IMU sample");
    spdlog::warn("[Estimator] Consider using GravityInitialization() with multiple samples for better accuracy");
    
    // Initialize state with first IMU measurement
    m_current_state.Reset();
    
    // Initial gravity alignment (assume stationary)
    Eigen::Vector3f acc_world = first_imu.acc;
    float acc_norm = acc_world.norm();
    float gravity_magnitude = m_params.gravity.norm();
    
    if (std::abs(acc_norm - gravity_magnitude) < 1.0f) {
        // Use accelerometer to initialize gravity direction
        m_current_state.m_gravity = -acc_world.normalized() * gravity_magnitude;
        
        // Gravity alignment: rotate world frame so gravity points to configured gravity
        Eigen::Vector3f gravity_target = m_params.gravity;
        Eigen::Quaternionf q_align = Eigen::Quaternionf::FromTwoVectors(
            m_current_state.m_gravity.normalized(),
            gravity_target.normalized()
        );
        Eigen::Matrix3f R_align = q_align.toRotationMatrix();
        
        // Apply alignment to initial rotation
        m_current_state.m_rotation = R_align;
        m_current_state.m_gravity = gravity_target;
        
        spdlog::info("[Estimator] Gravity initialized: [{:.3f}, {:.3f}, {:.3f}]",
                     m_current_state.m_gravity.x(), 
                     m_current_state.m_gravity.y(), 
                     m_current_state.m_gravity.z());
    } else {
        spdlog::warn("[Estimator] Accelerometer norm = {:.3f} (expected ~{:.3f}). Using default gravity.", acc_norm, gravity_magnitude);
        m_current_state.m_gravity = m_params.gravity;
        m_current_state.m_rotation = Eigen::Matrix3f::Identity();
    }
    
    // Initialize biases to zero (will be estimated)
    m_current_state.m_gyro_bias.setZero();
    m_current_state.m_acc_bias.setZero();
    
    // Initialize position and velocity
    m_current_state.m_position.setZero();
    m_current_state.m_velocity.setZero();
    
    // Initialize covariance with large uncertainty
    m_current_state.m_covariance = Eigen::Matrix<float, 18, 18>::Identity();
    m_current_state.m_covariance.block<3,3>(0,0) *= 0.1f;    // rotation
    m_current_state.m_covariance.block<3,3>(3,3) *= 1.0f;    // position
    m_current_state.m_covariance.block<3,3>(6,6) *= 0.5f;    // velocity
    m_current_state.m_covariance.block<3,3>(9,9) *= 0.01f;   // gyro bias
    m_current_state.m_covariance.block<3,3>(12,12) *= 0.1f;  // acc bias
    m_current_state.m_covariance.block<3,3>(15,15) *= 0.01f; // gravity
    
    m_last_update_time = first_imu.timestamp;
    
    m_initialized = true;
    spdlog::info("[Estimator] Initialization complete at t={:.6f}", first_imu.timestamp);
}

// ============================================================================
// IMU Processing (Forward Propagation)
// ============================================================================

void Estimator::ProcessIMU(const IMUData& imu_data) {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    
    if (!m_initialized) {
        spdlog::warn("[Estimator] Not initialized. Call Initialize() first.");
        return;
    }
    
    // Propagate state using current IMU measurement
    PropagateState(imu_data);
    
    // Save state history for undistortion between LiDAR frames
    StateWithTimestamp state_snapshot;
    state_snapshot.state = m_current_state;
    state_snapshot.timestamp = imu_data.timestamp;
    m_state_history.push_back(state_snapshot);
}

void Estimator::PropagateState(const IMUData& imu) {
    // Time step (only timestamp is double)
    double dt = imu.timestamp - m_last_update_time;

    m_last_update_time = imu.timestamp;

    if (dt <= 0.0 || dt > 1.0) {
        return;
    }
    float dt_f = static_cast<float>(dt);
    
    // Get current state (all float)
    Eigen::Matrix3f R = m_current_state.m_rotation;
    Eigen::Vector3f p = m_current_state.m_position;
    Eigen::Vector3f v = m_current_state.m_velocity;
    Eigen::Vector3f bg = m_current_state.m_gyro_bias;
    Eigen::Vector3f ba = m_current_state.m_acc_bias;
    Eigen::Vector3f g = m_current_state.m_gravity;
    
    // Corrected measurements (already float from IMUData)
    Eigen::Vector3f omega = imu.gyr - bg;  // angular velocity
    Eigen::Vector3f acc = imu.acc - ba;    // linear acceleration

    Eigen::Vector3f acc_world = R * acc + g;
    
    // --- Forward Propagation (Euler integration) ---
    // dR/dt = R * [omega]_x  =>  R(t+dt) = R(t) * Exp(omega * dt)
    Eigen::Vector3f omega_dt = omega * dt_f;
    Eigen::Matrix3f R_delta = SO3::Exp(omega_dt).Matrix();
    Eigen::Matrix3f R_new = R * R_delta;
    
    // dv/dt = R * acc + g  =>  v(t+dt) = v(t) + (R * acc + g) * dt
    Eigen::Vector3f v_new = v + (R * acc + g) * dt_f;
    
    // dp/dt = v  =>  p(t+dt) = p(t) + v * dt + 0.5 * (R * acc + g) * dt²
    Eigen::Vector3f p_new = p + v * dt_f + 0.5f * (R * acc + g) * dt_f * dt_f;
    
    // Biases: random walk (no change in mean)
    Eigen::Vector3f bg_new = bg;
    Eigen::Vector3f ba_new = ba;
    Eigen::Vector3f g_new = g;
    
    // --- Covariance Propagation ---
    // P(t+dt) = F * P(t) * F^T + Q * dt
    UpdateProcessNoise(dt);
    
    // Build state transition matrix F (18x18)
    // Simplified linearization around current state
    m_state_transition.setIdentity();
    
    // dR depends on omega (rotation dynamics)
    Eigen::Matrix3f omega_skew = Hat(omega);
    m_state_transition.block<3,3>(0,0) = Eigen::Matrix3f::Identity() - omega_skew * dt_f;
    m_state_transition.block<3,3>(0,9) = -R * dt_f;  // rotation vs gyro bias
    
    // dv depends on R and acc (velocity dynamics)
    Eigen::Matrix3f acc_skew = Hat(acc);
    m_state_transition.block<3,3>(6,0) = -R * acc_skew * dt_f;  // velocity vs rotation
    m_state_transition.block<3,3>(6,6) = Eigen::Matrix3f::Identity();
    m_state_transition.block<3,3>(6,12) = -R * dt_f;  // velocity vs acc bias
    m_state_transition.block<3,3>(6,15) = Eigen::Matrix3f::Identity() * dt_f;  // velocity vs gravity
    
    // dp depends on v (position dynamics)
    m_state_transition.block<3,3>(3,3) = Eigen::Matrix3f::Identity();
    m_state_transition.block<3,3>(3,6) = Eigen::Matrix3f::Identity() * dt_f;  // position vs velocity
    
    // Propagate covariance
    Eigen::Matrix<float, 18, 18> P = m_current_state.m_covariance;
    m_current_state.m_covariance = m_state_transition * P * m_state_transition.transpose() 
                                   + m_process_noise * dt_f;
    
    // Update state
    m_current_state.m_rotation = R_new;
    m_current_state.m_position = p_new;
    m_current_state.m_velocity = v_new;
    m_current_state.m_gyro_bias = bg_new;
    m_current_state.m_acc_bias = ba_new;
    m_current_state.m_gravity = g_new;
    
}

// ============================================================================
// LiDAR Processing (Iterated Kalman Update)
// ============================================================================

void Estimator::ProcessLidar(const LidarData& lidar) {
    if (!m_initialized) {
        spdlog::error("[Estimator] Not initialized! Cannot process LiDAR.");
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock_state(m_state_mutex);
    std::lock_guard<std::mutex> lock_map(m_map_mutex);
    
    // === 1. Preprocessing: Downsampling + Undistortion + Range filtering ===
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    
    // Temporal bin-based downsampling (if temporal_bins > 0)
    PointCloudPtr downsampled_scan;
    if (m_params.temporal_bins > 0) {
        downsampled_scan = TemporalBinDownsample(
            lidar.cloud,
            m_params.temporal_bins,
            static_cast<float>(m_params.scan_duration)
        );
        
        // Optionally apply voxel downsample after temporal bin
        if (m_params.temporal_then_voxel) {
            PointCloudPtr voxel_filtered = std::make_shared<PointCloud>();
            VoxelGrid scan_filter;
            scan_filter.SetInputCloud(downsampled_scan);
            scan_filter.SetLeafSize(static_cast<float>(m_params.voxel_size));
            scan_filter.SetPlanarityFilter(true);
            scan_filter.SetHierarchyFactor(m_params.voxel_hierarchy_factor);
            scan_filter.Filter(*voxel_filtered);
            downsampled_scan = voxel_filtered;
        }
    } else {
        // Fallback to voxel-based downsampling only
        downsampled_scan = std::make_shared<PointCloud>();
        VoxelGrid scan_filter;
        scan_filter.SetInputCloud(lidar.cloud);
        scan_filter.SetLeafSize(static_cast<float>(m_params.voxel_size));
        scan_filter.SetPlanarityFilter(true);
        scan_filter.SetHierarchyFactor(m_params.voxel_hierarchy_factor);
        scan_filter.Filter(*downsampled_scan);
    }

    PointCloudPtr undistorted_cloud = downsampled_scan;
    if (m_params.enable_undistortion) {
        double scan_start_time = m_first_lidar_frame ? lidar.timestamp - 0.1 : m_last_lidar_time;
        undistorted_cloud = UndistortPointCloud(
            downsampled_scan,
            scan_start_time,
            lidar.timestamp
        );
    }
    
    m_state_history.clear();

    PointCloudPtr range_filtered_scan = std::make_shared<PointCloud>();
    unsigned int initial_size = undistorted_cloud->size();
    const float min_range = static_cast<float>(m_params.min_range);
    const float max_range = static_cast<float>(m_params.max_map_distance);
    
    for(unsigned int i = 0; i < initial_size; ++i) {
        const auto& point = undistorted_cloud->at(i);
        float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        if (range >= min_range && range <= max_range) {
            range_filtered_scan->push_back(point);
        }
    }
    
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    
    LidarData downsampled_lidar(lidar.timestamp, range_filtered_scan);
    m_processed_cloud = range_filtered_scan;
    
    // First frame: initialize map with downsampled cloud
    if (m_first_lidar_frame) {
        spdlog::info("[Estimator] First LiDAR frame - initializing map");
        UpdateLocalMap(range_filtered_scan);
        m_first_lidar_frame = false;
        m_last_lidar_time = lidar.timestamp;
        m_last_lidar_state = m_current_state;
        m_frame_count++;
        return;
    }
    
    // === 2. IEKF Update ===
    auto iekf_start = std::chrono::high_resolution_clock::now();
    UpdateWithLidar(downsampled_lidar);
    auto iekf_end = std::chrono::high_resolution_clock::now();
    double iekf_time = std::chrono::duration<double, std::milli>(iekf_end - iekf_start).count();
    
    // === 3. Map Update ===
    auto map_start = std::chrono::high_resolution_clock::now();
    UpdateLocalMap(range_filtered_scan);
    auto map_end = std::chrono::high_resolution_clock::now();
    double map_time = std::chrono::duration<double, std::milli>(map_end - map_start).count();
    
    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    double processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    {
        std::lock_guard<std::mutex> lock_stats(m_stats_mutex);
        
        // Accumulate timing stats
        m_sum_preprocess_time += preprocess_time;
        m_sum_iekf_time += iekf_time;
        m_sum_map_time += map_time;
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
    
    // Log detailed timing every 100 frames
    if (m_frame_count % 100 == 0) {
        double avg_preprocess = m_sum_preprocess_time / 100.0;
        double avg_iekf = m_sum_iekf_time / 100.0;
        double avg_map = m_sum_map_time / 100.0;
        double avg_total = avg_preprocess + avg_iekf + avg_map;
        
        spdlog::info("[Estimator] Frame {}: Total={:.2f}ms (Preprocess={:.2f}ms, IEKF={:.2f}ms, Map={:.2f}ms)", 
                     m_frame_count, avg_total, avg_preprocess, avg_iekf, avg_map);
        
        // Reset accumulators
        m_sum_preprocess_time = 0.0;
        m_sum_iekf_time = 0.0;
        m_sum_map_time = 0.0;
    }
}

void Estimator::UpdateWithLidar(const LidarData& lidar) {
     // Reset PKO for new scan
      
    // Nested Iterated Extended Kalman Filter (IEKF)
    // Outer loop: Re-linearization (find new correspondences)
    // Inner loop: Convergence (update state with same correspondences)
     const int max_outer_iterations = m_params.max_iterations; // Re-linearization iterations from config
     const int max_inner_iterations = 4;                       // State update iterations per correspondence
     bool converged = false;

     int total_inner_iters = 0;
     double residual_normalization_scale = 1.0; // For PKO normalization

     // Last correspondences found (for map update after loop)
     std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, size_t>> correspondences;

     for (int outer_iter = 0; outer_iter < max_outer_iterations; outer_iter++)
     {

         if (m_pko)
         {
             m_pko->Reset();
         }

         // OUTER LOOP: Find correspondences at current state
         correspondences = FindCorrespondences(lidar.cloud);

         if (correspondences.empty())
         {
             break;
         }

         // INNER LOOP: Update state multiple times with SAME correspondences
         bool inner_converged = false;
         Eigen::Matrix<float, 18, 18> G_final;

         for (int inner_iter = 0; inner_iter < max_inner_iterations; inner_iter++)
         {
             total_inner_iters++;

             // Compute Jacobian H and residual at current state (with SAME correspondences)
             Eigen::MatrixXf H;
             Eigen::VectorXf residual;
             ComputeLidarJacobians(correspondences, H, residual);

             // Calculate residual normalization scale (only once at first iteration)
             if (inner_iter == 0 && residual.size() > 0)
             {
                 std::vector<double> residuals_for_scale;
                 residuals_for_scale.reserve(residual.size());
                 for (int i = 0; i < residual.size(); i++)
                 {
                     residuals_for_scale.push_back(static_cast<double>(residual(i)));
                 }
                //  std::sort(residuals_for_scale.begin(), residuals_for_scale.end());

                 double mean = std::accumulate(residuals_for_scale.begin(), residuals_for_scale.end(), 0.0) / residuals_for_scale.size();
                 double variance = 0.0;
                 for (double val : residuals_for_scale)
                 {
                     variance += (val - mean) * (val - mean);
                 }
                 variance /= residuals_for_scale.size();
                 double std_dev = std::sqrt(variance);

                 // Calculate residual normalization scale (std / 3)
                 residual_normalization_scale = std_dev / 3.0;

                 // spdlog::info("[PKO] Residual normalization scale (std/3): {:.6f}", residual_normalization_scale);
             }

             // Calculate adaptive Huber scale using PKO with normalized residuals
             double adaptive_huber_delta = 1.0; // Default value
             if (m_pko)
             {
                 // Convert residuals from float to double and normalize
                 std::vector<double> residuals_double(residual.size());
                 for (int i = 0; i < residual.size(); i++)
                 {
                     double normalized_residual = static_cast<double>(residual(i)) / std::max(residual_normalization_scale, 1e-6);
                     residuals_double[i] = normalized_residual;
                 }

                 // Calculate adaptive scale factor (alpha) using normalized residuals
                 adaptive_huber_delta = m_pko->CalculateScaleFactor(residuals_double);

                 // spdlog::info("[PKO] Outer iter {}, Inner iter {}: alpha = {:.6f}", outer_iter, inner_iter, adaptive_huber_delta);
             }

             // Normalize residuals for numerical stability
             float normalization_scale_f = static_cast<float>(residual_normalization_scale);
             Eigen::VectorXf normalized_residual = residual / std::max(normalization_scale_f, 1e-6f);

             int num_corr = correspondences.size();

             // Compute Huber weights using normalized residuals and adaptive delta
             float huber_threshold = static_cast<float>(adaptive_huber_delta);
             Eigen::VectorXf huber_weights(num_corr);

             for (int i = 0; i < num_corr; i++)
             {
                 float abs_normalized_residual = std::abs(normalized_residual(i));

                 if (abs_normalized_residual <= huber_threshold)
                 {
                     // L2 region: w = 1
                     huber_weights(i) = 1.0f;
                 }
                 else
                 {
                     // L1 region: w = threshold / |normalized_residual|
                     huber_weights(i) = huber_threshold / abs_normalized_residual;
                 }
             }

             // Compute R_inv with Huber weighting
             Eigen::VectorXf R_inv(num_corr);
             for (int i = 0; i < num_corr; i++)
             {
                 float sigma = m_params.lidar_noise_std * m_params.lidar_noise_std;
                 // Apply Huber weight to measurement noise inverse
                 R_inv(i) = huber_weights(i) / (0.001f + sigma);
             }

             // Compute H^T * R_inv (6 x num_corr)
             Eigen::MatrixXf H_6 = H.block(0, 0, num_corr, 6);
             Eigen::MatrixXf H_T_R_inv(6, num_corr);
             for (int i = 0; i < num_corr; i++)
             {
                 H_T_R_inv.col(i) = H_6.row(i).transpose() * R_inv(i);
             }

             // Compute H^T * R^-1 * H (6x6)
             Eigen::Matrix<float, 6, 6> H_T_R_inv_H = H_T_R_inv * H_6;

             // Compute H^T * R^-1 * z
             Eigen::Matrix<float, 6, 1> H_T_R_inv_z = H_T_R_inv * residual;

             // Get prior covariance P (full 18x18)
             Eigen::Matrix<float, 18, 18> P_prior = m_current_state.m_covariance;

             // Build H^T*R^-1*H for full state (18x18)
             Eigen::Matrix<float, 18, 18> H_T_R_inv_H_full = Eigen::Matrix<float, 18, 18>::Zero();
             H_T_R_inv_H_full.block<6, 6>(0, 0) = H_T_R_inv_H;

             // Compute Kalman gain: K_1 = (H^T * R^-1 * H + P^-1)^-1
             Eigen::Matrix<float, 18, 18> information_matrix = H_T_R_inv_H_full + P_prior.inverse();
             Eigen::Matrix<float, 18, 18> K_1 = information_matrix.inverse();

             // Compute G matrix: G = K_1 * H^T*R^-1*H
             Eigen::Matrix<float, 18, 18> G = Eigen::Matrix<float, 18, 18>::Zero();
             G.block<18, 6>(0, 0) = K_1.block<18, 6>(0, 0) * H_T_R_inv_H;
             G_final = G; // Save for covariance update

             // Compute state correction
             Eigen::Matrix<float, 18, 1> dx = K_1.block<18, 6>(0, 0) * H_T_R_inv_z;

             // Apply state correction
             ApplyStateCorrection(dx);

             // Check inner convergence
             float rot_norm = dx.segment<3>(0).norm();
             float pos_norm = dx.segment<3>(3).norm();

             // Inner convergence: state change is small
             float convergence_threshold_f = static_cast<float>(m_params.convergence_threshold);
             if (rot_norm < convergence_threshold_f && pos_norm < convergence_threshold_f)
             {
                 inner_converged = true;
                 break;
             }
         }

         // After inner loop: Check if we need outer re-linearization
         // If inner loop converged quickly and state didn't change much, we're done
         if (inner_converged && outer_iter > 0)
         {
             converged = true;

             // Update covariance: P = (I - G) * P
             Eigen::Matrix<float, 18, 18> I18 = Eigen::Matrix<float, 18, 18>::Identity();
             Eigen::Matrix<float, 18, 18> P_prior = m_current_state.m_covariance;
             m_current_state.m_covariance = (I18 - G_final) * P_prior;

             break;
         }

         // If inner loop didn't converge or this is first outer iteration, continue to re-linearize
     }

    m_last_correspondences = correspondences;
}

// ============================================================================
// Correspondence Finding
// ============================================================================

std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, size_t>> 
Estimator::FindCorrespondences(const PointCloudPtr scan) {
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, size_t>> correspondences;

    // Check if VoxelMap is available and has points
    if (!m_voxel_map || m_voxel_map->GetPointCount() == 0) {
        spdlog::warn("[Estimator] VoxelMap is empty, no correspondences found");
        return correspondences;
    }
    
    if (!scan || scan->empty()) {
        spdlog::warn("[Estimator] Scan is empty, no correspondences found");
        return correspondences;
    }
    
    // Get current state
    Eigen::Matrix3f R_wb = m_current_state.m_rotation;
    Eigen::Vector3f t_wb = m_current_state.m_position;
    
    
    // Build transformation matrix ONCE: T_world_lidar = T_world_body * T_body_lidar
    Eigen::Matrix4f T_wb = Eigen::Matrix4f::Identity();
    T_wb.block<3,3>(0,0) = R_wb;
    T_wb.block<3,1>(0,3) = t_wb;
    
    Eigen::Matrix4f T_il = Eigen::Matrix4f::Identity();
    T_il.block<3,3>(0,0) = m_params.R_il;
    T_il.block<3,1>(0,3) = m_params.t_il;
    
    Eigen::Matrix4f T_wl = T_wb * T_il;  // Combined transformation
    
    // Process points and find correspondences using L1 surfels
    int valid_correspondences = 0;
    int total_attempts = 0;
    int no_surfel_count = 0;

    for (size_t i = 0; i < scan->size(); ++i) {
        // Early termination: stop when we have enough correspondences
        
        total_attempts++;
        
        // === 1. Transform point to world frame ===
        const auto& pt_scan = scan->at(i);
        Eigen::Vector4f pt_homo(pt_scan.x, pt_scan.y, pt_scan.z, 1.0f);
        Eigen::Vector4f pt_world_homo = T_wl * pt_homo;
        
        // Create query point in world frame
        Point3D query_point;
        query_point.x = pt_world_homo.x();
        query_point.y = pt_world_homo.y();
        query_point.z = pt_world_homo.z();
        
        // === 2. Get surfel from the L1 voxel containing this point ===
        Eigen::Vector3f surfel_normal;
        Eigen::Vector3f surfel_centroid;
        float planarity_score;
        
        bool has_surfel = m_voxel_map->GetSurfelAtPoint(query_point, surfel_normal, surfel_centroid, planarity_score);

        if (!has_surfel) {
            no_surfel_count++;
            continue;  // No surfel in this L1 voxel
        }

        // === 3. Calculate point-to-plane distance ===
        Eigen::Vector3f p_world(query_point.x, query_point.y, query_point.z);
        float dist_to_plane = std::abs(surfel_normal.dot(p_world - surfel_centroid));
        // // voxel size
        // if(dist_to_plane > 0.5f) {
        //     continue;  // Discard point if too far from surfel
        // }

        // === 4. Add valid correspondence ===
        // Plane equation: n^T * x + d = 0, where d = -n^T * centroid
        float plane_d = -surfel_normal.dot(surfel_centroid);

 
        
        // Store original lidar point
        Eigen::Vector3f p_lidar(pt_scan.x, pt_scan.y, pt_scan.z);
        
        // Store: (p_lidar, plane_normal_world, plane_d, scan_index)
        correspondences.emplace_back(p_lidar, surfel_normal, plane_d, i);
        valid_correspondences++;
    }
    
    // Update valid correspondence count
    m_num_valid_correspondences = valid_correspondences;

    return correspondences;
}

// ============================================================================
// Local Map Management
// ============================================================================

void Estimator::UpdateLocalMap(const PointCloudPtr scan) {
    auto start_total = std::chrono::high_resolution_clock::now();
    
    Eigen::Matrix3f R_wb = m_current_state.m_rotation;
    Eigen::Vector3f t_wb = m_current_state.m_position;
    
    // Every frame is a keyframe (always add to map)
    bool is_keyframe = true;
    
    // ===== Add new scan to map =====
    auto start_transform = std::chrono::high_resolution_clock::now();
    
    // Transform scan to world frame
    auto transformed_scan = std::make_shared<PointCloud>();
    int added_count = 0;
    for (const auto& pt : *scan) {
        // LiDAR point in sensor frame
        Eigen::Vector3f p_lidar(pt.x, pt.y, pt.z);
        
        // Transform: p_world = R_wb * (R_il * p_lidar + t_il) + t_wb
        Eigen::Vector3f p_imu = m_params.R_il * p_lidar + m_params.t_il;
        Eigen::Vector3f p_world = R_wb * p_imu + t_wb;
        
        // Add to transformed scan
        Point3D map_pt;
        map_pt.x = p_world.x();
        map_pt.y = p_world.y();
        map_pt.z = p_world.z();
        map_pt.intensity = pt.intensity;
        map_pt.offset_time = pt.offset_time;
        transformed_scan->push_back(map_pt);
        added_count++;
    }
    auto end_transform = std::chrono::high_resolution_clock::now();
    double transform_time = std::chrono::duration<double, std::milli>(end_transform - start_transform).count();

    // ===== Update VoxelMap: Add new points and remove distant voxels =====
    auto start_voxelmap_update = std::chrono::high_resolution_clock::now();
    
    // Get current sensor position in world frame
    Eigen::Vector3d sensor_position = t_wb.cast<double>();
    
    // Update voxel map: add new points and remove voxels outside map box
    if (!m_voxel_map) {
        m_voxel_map = std::make_shared<VoxelMap>(static_cast<float>(m_params.voxel_size));
        m_voxel_map->SetHierarchyFactor(m_params.voxel_hierarchy_factor);
        m_voxel_map->SetPlanarityThreshold(static_cast<float>(m_params.map_planarity_threshold));
        m_voxel_map->SetPointToSurfelThreshold(static_cast<float>(m_params.point_to_surfel_threshold));
        m_voxel_map->SetMinSurfelInliers(m_params.min_surfel_inliers);
        m_voxel_map->SetMinLinearityRatio(static_cast<float>(m_params.min_linearity_ratio));
        m_voxel_map->SetMapBoxMultiplier(static_cast<float>(m_params.map_box_multiplier));
    }

    m_voxel_map->UpdateVoxelMap(transformed_scan, sensor_position, m_params.max_map_distance, is_keyframe);
    auto end_voxelmap_update = std::chrono::high_resolution_clock::now();
    double voxelmap_update_time = std::chrono::duration<double, std::milli>(end_voxelmap_update - start_voxelmap_update).count();
    
    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
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
        
        // Rebuild VoxelMap after cleaning
        if (!m_map_cloud->empty()) {
            m_voxel_map = std::make_shared<VoxelMap>(static_cast<float>(m_params.voxel_size));
            m_voxel_map->SetHierarchyFactor(m_params.voxel_hierarchy_factor);
            m_voxel_map->SetPlanarityThreshold(static_cast<float>(m_params.map_planarity_threshold));
            m_voxel_map->SetPointToSurfelThreshold(static_cast<float>(m_params.point_to_surfel_threshold));
            m_voxel_map->SetMinSurfelInliers(m_params.min_surfel_inliers);
            m_voxel_map->SetMinLinearityRatio(static_cast<float>(m_params.min_linearity_ratio));
            m_voxel_map->SetMapBoxMultiplier(static_cast<float>(m_params.map_box_multiplier));
            m_voxel_map->AddPointCloud(m_map_cloud);
            spdlog::debug("[Estimator] VoxelMap rebuilt after cleaning");
        }
        
        spdlog::info("[Estimator] Map cleaned: {} points", m_map_cloud->size());
    }
}

// ============================================================================
// Jacobian Computation
// ============================================================================

void Estimator::ComputeLidarJacobians(
    const std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, size_t>>& correspondences,
    Eigen::MatrixXf& H,
    Eigen::VectorXf& residual) 
{
    // Compute Jacobian matrix H and residual vector for point-to-plane correspondences
    // State: [rotation(3), position(3), velocity(3), gyro_bias(3), acc_bias(3), gravity(3)]
    // LiDAR only observes rotation and position, other states have zero Jacobian
    
    int num_corr = correspondences.size();
    H.resize(num_corr, 18);
    residual.resize(num_corr);
    
    H.setZero();
    residual.setZero();
    
    // Get current state
    Eigen::Matrix3f R_wb = m_current_state.m_rotation;
    Eigen::Vector3f t_wb = m_current_state.m_position;
    
    // Process each correspondence
    for (int i = 0; i < num_corr; i++) {
        // Extract correspondence data: (p_lidar, plane_normal, plane_d)
        const Eigen::Vector3f& p_lidar = std::get<0>(correspondences[i]);
        const Eigen::Vector3f& norm_vec = std::get<1>(correspondences[i]);  // plane normal (world frame)
        const float plane_d = std::get<2>(correspondences[i]);
        
        // Transform point through chain: LiDAR -> IMU -> World
        // p_imu = R_il * p_lidar + t_il
        Eigen::Vector3f p_imu = m_params.R_il * p_lidar + m_params.t_il;
        
        // p_world = R_wb * p_imu + t_wb
        Eigen::Vector3f p_world = R_wb * p_imu + t_wb;
        
        // ===== Residual Computation =====
        // Point-to-plane distance: dis_to_plane = n^T * p_w + d
        // Measurement vector: meas_vec(i) = -dis_to_plane
        // Therefore: residual = -(n^T * p_world + d)
        residual(i) = -(norm_vec.dot(p_world) + plane_d);
        
        // ===== Jacobian Computation =====
        
        // Transform normal to body frame: C = R_wb^T * n
        Eigen::Vector3f C = R_wb.transpose() * norm_vec;
        
        // Rotation Jacobian: A = [p_imu]× * C
        // A = point_crossmat * state_rotation.transpose() * normal
        // Using POSITIVE sign for proper gradient direction
        Eigen::Matrix3f p_imu_skew = Hat(p_imu);
        Eigen::Vector3f A = p_imu_skew * C;
        
        // Position Jacobian: simply the normal vector
        // ∂r/∂t = ∂(n^T * (R * p_imu + t))/∂t = n^T
        
        // Fill Jacobian row (1×18)
        // State order: [rotation(3), position(3), velocity(3), gyro_bias(3), acc_bias(3), gravity(3)]
        H.block<1, 3>(i, 0) = A.transpose();           // ∂r/∂rotation
        H.block<1, 3>(i, 3) = norm_vec.transpose();    // ∂r/∂position
        // H.block<1, 12>(i, 6) = 0;                   // velocity, biases, gravity (already zero)
    }
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
    m_measurement_noise = Eigen::MatrixXf::Identity(num_correspondences, num_correspondences);
    m_measurement_noise *= m_params.lidar_noise_std * m_params.lidar_noise_std;
}

void Estimator::ApplyStateCorrection(const Eigen::VectorXf& dx) {
    // Apply state correction on manifold (IEKF update)
    // State: [rotation(3), position(3), velocity(3), gyro_bias(3), acc_bias(3), gravity(3)]
    
    if (dx.size() != 18) {
        spdlog::error("[Estimator] Invalid state correction size: {} (expected 18)", dx.size());
        return;
    }
    
    // 1. Rotation: R_new = R * Exp(δθ)  (right perturbation on SO(3))
    Eigen::Vector3f dtheta = dx.segment<3>(0);
    Eigen::Matrix3f dR = SO3::Exp(dtheta).Matrix();
    m_current_state.m_rotation = m_current_state.m_rotation * dR;
    
    // 2. Position: p_new = p + δp  (additive in R^3)
    m_current_state.m_position += dx.segment<3>(3);
    
    // 3. Velocity: v_new = v + δv  (additive in R^3)
    m_current_state.m_velocity += dx.segment<3>(6);
    
    // 4. Gyroscope bias: bg_new = bg + δbg  (additive in R^3)
    m_current_state.m_gyro_bias += dx.segment<3>(9);
    
    // 5. Accelerometer bias: ba_new = ba + δba  (additive in R^3)
    m_current_state.m_acc_bias += dx.segment<3>(12);
    
    // 6. Gravity: g_new = g + δg  (additive in R^3)
    m_current_state.m_gravity += dx.segment<3>(15);
    
    // Log correction magnitude for debugging
    spdlog::debug("[Estimator] State correction applied: rotation={:.6f}, position={:.6f}, velocity={:.6f}",
                  dtheta.norm(), dx.segment<3>(3).norm(), dx.segment<3>(6).norm());
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

std::shared_ptr<VoxelMap> Estimator::GetVoxelMap() const {
    std::lock_guard<std::mutex> lock(m_map_mutex);
    return m_voxel_map;
}

void Estimator::PrintProcessingTimeStatistics() const {
    if (m_processing_times.empty()) {
        spdlog::warn("No processing time data collected");
        return;
    }
    
    double sum = 0.0;
    
    for (double t : m_processing_times) {
        sum += t;
    }
    
    double avg_time_ms = sum / m_processing_times.size();
    double avg_fps = 1000.0 / avg_time_ms;  // Convert ms to FPS
    
    spdlog::info("");
    spdlog::info("Processing Time Statistics (Total {} frames)", m_processing_times.size());
    spdlog::info("   Average: {:.2f} ms  ({:.1f} FPS)", avg_time_ms, avg_fps);
    spdlog::info("");
}

// ============================================================================
// Undistortion & Interpolation (Placeholder)
// ============================================================================

PointCloudPtr Estimator::UndistortPointCloud(
    const PointCloudPtr cloud,
    double scan_start_time,
    double scan_end_time) 
{
    // Optimized motion compensation using pre-computed transforms
    // Key optimizations:
    // 1. Pre-compute N transforms instead of per-point interpolation
    // 2. Lookup table for O(1) access
    
    if (!m_params.enable_undistortion || m_state_history.empty()) {
        return cloud;
    }
    
    PointCloudPtr undistorted_cloud(new PointCloud);
    undistorted_cloud->reserve(cloud->size());
    
    // === 1. Pre-compute N transforms ===
    const int N = 100;  // Number of time segments
    
    // Store relative rotation and translation for each segment
    // These transform points from LiDAR at t_k to LiDAR at t_end
    std::vector<Eigen::Matrix3f> R_rel(N);
    std::vector<Eigen::Vector3f> t_rel(N);
    
    double scan_duration = scan_end_time - scan_start_time;
    double dt = scan_duration / N;
    
    // Extrinsics: LiDAR -> IMU
    const Eigen::Matrix3f& R_il = m_params.R_il;  // R_imu_lidar
    const Eigen::Vector3f& t_il = m_params.t_il;  // t_imu_lidar
    
    // Get end state (reference frame)
    State state_end = InterpolateState(scan_end_time);
    const Eigen::Matrix3f& R_end = state_end.m_rotation;  // R_world_imu at end
    const Eigen::Vector3f& t_end = state_end.m_position;  // t_world_imu at end
    
    // Pre-compute for each time segment
    for (int k = 0; k < N; k++) {
        double t_k = scan_start_time + k * dt;
        
        // Get interpolated state at t_k
        State state_k = InterpolateState(t_k);
        const Eigen::Matrix3f& R_i = state_k.m_rotation;  // R_world_imu at t_i
        const Eigen::Vector3f& t_i = state_k.m_position;  // t_world_imu at t_i
        
        // Compute relative transform from LiDAR at t_k to LiDAR at t_end
        // Following the original transformation chain:
        // 1. p_imu_i = R_il * p_lidar + t_il
        // 2. p_world = R_i * p_imu_i + t_i
        // 3. p_imu_end = R_end^T * (p_world - t_end)
        // 4. p_undistorted = R_il^T * (p_imu_end - t_il)
        //
        // Combining: p_undistorted = R_il^T * (R_end^T * (R_i * (R_il * p + t_il) + t_i - t_end) - t_il)
        //          = R_il^T * R_end^T * R_i * R_il * p 
        //            + R_il^T * R_end^T * R_i * t_il 
        //            + R_il^T * R_end^T * t_i 
        //            - R_il^T * R_end^T * t_end 
        //            - R_il^T * t_il
        //
        // R_rel = R_il^T * R_end^T * R_i * R_il
        // t_rel = R_il^T * R_end^T * (R_i * t_il + t_i - t_end) - R_il^T * t_il
        
        Eigen::Matrix3f R_end_T = R_end.transpose();
        
        R_rel[k] = R_il.transpose() * R_end_T * R_i * R_il;
        t_rel[k] = R_il.transpose() * R_end_T * (R_i * t_il + t_i - t_end) - R_il.transpose() * t_il;
    }
    
    // === 2. Apply pre-computed transforms to each point ===
    for (const auto& point : *cloud) {
        // Find nearest pre-computed transform index from offset_time
        int idx = static_cast<int>(point.offset_time / dt);
        idx = std::max(0, std::min(idx, N - 1));
        
        // Apply relative transform
        Eigen::Vector3f p_lidar(point.x, point.y, point.z);
        Eigen::Vector3f p_undistorted = R_rel[idx] * p_lidar + t_rel[idx];
        
        // Create undistorted point
        Point3D undistorted_point(
            p_undistorted.x(), 
            p_undistorted.y(), 
            p_undistorted.z(),
            point.intensity,
            0.0f  // All points are now aligned to scan end time
        );
        undistorted_cloud->push_back(undistorted_point);
    }
    
    return undistorted_cloud;
}

State Estimator::InterpolateState(double timestamp) const {
    // Linear interpolation between two nearest states in history
    
    if (m_state_history.empty()) {
        return m_current_state;
    }
    
    // Find two closest states: one before and one after timestamp
    const StateWithTimestamp* state_before = nullptr;
    const StateWithTimestamp* state_after = nullptr;
    
    for (size_t i = 0; i < m_state_history.size(); ++i) {
        if (m_state_history[i].timestamp <= timestamp) {
            state_before = &m_state_history[i];
        }
        if (m_state_history[i].timestamp >= timestamp && state_after == nullptr) {
            state_after = &m_state_history[i];
            break;
        }
    }
    
    // Case 1: timestamp is before all states -> return first state
    if (state_before == nullptr && state_after != nullptr) {
        return state_after->state;
    }
    
    // Case 2: timestamp is after all states -> return last state
    if (state_before != nullptr && state_after == nullptr) {
        return state_before->state;
    }
    
    // Case 3: timestamp is between two states -> linear interpolation
    if (state_before != nullptr && state_after != nullptr) {
        double t1 = state_before->timestamp;
        double t2 = state_after->timestamp;
        
        // If same timestamp, return either
        if (std::abs(t2 - t1) < 1e-9) {
            return state_after->state;
        }
        
        // Interpolation factor: alpha = 0 at t1, alpha = 1 at t2
        double alpha = (timestamp - t1) / (t2 - t1);
        float alpha_f = static_cast<float>(alpha);
        
        State interpolated_state;
        
        // Linear interpolation for position
        interpolated_state.m_position = (1.0f - alpha_f) * state_before->state.m_position 
                                       + alpha_f * state_after->state.m_position;
        
        // Linear interpolation for velocity
        interpolated_state.m_velocity = (1.0f - alpha_f) * state_before->state.m_velocity 
                                       + alpha_f * state_after->state.m_velocity;
        
        // Spherical linear interpolation (SLERP) for rotation
        Eigen::Quaternionf q1(state_before->state.m_rotation);
        Eigen::Quaternionf q2(state_after->state.m_rotation);
        Eigen::Quaternionf q_interp = q1.slerp(alpha_f, q2);
        interpolated_state.m_rotation = q_interp.toRotationMatrix();
        
        // Linear interpolation for biases
        interpolated_state.m_gyro_bias = (1.0f - alpha_f) * state_before->state.m_gyro_bias 
                                        + alpha_f * state_after->state.m_gyro_bias;
        interpolated_state.m_acc_bias = (1.0f - alpha_f) * state_before->state.m_acc_bias 
                                       + alpha_f * state_after->state.m_acc_bias;
        
        // Gravity should be constant
        interpolated_state.m_gravity = state_after->state.m_gravity;
        
        // Covariance: use the closer state's covariance
        if (alpha_f < 0.5f) {
            interpolated_state.m_covariance = state_before->state.m_covariance;
        } else {
            interpolated_state.m_covariance = state_after->state.m_covariance;
        }
        
        return interpolated_state;
    }
    
    // Fallback: return current state
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
