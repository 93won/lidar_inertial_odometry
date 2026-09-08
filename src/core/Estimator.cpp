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
#include <unsupported/Eigen/MatrixFunctions>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lio {

// ============================================================================
// Constructor & Destructor
// ============================================================================

Estimator::Estimator()
{
	m_params.t_il = {0.04165, 0.02326, -0.0284};
    m_map_cloud = std::make_shared<PointCloud>();
    m_processed_cloud = std::make_shared<PointCloud>();

    PKOConfig pko_config;
    pko_config.use_adaptive = true;
    pko_config.min_scale_factor = 0.001;
    pko_config.max_scale_factor = 10.0;
    pko_config.num_alpha_segments = 25;
    pko_config.truncated_threshold = 10.0;
    pko_config.gmm_components = 2;
    pko_config.gmm_sample_size = 100;
    m_pko = std::make_shared<ProbabilisticKernelOptimizer>(pko_config);
}

Estimator::~Estimator() {
    std::lock_guard<std::mutex> lock_state(m_state_mutex);
    std::lock_guard<std::mutex> lock_map(m_map_mutex);
    std::lock_guard<std::mutex> lock_stats(m_stats_mutex);
}

void Estimator::UpdateProcessNoise() {
	m_process_noise.setZero();
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
    Eigen::Vector3d mean_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d mean_gyr = Eigen::Vector3d::Zero();
    
    for (const auto& imu : imu_buffer) {
        mean_acc += imu.acc;
        mean_gyr += imu.gyr;
    }
    mean_acc /= static_cast<double>(imu_buffer.size());
    mean_gyr /= static_cast<double>(imu_buffer.size());
    
    // 3. Compute variance to check if robot is stationary
    double acc_variance = 0.0;
    double gyr_variance = 0.0;
    
    for (const auto& imu : imu_buffer) {
        acc_variance += (imu.acc - mean_acc).squaredNorm();
        gyr_variance += (imu.gyr - mean_gyr).squaredNorm();
    }
    acc_variance /= static_cast<double>(imu_buffer.size());
    gyr_variance /= static_cast<double>(imu_buffer.size());
    
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
    const double acc_norm = mean_acc.norm();
    const double gravity_magnitude = m_params.gravity.norm();
    
    if (std::abs(acc_norm - gravity_magnitude) > 1.5f) {
        spdlog::error("[Estimator] Accelerometer norm = {:.3f} m/s² (expected ~{:.3f})", acc_norm, gravity_magnitude);
        spdlog::error("[Estimator] Sensor may be moving or miscalibrated. Initialization failed.");
        return false;
    }
    
    // 7. Initialize gravity vector (measured acceleration = -gravity in sensor frame)
    Eigen::Vector3d gravity_measured = -mean_acc.normalized() * gravity_magnitude;
    
    // 8. Set initial gravity (not yet aligned)
    m_current_state.m_gravity = gravity_measured;
    
    // 9. Initialize rotation to identity (will be aligned after)
    m_current_state.m_rotation = Eigen::Matrix3d::Identity();
    
    spdlog::info("[Estimator] Initial gravity (sensor frame): [{:.3f}, {:.3f}, {:.3f}]", 
                 gravity_measured.x(), gravity_measured.y(), gravity_measured.z());
    
    // 10. Gravity alignment: align world frame so gravity points to configured gravity direction
    // This rotates all states to make gravity vertical
    Eigen::Vector3d gravity_target = m_params.gravity;
    Eigen::Quaterniond q_align = Eigen::Quaterniond::FromTwoVectors(
        m_current_state.m_gravity.normalized(),
        gravity_target.normalized()
    );
    Eigen::Matrix3d R_align = q_align.toRotationMatrix();
    
   
    
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
    Eigen::Vector3d g_aligned = m_params.gravity;

    // Correct formula: bias = mean_acc + R^T * g
    Eigen::Vector3d acc_bias_estimate = mean_acc + m_current_state.m_rotation.transpose() * g_aligned;
    m_current_state.m_acc_bias = acc_bias_estimate;
    
    // 13. Initialize position and velocity to zero
    m_current_state.m_position.setZero();
    m_current_state.m_velocity.setZero();
    
    // 14. Initialize covariance with appropriate uncertainty
    m_current_state.m_covariance = State::Covariance::Identity();
    m_current_state.m_covariance.block<3,3>(0,0) *= 0.01f;   // rotation (small, well aligned)
    m_current_state.m_covariance.block<3,3>(3,3) *= 1.0f;    // position (unknown)
    m_current_state.m_covariance.block<3,3>(6,6) *= 0.1f;    // velocity (should be zero)
    m_current_state.m_covariance.block<3,3>(9,9) *= 0.001f;  // gyro bias (estimated from data)
    m_current_state.m_covariance.block<3,3>(12,12) *= 0.01; // acc bias (estimated from data)
    m_current_state.m_covariance.block<2,2>(15,15) *= 0.001; // gravity direction
    
    // 15. Set timestamp
    m_last_update_time = imu_buffer.back().timestamp;
    m_previous_imu = imu_buffer.back();
    m_has_previous_imu = true;
    m_state_history.emplace_back(m_current_state, m_last_update_time);
    
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
    Eigen::Vector3d acc_world = first_imu.acc;
    const double acc_norm = acc_world.norm();
    const double gravity_magnitude = m_params.gravity.norm();
    
    if (std::abs(acc_norm - gravity_magnitude) < 1.0f) {
        // Use accelerometer to initialize gravity direction
        m_current_state.m_gravity = -acc_world.normalized() * gravity_magnitude;
        
        // Gravity alignment: rotate world frame so gravity points to configured gravity
        Eigen::Vector3d gravity_target = m_params.gravity;
        Eigen::Quaterniond q_align = Eigen::Quaterniond::FromTwoVectors(
            m_current_state.m_gravity.normalized(),
            gravity_target.normalized()
        );
        Eigen::Matrix3d R_align = q_align.toRotationMatrix();
        
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
        m_current_state.m_rotation = Eigen::Matrix3d::Identity();
    }
    
    // Initialize biases to zero (will be estimated)
    m_current_state.m_gyro_bias.setZero();
    m_current_state.m_acc_bias.setZero();
    
    // Initialize position and velocity
    m_current_state.m_position.setZero();
    m_current_state.m_velocity.setZero();
    
    // Initialize covariance with large uncertainty
    m_current_state.m_covariance = State::Covariance::Identity();
    m_current_state.m_covariance.block<3,3>(0,0) *= 0.1f;    // rotation
    m_current_state.m_covariance.block<3,3>(3,3) *= 1.0f;    // position
    m_current_state.m_covariance.block<3,3>(6,6) *= 0.5f;    // velocity
    m_current_state.m_covariance.block<3,3>(9,9) *= 0.01f;   // gyro bias
    m_current_state.m_covariance.block<3,3>(12,12) *= 0.1f;  // acc bias
    m_current_state.m_covariance.block<2,2>(15,15) *= 0.01;
    
    m_last_update_time = first_imu.timestamp;
    m_previous_imu = first_imu;
    m_has_previous_imu = true;
    m_state_history.emplace_back(m_current_state, m_last_update_time);
    
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
    
    PropagateState(imu_data);
    m_state_history.emplace_back(m_current_state, imu_data.timestamp);
    const double oldest_needed = imu_data.timestamp - 2.0 * m_params.scan_duration;
    while (m_state_history.size() > 2 && m_state_history[1].timestamp < oldest_needed) {
        m_state_history.pop_front();
    }
}

void Estimator::PropagateState(const IMUData& imu) {
    const double dt = imu.timestamp - m_last_update_time;
    m_last_update_time = imu.timestamp;
    if (dt <= 0.0 || dt > 1.0) {
        m_previous_imu = imu;
        m_has_previous_imu = true;
        return;
    }

    const IMUData& previous = m_has_previous_imu ? m_previous_imu : imu;
    const Eigen::Vector3d omega = 0.5 * (previous.gyr + imu.gyr) - m_current_state.m_gyro_bias;
    const Eigen::Vector3d acceleration = 0.5 * (previous.acc + imu.acc) - m_current_state.m_acc_bias;
    const Eigen::Matrix3d rotation = m_current_state.m_rotation;
    const Eigen::Matrix3d rotation_mid = rotation * SO3::Exp(0.5 * omega * dt).Matrix();
    const Eigen::Vector3d acceleration_world = rotation_mid * acceleration + m_current_state.m_gravity;

    BuildDiscreteImuModel(omega, acceleration, rotation_mid, dt);
    m_current_state.m_covariance = m_state_transition * m_current_state.m_covariance
        * m_state_transition.transpose() + m_process_noise;
    StabilizeCovariance(m_current_state.m_covariance);

    m_current_state.m_position += m_current_state.m_velocity * dt
        + 0.5 * acceleration_world * dt * dt;
    m_current_state.m_velocity += acceleration_world * dt;
    m_current_state.m_rotation = rotation * SO3::Exp(omega * dt).Matrix();
    m_current_state.m_gravity = m_current_state.m_gravity.normalized() * m_params.gravity.norm();
    m_previous_imu = imu;
    m_has_previous_imu = true;
}

void Estimator::BuildDiscreteImuModel(const Eigen::Vector3d& omega,
                                      const Eigen::Vector3d& acceleration,
                                      const Eigen::Matrix3d& rotation_mid,
                                      double dt) {
    const double dt2 = dt * dt;
    const Eigen::Vector3d rotation_increment = omega * dt;
    const Eigen::Vector3d half_rotation_increment = 0.5 * rotation_increment;
    const auto right_jacobian = [](const Eigen::Vector3d& rotation_vector) {
        const double angle = rotation_vector.norm();
        const Eigen::Matrix3d skew = Hat(rotation_vector);
        Eigen::Matrix3d result = Eigen::Matrix3d::Identity();
        if (angle < 1e-10) {
            result -= 0.5 * skew - skew * skew / 6.0;
        } else {
            result -= (1.0 - std::cos(angle)) / (angle * angle) * skew;
            result += (angle - std::sin(angle)) / (angle * angle * angle) * skew * skew;
        }
        return result;
    };

    const Eigen::Matrix<double, 3, 2> gravity_basis =
        State::GravityTangentBasis(m_current_state.m_gravity);
    const Eigen::Matrix3d half_rotation = SO3::Exp(half_rotation_increment).Matrix();
    const Eigen::Matrix3d acceleration_rotation_jacobian =
        -rotation_mid * Hat(acceleration) * half_rotation.transpose();
    const Eigen::Matrix3d acceleration_gyro_bias_jacobian =
        0.5 * rotation_mid * Hat(acceleration)
        * right_jacobian(half_rotation_increment) * dt;

    m_state_transition.setIdentity();
    m_state_transition.block<3, 3>(0, 0) = SO3::Exp(-rotation_increment).Matrix();
    m_state_transition.block<3, 3>(0, 9) = -right_jacobian(rotation_increment) * dt;
    m_state_transition.block<3, 3>(3, 0) = 0.5 * acceleration_rotation_jacobian * dt2;
    m_state_transition.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
    m_state_transition.block<3, 3>(3, 9) = 0.5 * acceleration_gyro_bias_jacobian * dt2;
    m_state_transition.block<3, 3>(3, 12) = -0.5 * rotation_mid * dt2;
    m_state_transition.block<3, 2>(3, 15) = 0.5 * gravity_basis * dt2;
    m_state_transition.block<3, 3>(6, 0) = acceleration_rotation_jacobian * dt;
    m_state_transition.block<3, 3>(6, 9) = acceleration_gyro_bias_jacobian * dt;
    m_state_transition.block<3, 3>(6, 12) = -rotation_mid * dt;
    m_state_transition.block<3, 2>(6, 15) = gravity_basis * dt;

    m_noise_jacobian.setZero();
    m_noise_jacobian.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    m_noise_jacobian.block<3, 3>(6, 3) = -rotation_mid;
    m_noise_jacobian.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
    m_noise_jacobian.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();

    const double gyro_variance = m_params.gyr_noise_std * m_params.gyr_noise_std;
    const double acc_variance = m_params.acc_noise_std * m_params.acc_noise_std;
    const double gyro_bias_variance =
        m_params.gyr_bias_noise_std * m_params.gyr_bias_noise_std;
    const double acc_bias_variance =
        m_params.acc_bias_noise_std * m_params.acc_bias_noise_std;

    State::Covariance continuous_dynamics = State::Covariance::Zero();
    continuous_dynamics.block<3, 3>(0, 0) = -Hat(omega);
    continuous_dynamics.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity();
    continuous_dynamics.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
    continuous_dynamics.block<3, 3>(6, 0) = -rotation_mid * Hat(acceleration);
    continuous_dynamics.block<3, 3>(6, 12) = -rotation_mid;
    continuous_dynamics.block<3, 2>(6, 15) = gravity_basis;

    Eigen::Matrix<double, 12, 12> continuous_noise =
        Eigen::Matrix<double, 12, 12>::Zero();
    continuous_noise.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * gyro_variance;
    continuous_noise.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * acc_variance;
    continuous_noise.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * gyro_bias_variance;
    continuous_noise.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * acc_bias_variance;
    const State::Covariance covariance_rate =
        m_noise_jacobian * continuous_noise * m_noise_jacobian.transpose();

    static constexpr int AUGMENTED_STATE_DIM = 2 * State::kStateDim;
    Eigen::MatrixXd van_loan = Eigen::MatrixXd::Zero(AUGMENTED_STATE_DIM, AUGMENTED_STATE_DIM);
    van_loan.block<State::kStateDim, State::kStateDim>(0, 0) = continuous_dynamics;
    van_loan.block<State::kStateDim, State::kStateDim>(0, State::kStateDim) = covariance_rate;
    van_loan.block<State::kStateDim, State::kStateDim>(State::kStateDim, State::kStateDim) =
        -continuous_dynamics.transpose();
    const Eigen::MatrixXd exponential = (van_loan * dt).exp();
    const State::Covariance van_loan_transition =
        exponential.block<State::kStateDim, State::kStateDim>(0, 0);
    m_process_noise = exponential.block<State::kStateDim, State::kStateDim>(
        0, State::kStateDim) * van_loan_transition.transpose();
    m_process_noise = 0.5 * (m_process_noise + m_process_noise.transpose());
}

void Estimator::StabilizeCovariance(State::Covariance& covariance) {
    covariance = 0.5 * (covariance + covariance.transpose());
    Eigen::LLT<State::Covariance> cholesky(covariance);
    if (cholesky.info() == Eigen::Success) {
        return;
    }
    Eigen::SelfAdjointEigenSolver<State::Covariance> solver(covariance);
    if (solver.info() != Eigen::Success) {
        covariance += State::Covariance::Identity() * 1e-9;
        return;
    }
    Eigen::Matrix<double, State::kStateDim, 1> eigenvalues =
        solver.eigenvalues().cwiseMax(1e-12);
    covariance = solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
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
    
    auto preprocess_start = std::chrono::high_resolution_clock::now();

    const size_t raw_scan_size = lidar.cloud ? lidar.cloud->size() : 0;
    PointCloudPtr undistorted_cloud = lidar.cloud;
    if (m_params.enable_undistortion) {
        undistorted_cloud = UndistortPointCloud(
            lidar.cloud, lidar.timestamp - m_params.scan_duration, lidar.timestamp);
    }
    m_state_history.clear();  // Never mix pre-update and post-update states during deskew.

    PointCloudPtr downsampled_scan;
    size_t stride_scan_size = undistorted_cloud ? undistorted_cloud->size() : 0;
    if (m_params.stride > 1) {
        downsampled_scan = StrideDownsample(undistorted_cloud, m_params.stride);
        stride_scan_size = downsampled_scan->size();
        if (m_params.stride_then_voxel) {
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
        downsampled_scan = std::make_shared<PointCloud>();
        VoxelGrid scan_filter;
        scan_filter.SetInputCloud(undistorted_cloud);
        scan_filter.SetLeafSize(static_cast<float>(m_params.voxel_size));
        scan_filter.SetPlanarityFilter(true);
        scan_filter.SetHierarchyFactor(m_params.voxel_hierarchy_factor);
        scan_filter.Filter(*downsampled_scan);
    }

    PointCloudPtr range_filtered_scan = std::make_shared<PointCloud>();
    const size_t initial_size = downsampled_scan->size();
    const float min_range = static_cast<float>(m_params.min_range);
    const float max_range = static_cast<float>(m_params.max_map_distance);
    
    for (size_t i = 0; i < initial_size; ++i) {
        const auto& point = downsampled_scan->at(i);
        const float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        if (range >= min_range && range <= max_range) {
            range_filtered_scan->push_back(point);
        }
    }
    
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    
    LidarData downsampled_lidar(lidar.timestamp, range_filtered_scan);
    m_processed_cloud = range_filtered_scan;
    if (std::getenv("LIO_DIAGNOSTICS") != nullptr) {
        spdlog::info(
            "[PreprocessDiag] frame={} raw={} undistorted={} stride={} voxel={} range={}",
            m_frame_count, raw_scan_size, undistorted_cloud->size(), stride_scan_size,
            downsampled_scan->size(), range_filtered_scan->size());
    }
    
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
    const bool registered = UpdateWithLidar(downsampled_lidar);
    auto iekf_end = std::chrono::high_resolution_clock::now();
    double iekf_time = std::chrono::duration<double, std::milli>(iekf_end - iekf_start).count();
    
    // === 3. Map Update ===
    auto map_start = std::chrono::high_resolution_clock::now();
    if (ShouldUpdateMap(registered)
        || std::getenv("LIO_DIAGNOSTIC_UPDATE_MAP_ON_FAILURE") != nullptr) {
        UpdateLocalMap(range_filtered_scan);
    }
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
        m_statistics.successful_registrations += registered ? 1 : 0;
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
    m_state_history.emplace_back(m_current_state, lidar.timestamp);
    m_frame_count++;
    
    // Log detailed timing every 100 frames
    if (m_frame_count % 100 == 0) {
        double avg_preprocess = m_sum_preprocess_time / 100.0;
        double avg_iekf = m_sum_iekf_time / 100.0;
        double avg_map = m_sum_map_time / 100.0;
        double avg_total = avg_preprocess + avg_iekf + avg_map;
        
        spdlog::info("[Estimator] Frame {}: Total={:.2f}ms (Preprocess={:.2f}ms, IEKF={:.2f}ms, Map={:.2f}ms), Corr={}",
                     m_frame_count, avg_total, avg_preprocess, avg_iekf, avg_map,
                     m_num_valid_correspondences);
        
        // Reset accumulators
        m_sum_preprocess_time = 0.0;
        m_sum_iekf_time = 0.0;
        m_sum_map_time = 0.0;
    }
}

bool Estimator::ShouldUpdateMap(bool registered) {
    if (registered) {
        m_map_recovery_frames_remaining = 0;
        return true;
    }
    if (m_num_valid_correspondences > 0) {
        m_map_recovery_frames_remaining = std::max(0, m_params.map_recovery_frames);
    }
    if (m_map_recovery_frames_remaining == 0) {
        return false;
    }
    --m_map_recovery_frames_remaining;
    return true;
}

bool Estimator::UpdateWithLidar(const LidarData& lidar) {
    const char* diagnostic_minimum =
        std::getenv("LIO_DIAGNOSTIC_MIN_CORRESPONDENCES");
    const int kMinimumCorrespondences = diagnostic_minimum == nullptr
        ? 6
        : std::max(1, std::atoi(diagnostic_minimum));
    constexpr int kMaximumLineSearchSteps = 8;
    m_num_valid_correspondences = 0;
    const State predicted_state = m_current_state;
    const State::Covariance predicted_covariance = m_current_state.m_covariance;
    const bool diagnostics_enabled = std::getenv("LIO_DIAGNOSTICS") != nullptr;
    VoxelMap::MatchDiagnostics diagnostics;
    auto log_diagnostics = [&](const char* outcome) {
        if (!diagnostics_enabled) {
            return;
        }
        const SpatialIndex::Statistics index = m_voxel_map->GetPointIndexStatistics();
        spdlog::info(
            "[CorrespondenceDiag] frame={} t={:.3f} outcome={} queries={} accepted={} "
            "map={} neighbors={} centroid={} eigen={} degenerate={} planarity={} "
            "neighbor_plane={} query_plane={} nodes={} valid={} deleted={} depth={} rebuilds={} "
            "v={:.6g} bg={:.6g} ba={:.6g} pose_cov={:.6g} bias_cov={:.6g}",
            m_frame_count, lidar.timestamp, outcome, diagnostics.queries, diagnostics.accepted,
            diagnostics.insufficient_map, diagnostics.insufficient_neighbors,
            diagnostics.centroid_gate, diagnostics.eigen_failure,
            diagnostics.degenerate_neighbors, diagnostics.planarity_gate,
            diagnostics.neighbor_plane_gate, diagnostics.query_plane_gate,
            index.node_count, index.valid_count, index.deleted_count, index.max_depth,
            index.rebuild_count, predicted_state.m_velocity.norm(),
            predicted_state.m_gyro_bias.norm(),
            predicted_state.m_acc_bias.norm(),
            predicted_covariance.block<6, 6>(State::kRotationIndex, State::kRotationIndex).trace(),
            predicted_covariance.block<6, 6>(State::kGyroBiasIndex, State::kGyroBiasIndex).trace());
    };
    Eigen::LDLT<State::Covariance> prior_solver(predicted_covariance);
    if (prior_solver.info() != Eigen::Success || !prior_solver.isPositive()) {
        log_diagnostics("prior_covariance");
        return false;
    }
    const State::Covariance prior_information =
        prior_solver.solve(State::Covariance::Identity());

    auto prior_jacobian = [&](const State& state) {
        State::Covariance jacobian = State::Covariance::Identity();
        const State::Vector error = state - predicted_state;
        const Eigen::Vector3d rotation_error = error.segment<3>(State::kRotationIndex);
        const double angle = rotation_error.norm();
        const Eigen::Matrix3d skew = Hat(rotation_error);
        Eigen::Matrix3d inverse_right_jacobian =
            Eigen::Matrix3d::Identity() + 0.5 * skew;
        if (angle < 1e-10) {
            inverse_right_jacobian += skew * skew / 12.0;
        } else {
            const double coefficient = 1.0 / (angle * angle)
                - (1.0 + std::cos(angle)) / (2.0 * angle * std::sin(angle));
            inverse_right_jacobian += coefficient * skew * skew;
        }
        jacobian.block<3, 3>(State::kRotationIndex, State::kRotationIndex) =
            inverse_right_jacobian;
        jacobian.block<2, 2>(State::kGravityIndex, State::kGravityIndex) =
            State::GravityTangentBasis(predicted_state.m_gravity).transpose()
            * State::GravityTangentBasis(state.m_gravity);
        return jacobian;
    };

    auto inverse_noise = [&](const Eigen::VectorXd& residual) {
        std::vector<double> absolute_residuals(residual.size());
        for (int index = 0; index < residual.size(); ++index) {
            absolute_residuals[index] = std::abs(residual(index));
        }
        const size_t middle = absolute_residuals.size() / 2;
        std::nth_element(absolute_residuals.begin(),
            absolute_residuals.begin() + middle, absolute_residuals.end());
        const double scale = std::max(
            1.4826 * absolute_residuals[middle], m_params.lidar_noise_std);
        const double threshold = 1.345 * scale;
        Eigen::VectorXd result(residual.size());
        const double variance = m_params.lidar_noise_std * m_params.lidar_noise_std;
        for (int index = 0; index < residual.size(); ++index) {
            const double magnitude = std::abs(residual(index));
            const double weight = magnitude <= threshold ? 1.0 : threshold / magnitude;
            result(index) = weight / std::max(variance, 1e-12);
        }
        return result;
    };

    std::vector<Correspondence> correspondences;
    for (int iteration = 0; iteration < m_params.max_iterations; ++iteration) {
        correspondences = FindCorrespondences(
            lidar.cloud, 0, diagnostics_enabled && iteration == 0 ? &diagnostics : nullptr);
        if (correspondences.size() < kMinimumCorrespondences) {
            m_num_valid_correspondences = correspondences.size();
            m_current_state = predicted_state;
            log_diagnostics("initial_correspondences");
            return false;
        }

        Eigen::MatrixXd jacobian;
        Eigen::VectorXd residual;
        ComputeLidarJacobians(correspondences, jacobian, residual);
        const Eigen::VectorXd weights = inverse_noise(residual);
        const Eigen::MatrixXd weighted_jacobian =
            weights.asDiagonal() * jacobian;
        const State::Vector prior_error = m_current_state - predicted_state;
        const State::Covariance retraction_jacobian = prior_jacobian(m_current_state);
        const State::Covariance information =
            retraction_jacobian.transpose() * prior_information * retraction_jacobian
            + jacobian.transpose() * weighted_jacobian;
        const State::Vector right_hand_side =
            jacobian.transpose() * (weights.asDiagonal() * residual)
            - retraction_jacobian.transpose() * prior_information * prior_error;

        Eigen::LDLT<State::Covariance> solver(information);
        if (solver.info() != Eigen::Success || !solver.isPositive()) {
            m_num_valid_correspondences = correspondences.size();
            m_current_state = predicted_state;
            log_diagnostics("information_solver");
            return false;
        }
        const State::Vector correction = solver.solve(right_hand_side);
        if (!correction.allFinite()) {
            m_num_valid_correspondences = correspondences.size();
            m_current_state = predicted_state;
            log_diagnostics("nonfinite_correction");
            return false;
        }

        const State linearization_state = m_current_state;
        const double current_objective =
            prior_error.dot(prior_information * prior_error)
            + residual.dot(weights.cwiseProduct(residual));
        State::Vector accepted_correction = State::Vector::Zero();
        bool accepted = false;
        double step_size = 1.0;
        for (int line_search_step = 0;
             line_search_step < kMaximumLineSearchSteps;
             ++line_search_step) {
            accepted_correction = step_size * correction;
            m_current_state = linearization_state + accepted_correction;

            Eigen::MatrixXd candidate_jacobian;
            Eigen::VectorXd candidate_residual;
            ComputeLidarJacobians(
                correspondences, candidate_jacobian, candidate_residual);
            const Eigen::VectorXd candidate_weights = inverse_noise(candidate_residual);
            const State::Vector candidate_prior_error =
                m_current_state - predicted_state;
            const double candidate_objective =
                candidate_prior_error.dot(prior_information * candidate_prior_error)
                + candidate_residual.dot(
                    candidate_weights.cwiseProduct(candidate_residual));
            const auto candidate_correspondences =
                FindCorrespondences(lidar.cloud, kMinimumCorrespondences);
            if (candidate_correspondences.size() >= kMinimumCorrespondences
                && std::isfinite(candidate_objective)
                && candidate_objective
                    <= current_objective + 1e-9 * std::max(1.0, current_objective)) {
                accepted = true;
                break;
            }
            step_size *= 0.5;
        }
        if (!accepted) {
            m_num_valid_correspondences = correspondences.size();
            m_current_state = predicted_state;
            log_diagnostics("line_search");
            return false;
        }

        if (accepted_correction.segment<3>(State::kRotationIndex).norm()
                < m_params.convergence_threshold
            && accepted_correction.segment<3>(State::kPositionIndex).norm()
                < m_params.convergence_threshold) {
            break;
        }
    }

    correspondences = FindCorrespondences(lidar.cloud);
    if (correspondences.size() < kMinimumCorrespondences) {
        m_num_valid_correspondences = correspondences.size();
        m_current_state = predicted_state;
        log_diagnostics("final_correspondences");
        return false;
    }
    Eigen::MatrixXd final_jacobian;
    Eigen::VectorXd final_residual;
    ComputeLidarJacobians(correspondences, final_jacobian, final_residual);
    const Eigen::VectorXd final_weights = inverse_noise(final_residual);
    const State::Covariance final_prior_jacobian = prior_jacobian(m_current_state);
    const State::Covariance final_information =
        final_prior_jacobian.transpose() * prior_information * final_prior_jacobian
        + final_jacobian.transpose() * final_weights.asDiagonal() * final_jacobian;
    Eigen::LDLT<State::Covariance> final_solver(final_information);
    if (final_solver.info() != Eigen::Success || !final_solver.isPositive()) {
        m_num_valid_correspondences = correspondences.size();
        m_current_state = predicted_state;
        log_diagnostics("final_solver");
        return false;
    }

    const State::Covariance posterior_covariance =
        final_solver.solve(State::Covariance::Identity());
    if (!posterior_covariance.allFinite()) {
        m_num_valid_correspondences = correspondences.size();
        m_current_state = predicted_state;
        log_diagnostics("posterior_covariance");
        return false;
    }
    m_current_state.m_covariance = posterior_covariance;
    StabilizeCovariance(m_current_state.m_covariance);
    m_last_correspondences = correspondences;
    log_diagnostics("success");
    return true;
}
// Correspondence Finding
// ============================================================================

std::vector<Estimator::Correspondence>
Estimator::FindCorrespondences(
    const PointCloudPtr scan, std::size_t maximum_correspondences,
    VoxelMap::MatchDiagnostics* diagnostics) {
    std::vector<Correspondence> correspondences;
    m_num_valid_correspondences = 0;
    if (!m_voxel_map || m_voxel_map->GetPointCount() == 0 || !scan || scan->empty()
        || m_params.max_correspondences <= 0) {
        return correspondences;
    }
    if (maximum_correspondences == 0) {
        maximum_correspondences = static_cast<std::size_t>(m_params.max_correspondences);
    }
    correspondences.reserve(std::min(maximum_correspondences, scan->size()));

    const Eigen::Matrix3d rotation_world_lidar =
        m_current_state.m_rotation * m_params.R_il;
    const Eigen::Vector3d translation_world_lidar =
        m_current_state.m_rotation * m_params.t_il + m_current_state.m_position;
    const float centroid_distance_gate =
        static_cast<float>(m_params.max_correspondence_distance);
    const float plane_distance_gate =
        static_cast<float>(m_params.kdtree_max_plane_residual);

    for (size_t index = 0;
         index < scan->size() && correspondences.size() < maximum_correspondences;
         ++index) {
            const Point3D& point = scan->at(index);
            const Eigen::Vector3d point_lidar(point.x, point.y, point.z);
            const Eigen::Vector3d point_world =
                rotation_world_lidar * point_lidar + translation_world_lidar;
            Point3D query;
            query.x = static_cast<float>(point_world.x());
            query.y = static_cast<float>(point_world.y());
            query.z = static_cast<float>(point_world.z());

            Eigen::Vector3f normal_float;
            Eigen::Vector3f centroid_float;
            float planarity = 0.0f;
            if (!m_voxel_map->GetClosestSurfel(
                    query, centroid_distance_gate, plane_distance_gate,
                    normal_float, centroid_float, planarity, diagnostics)) {
                continue;
            }
            const Eigen::Vector3d normal = normal_float.cast<double>();
            const Eigen::Vector3d centroid = centroid_float.cast<double>();
            const double distance = std::abs(normal.dot(point_world - centroid));
            if (distance > plane_distance_gate) {
                continue;
            }
            correspondences.emplace_back(
                point_lidar, normal, -normal.dot(centroid), index);
    }

    m_num_valid_correspondences = correspondences.size();
    return correspondences;
}
// Local Map Management
// ============================================================================

void Estimator::UpdateLocalMap(const PointCloudPtr scan) {
    auto start_total = std::chrono::high_resolution_clock::now();
    
    Eigen::Matrix3d R_wb = m_current_state.m_rotation;
    Eigen::Vector3d t_wb = m_current_state.m_position;
    
    // Every frame is a keyframe (always add to map)
    bool is_keyframe = true;
    
    // ===== Add new scan to map =====
    auto start_transform = std::chrono::high_resolution_clock::now();
    
    // Transform scan to world frame
    auto transformed_scan = std::make_shared<PointCloud>();
    int added_count = 0;
    for (const auto& pt : *scan) {
        // LiDAR point in sensor frame
        Eigen::Vector3d p_lidar(pt.x, pt.y, pt.z);
        
        // Transform: p_world = R_wb * (R_il * p_lidar + t_il) + t_wb
        Eigen::Vector3d p_imu = m_params.R_il * p_lidar + m_params.t_il;
        Eigen::Vector3d p_world = R_wb * p_imu + t_wb;
        
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
        m_voxel_map = std::make_shared<VoxelMap>(static_cast<float>(m_params.map_voxel_size));
        m_voxel_map->SetHierarchyFactor(m_params.voxel_hierarchy_factor);
        m_voxel_map->SetPlanarityThreshold(static_cast<float>(m_params.map_planarity_threshold));
        m_voxel_map->SetPointToSurfelThreshold(static_cast<float>(m_params.point_to_surfel_threshold));
        m_voxel_map->SetKNearestNeighbors(m_params.kdtree_knn);
        m_voxel_map->SetKnnPlanarityThreshold(
            static_cast<float>(m_params.kdtree_planarity_threshold));
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
            m_voxel_map = std::make_shared<VoxelMap>(static_cast<float>(m_params.map_voxel_size));
            m_voxel_map->SetHierarchyFactor(m_params.voxel_hierarchy_factor);
            m_voxel_map->SetPlanarityThreshold(static_cast<float>(m_params.map_planarity_threshold));
            m_voxel_map->SetPointToSurfelThreshold(static_cast<float>(m_params.point_to_surfel_threshold));
            m_voxel_map->SetKNearestNeighbors(m_params.kdtree_knn);
            m_voxel_map->SetKnnPlanarityThreshold(
                static_cast<float>(m_params.kdtree_planarity_threshold));
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
    const std::vector<Correspondence>& correspondences,
    Eigen::MatrixXd& H,
    Eigen::VectorXd& residual)
{
    // Compute Jacobian matrix H and residual vector for point-to-plane correspondences
    // State: [rotation(3), position(3), velocity(3), gyro_bias(3), acc_bias(3), gravity tangent(2)]
    // LiDAR only observes rotation and position, other states have zero Jacobian
    
    int num_corr = correspondences.size();
    H.resize(num_corr, State::kStateDim);
    residual.resize(num_corr);
    
    H.setZero();
    residual.setZero();
    
    // Get current state
    Eigen::Matrix3d R_wb = m_current_state.m_rotation;
    Eigen::Vector3d t_wb = m_current_state.m_position;
    
    // Process each correspondence
    for (int i = 0; i < num_corr; i++) {
        // Extract correspondence data: (p_lidar, plane_normal, plane_d)
        const Eigen::Vector3d& p_lidar = std::get<0>(correspondences[i]);
        const Eigen::Vector3d& norm_vec = std::get<1>(correspondences[i]);  // plane normal (world frame)
        const double plane_d = std::get<2>(correspondences[i]);
        
        // Transform point through chain: LiDAR -> IMU -> World
        // p_imu = R_il * p_lidar + t_il
        Eigen::Vector3d p_imu = m_params.R_il * p_lidar + m_params.t_il;
        
        // p_world = R_wb * p_imu + t_wb
        Eigen::Vector3d p_world = R_wb * p_imu + t_wb;
        
        // ===== Residual Computation =====
        // Point-to-plane distance: dis_to_plane = n^T * p_w + d
        // Measurement vector: meas_vec(i) = -dis_to_plane
        // Therefore: residual = -(n^T * p_world + d)
        residual(i) = -(norm_vec.dot(p_world) + plane_d);
        
        // ===== Jacobian Computation =====
        
        // Transform normal to body frame: C = R_wb^T * n
        Eigen::Vector3d C = R_wb.transpose() * norm_vec;
        
        // Rotation Jacobian: A = [p_imu]× * C
        // A = point_crossmat * state_rotation.transpose() * normal
        // Using POSITIVE sign for proper gradient direction
        Eigen::Matrix3d p_imu_skew = Hat(p_imu);
        Eigen::Vector3d A = p_imu_skew * C;
        
        // Position Jacobian: simply the normal vector
        // ∂r/∂t = ∂(n^T * (R * p_imu + t))/∂t = n^T
        
        // LiDAR directly observes only pose.
        H.block<1, 3>(i, 0) = A.transpose();           // ∂r/∂rotation
        H.block<1, 3>(i, 3) = norm_vec.transpose();    // ∂r/∂position
        // H.block<1, 12>(i, 6) = 0;                   // velocity, biases, gravity (already zero)
    }
}

void Estimator::ApplyStateCorrection(const State::Vector& correction) {
    m_current_state += correction;
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
    std::vector<Eigen::Matrix3d> R_rel(N);
    std::vector<Eigen::Vector3d> t_rel(N);
    double maximum_relative_rotation = 0.0;
    double maximum_relative_translation = 0.0;
    
    double scan_duration = scan_end_time - scan_start_time;
    if (scan_duration <= 0.0) {
        return cloud;
    }
    double dt = scan_duration / N;
    
    // Extrinsics: LiDAR -> IMU
    const Eigen::Matrix3d& R_il = m_params.R_il;  // R_imu_lidar
    const Eigen::Vector3d& t_il = m_params.t_il;  // t_imu_lidar
    
    // Get end state (reference frame)
    State state_end = InterpolateState(scan_end_time);
    const Eigen::Matrix3d& R_end = state_end.m_rotation;  // R_world_imu at end
    const Eigen::Vector3d& t_end = state_end.m_position;  // t_world_imu at end
    
    // Pre-compute for each time segment
    for (int k = 0; k < N; k++) {
        double t_k = scan_start_time + k * dt;
        
        // Get interpolated state at t_k
        State state_k = InterpolateState(t_k);
        const Eigen::Matrix3d& R_i = state_k.m_rotation;  // R_world_imu at t_i
        const Eigen::Vector3d& t_i = state_k.m_position;  // t_world_imu at t_i
        
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
        
        Eigen::Matrix3d R_end_T = R_end.transpose();
        
        R_rel[k] = R_il.transpose() * R_end_T * R_i * R_il;
        t_rel[k] = R_il.transpose() * R_end_T * (R_i * t_il + t_i - t_end) - R_il.transpose() * t_il;
        maximum_relative_rotation = std::max(
            maximum_relative_rotation, Eigen::AngleAxisd(R_rel[k]).angle());
        maximum_relative_translation = std::max(
            maximum_relative_translation, t_rel[k].norm());
    }
    if (std::getenv("LIO_DIAGNOSTICS") != nullptr) {
        spdlog::info(
            "[DeskewDiag] frame={} history={} duration={:.6g} max_rotation={:.6g} "
            "max_translation={:.6g}",
            m_frame_count, m_state_history.size(), scan_duration,
            maximum_relative_rotation, maximum_relative_translation);
    }
    
    // === 2. Apply pre-computed transforms to each point ===
    for (const auto& point : *cloud) {
        // Find nearest pre-computed transform index from offset_time
        int idx = static_cast<int>(point.offset_time / dt);
        idx = std::max(0, std::min(idx, N - 1));
        
        // Apply relative transform
        Eigen::Vector3d p_lidar(point.x, point.y, point.z);
        Eigen::Vector3d p_undistorted = R_rel[idx] * p_lidar + t_rel[idx];
        
        // Create undistorted point
        Point3D undistorted_point(
            static_cast<float>(p_undistorted.x()),
            static_cast<float>(p_undistorted.y()),
            static_cast<float>(p_undistorted.z()),
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
        const double alpha = (timestamp - t1) / (t2 - t1);
        
        State interpolated_state;
        
        // Linear interpolation for position
        interpolated_state.m_position = (1.0 - alpha) * state_before->state.m_position
                                       + alpha * state_after->state.m_position;
        
        // Linear interpolation for velocity
        interpolated_state.m_velocity = (1.0 - alpha) * state_before->state.m_velocity
                                       + alpha * state_after->state.m_velocity;
        
        // Spherical linear interpolation (SLERP) for rotation
        Eigen::Quaterniond q1(state_before->state.m_rotation);
        Eigen::Quaterniond q2(state_after->state.m_rotation);
        Eigen::Quaterniond q_interp = q1.slerp(alpha, q2);
        interpolated_state.m_rotation = q_interp.toRotationMatrix();
        
        // Linear interpolation for biases
        interpolated_state.m_gyro_bias = (1.0 - alpha) * state_before->state.m_gyro_bias
                                        + alpha * state_after->state.m_gyro_bias;
        interpolated_state.m_acc_bias = (1.0 - alpha) * state_before->state.m_acc_bias
                                       + alpha * state_after->state.m_acc_bias;
        
        // Gravity should be constant
        interpolated_state.m_gravity = state_after->state.m_gravity;
        
        // Covariance: use the closer state's covariance
        if (alpha < 0.5) {
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
