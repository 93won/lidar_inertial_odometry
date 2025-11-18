/**
 * @file      State.cpp
 * @brief     Implementation of LiDAR-Inertial state representation.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "State.h"
#include <iostream>

namespace lio {

// Constants
const double kGravityNorm = 9.81;  // m/s²
const double kInitCov = 0.01;

State::State() {
    Reset();
}

State::State(const State& other)
    : m_rotation(other.m_rotation)
    , m_position(other.m_position)
    , m_velocity(other.m_velocity)
    , m_gyro_bias(other.m_gyro_bias)
    , m_acc_bias(other.m_acc_bias)
    , m_gravity(other.m_gravity)
    , m_covariance(other.m_covariance) {
}

void State::Reset() {
    m_rotation = Eigen::Matrix3d::Identity();
    m_position = Eigen::Vector3d::Zero();
    m_velocity = Eigen::Vector3d::Zero();
    m_gyro_bias = Eigen::Vector3d::Zero();
    m_acc_bias = Eigen::Vector3d::Zero();
    m_gravity = Eigen::Vector3d(0, 0, -kGravityNorm);
    
    // Initialize covariance
    m_covariance = Eigen::Matrix<double, 18, 18>::Identity() * kInitCov;
    
    // Set smaller initial covariance for position
    m_covariance.block<3, 3>(m_pos_idx, m_pos_idx) = 
        Eigen::Matrix3d::Identity() * 0.00001;
    
    // Set smaller initial covariance for biases and gravity
    m_covariance.block<9, 9>(m_gyr_bias_idx, m_gyr_bias_idx) = 
        Eigen::Matrix<double, 9, 9>::Identity() * 0.00001;
}

State State::operator+(const Eigen::Matrix<double, 18, 1>& delta) const {
    State result;
    
    // Rotation: R_new = R * Exp(delta_rot)
    Eigen::Vector3d delta_rot = delta.segment<3>(m_rot_idx);
    Eigen::Vector3f delta_rot_f = delta_rot.cast<float>();
    SO3 delta_R = SO3::Exp(delta_rot_f);
    result.m_rotation = (SO3(m_rotation.cast<float>()) * delta_R).Matrix().cast<double>();
    
    // Position: p_new = p + delta_pos
    result.m_position = m_position + delta.segment<3>(m_pos_idx);
    
    // Velocity: v_new = v + delta_vel
    result.m_velocity = m_velocity + delta.segment<3>(m_vel_idx);
    
    // Gyro bias: bg_new = bg + delta_bg
    result.m_gyro_bias = m_gyro_bias + delta.segment<3>(m_gyr_bias_idx);
    
    // Acc bias: ba_new = ba + delta_ba
    result.m_acc_bias = m_acc_bias + delta.segment<3>(m_acc_bias_idx);
    
    // Gravity: g_new = g + delta_g
    result.m_gravity = m_gravity + delta.segment<3>(m_grav_idx);
    
    // Covariance stays the same
    result.m_covariance = m_covariance;
    
    return result;
}

Eigen::Matrix<double, 18, 1> State::operator-(const State& other) const {
    Eigen::Matrix<double, 18, 1> delta;
    
    // Rotation difference: Log(R_other^T * R)
    Eigen::Matrix3d rot_diff = other.m_rotation.transpose() * m_rotation;
    SO3 rot_diff_so3 = SO3(rot_diff.cast<float>());
    Eigen::Vector3f log_rot = rot_diff_so3.Log();
    delta.segment<3>(m_rot_idx) = log_rot.cast<double>();
    
    // Position difference
    delta.segment<3>(m_pos_idx) = m_position - other.m_position;
    
    // Velocity difference
    delta.segment<3>(m_vel_idx) = m_velocity - other.m_velocity;
    
    // Gyro bias difference
    delta.segment<3>(m_gyr_bias_idx) = m_gyro_bias - other.m_gyro_bias;
    
    // Acc bias difference
    delta.segment<3>(m_acc_bias_idx) = m_acc_bias - other.m_acc_bias;
    
    // Gravity difference
    delta.segment<3>(m_grav_idx) = m_gravity - other.m_gravity;
    
    return delta;
}

State& State::operator=(const State& other) {
    if (this != &other) {
        m_rotation = other.m_rotation;
        m_position = other.m_position;
        m_velocity = other.m_velocity;
        m_gyro_bias = other.m_gyro_bias;
        m_acc_bias = other.m_acc_bias;
        m_gravity = other.m_gravity;
        m_covariance = other.m_covariance;
    }
    return *this;
}

State& State::operator+=(const Eigen::Matrix<double, 18, 1>& delta) {
    *this = *this + delta;
    return *this;
}

SE3 State::GetPose() const {
    return SE3(SO3(m_rotation.cast<float>()), m_position.cast<float>());
}

void State::SetPose(const SE3& pose) {
    m_rotation = pose.RotationMatrix().cast<double>();
    m_position = pose.Translation().cast<double>();
}

Eigen::Matrix<double, 18, 1> State::ToVector() const {
    Eigen::Matrix<double, 18, 1> vec;
    
    // Convert rotation to axis-angle
    SO3 rot_so3 = SO3(m_rotation.cast<float>());
    Eigen::Vector3f log_rot = rot_so3.Log();
    vec.segment<3>(m_rot_idx) = log_rot.cast<double>();
    vec.segment<3>(m_pos_idx) = m_position;
    vec.segment<3>(m_vel_idx) = m_velocity;
    vec.segment<3>(m_gyr_bias_idx) = m_gyro_bias;
    vec.segment<3>(m_acc_bias_idx) = m_acc_bias;
    vec.segment<3>(m_grav_idx) = m_gravity;
    
    return vec;
}

void State::FromVector(const Eigen::Matrix<double, 18, 1>& state_vector) {
    // Convert axis-angle to rotation matrix
    Eigen::Vector3f rot_vec = state_vector.segment<3>(m_rot_idx).cast<float>();
    SO3 rot_so3 = SO3::Exp(rot_vec);
    m_rotation = rot_so3.Matrix().cast<double>();
    m_position = state_vector.segment<3>(m_pos_idx);
    m_velocity = state_vector.segment<3>(m_vel_idx);
    m_gyro_bias = state_vector.segment<3>(m_gyr_bias_idx);
    m_acc_bias = state_vector.segment<3>(m_acc_bias_idx);
    m_gravity = state_vector.segment<3>(m_grav_idx);
}

void State::Print() const {
    std::cout << "========== LIO State ==========" << std::endl;
    std::cout << "Position: " << m_position.transpose() << std::endl;
    std::cout << "Velocity: " << m_velocity.transpose() << std::endl;
    std::cout << "Gyro Bias: " << m_gyro_bias.transpose() << std::endl;
    std::cout << "Acc Bias: " << m_acc_bias.transpose() << std::endl;
    std::cout << "Gravity: " << m_gravity.transpose() << std::endl;
    
    Eigen::Vector3d euler = m_rotation.eulerAngles(0, 1, 2) * 180.0 / M_PI;
    std::cout << "Rotation (RPY deg): " << euler.transpose() << std::endl;
    std::cout << "===============================" << std::endl;
}

} // namespace lio