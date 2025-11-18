/**
 * @file      State.h
 * @brief     18-dimensional LiDAR-Inertial state representation.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <Eigen/Dense>
#include "LieUtils.h"
#include <memory>

namespace lio {

/**
 * @brief 18-dimensional LIO state vector
 * 
 * State composition:
 * - Rotation (SO3): 3 dimensions
 * - Position: 3 dimensions  
 * - Velocity: 3 dimensions
 * - Gyroscope bias: 3 dimensions
 * - Accelerometer bias: 3 dimensions
 * - Gravity: 3 dimensions
 * 
 * Total: 18 dimensions
 */
class State {
public:
    using Ptr = std::shared_ptr<State>;
    using ConstPtr = std::shared_ptr<const State>;
    
    // State dimensions (member variables)
    int m_state_dim = 18;
    int m_rot_dim = 3;
    int m_pos_dim = 3;
    int m_vel_dim = 3;
    int m_gyr_bias_dim = 3;
    int m_acc_bias_dim = 3;
    int m_grav_dim = 3;
    
    // State vector indices (member variables)
    int m_rot_idx = 0;
    int m_pos_idx = 3;
    int m_vel_idx = 6;
    int m_gyr_bias_idx = 9;
    int m_acc_bias_idx = 12;
    int m_grav_idx = 15;
    
    // ===== Constructor & Destructor =====
    
    State();
    State(const State& other);
    ~State() = default;
    
    // ===== State Variables (public for direct access) =====
    
    Eigen::Matrix3d m_rotation;        ///< SO(3) rotation matrix
    Eigen::Vector3d m_position;        ///< Position in world frame (m)
    Eigen::Vector3d m_velocity;        ///< Velocity in world frame (m/s)
    Eigen::Vector3d m_gyro_bias;       ///< Gyroscope bias (rad/s)
    Eigen::Vector3d m_acc_bias;        ///< Accelerometer bias (m/s²)
    Eigen::Vector3d m_gravity;         ///< Gravity vector (m/s²)
    
    Eigen::Matrix<double, 18, 18> m_covariance;  ///< State covariance matrix
    
    // ===== Operators =====
    
    /**
     * @brief Addition operator: state + delta
     * @param delta State increment (18x1 vector)
     * @return New state after applying delta
     */
    State operator+(const Eigen::Matrix<double, 18, 1>& delta) const;
    
    /**
     * @brief Subtraction operator: state - other_state
     * @param other Other state
     * @return State difference (18x1 vector)
     */
    Eigen::Matrix<double, 18, 1> operator-(const State& other) const;
    
    /**
     * @brief Assignment operator
     */
    State& operator=(const State& other);
    
    /**
     * @brief In-place addition operator
     */
    State& operator+=(const Eigen::Matrix<double, 18, 1>& delta);
    
    // ===== Utility Functions =====
    
    /**
     * @brief Reset state to initial values
     */
    void Reset();
    
    /**
     * @brief Get SE3 pose (rotation + translation)
     * @return SE3 pose
     */
    SE3 GetPose() const;
    
    /**
     * @brief Set pose from SE3
     * @param pose SE3 pose
     */
    void SetPose(const SE3& pose);
    
    /**
     * @brief Get state as vector representation
     * @return 18x1 state vector
     */
    Eigen::Matrix<double, 18, 1> ToVector() const;
    
    /**
     * @brief Set state from vector representation
     * @param state_vector 18x1 state vector
     */
    void FromVector(const Eigen::Matrix<double, 18, 1>& state_vector);
    
    /**
     * @brief Print state for debugging
     */
    void Print() const;
    
    /**
     * @brief Create shared pointer
     */
    static Ptr Create() { return std::make_shared<State>(); }
};

} // namespace lio
