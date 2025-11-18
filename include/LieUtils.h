/**
 * @file      LieUtils.h
 * @brief     Lie group utilities for SO(3) and SE(3) operations without Sophus dependency
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LIE_UTILS_H
#define LIE_UTILS_H

#include <Eigen/Dense>
#include <cmath>

namespace lio {

// Constants
constexpr float kEpsilon = 1e-6f;
constexpr float kPi = 3.14159f;

/**
 * @brief SO(3) Lie group utilities
 * 
 * Implements essential operations for 3D rotations:
 * - exp: axis-angle → rotation matrix
 * - log: rotation matrix → axis-angle  
 * - hat: vector → skew-symmetric matrix
 * - vee: skew-symmetric matrix → vector
 */
class SO3 {
public:
    SO3() = default;
    
    /// Create from rotation matrix (automatically normalizes to SO(3))
    explicit SO3(const Eigen::Matrix3f& R);
    
    /// Normalize current rotation matrix to ensure it's in SO(3)
    void Normalize();
    
    /// Create from axis-angle vector
    static SO3 Exp(const Eigen::Vector3f& omega);
    
    /// Convert to axis-angle vector
    Eigen::Vector3f Log() const;
    
    /// Get rotation matrix
    const Eigen::Matrix3f& Matrix() const { return m_matrix; }
    Eigen::Matrix3f& Matrix() { return m_matrix; }
    
    /// Composition
    SO3 operator*(const SO3& other) const {
        return SO3(m_matrix * other.m_matrix);
    }
    
    /// Apply rotation to vector
    Eigen::Vector3f operator*(const Eigen::Vector3f& v) const {
        return m_matrix * v;
    }
    
    /// Inverse
    SO3 Inverse() const {
        return SO3(m_matrix.transpose());
    }
    
    /// Identity
    static SO3 Identity() {
        return SO3(Eigen::Matrix3f::Identity());
    }
    
private:
    Eigen::Matrix3f m_matrix = Eigen::Matrix3f::Identity();
};

/**
 * @brief SE(3) Lie group utilities
 * 
 * Implements essential operations for 3D poses:
 * - exp: 6D twist → transformation matrix
 * - log: transformation matrix → 6D twist
 * - composition and inverse operations
 */
class SE3 {
public:
    SE3() = default;
    SE3(const SO3& rotation, const Eigen::Vector3f& translation) 
        : m_rotation(rotation), m_translation(translation) {}
    SE3(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
        : m_rotation(R), m_translation(t) {}
    explicit SE3(const Eigen::Matrix4f& matrix);
    
    /// Create from transformation matrix
    static SE3 FromMatrix(const Eigen::Matrix4f& T);
    
    /// Create from 6D twist vector [rho, phi] where rho=translation, phi=rotation
    static SE3 Exp(const Eigen::Matrix<float, 6, 1>& xi);
    
    /// Convert to 6D twist vector
    Eigen::Matrix<float, 6, 1> Log() const;
    
    /// Get transformation matrix
    Eigen::Matrix4f Matrix() const;
    
    /// Get rotation part
    const SO3& Rotation() const { return m_rotation; }
    SO3& Rotation() { return m_rotation; }
    
    /// Get rotation matrix
    Eigen::Matrix3f RotationMatrix() const { return m_rotation.Matrix(); }
    
    /// Get translation part
    const Eigen::Vector3f& Translation() const { return m_translation; }
    Eigen::Vector3f& Translation() { return m_translation; }
    
    /// Composition
    SE3 operator*(const SE3& other) const {
        return SE3(m_rotation * other.m_rotation,
                   m_translation + m_rotation * other.m_translation);
    }
    
    /// Apply transformation to point
    Eigen::Vector3f operator*(const Eigen::Vector3f& p) const {
        return m_rotation * p + m_translation;
    }
    
    /// Inverse
    SE3 Inverse() const {
        SO3 R_inv = m_rotation.Inverse();
        return SE3(R_inv, R_inv * (-m_translation));
    }
    
    /// Identity
    static SE3 Identity() {
        return SE3(SO3::Identity(), Eigen::Vector3f::Zero());
    }
    
private:
    SO3 m_rotation;
    Eigen::Vector3f m_translation = Eigen::Vector3f::Zero();
};

// ===== Utility Functions =====

/**
 * @brief Convert vector to skew-symmetric matrix
 * @param v 3D vector
 * @return 3x3 skew-symmetric matrix
 */
Eigen::Matrix3f Hat(const Eigen::Vector3f& v);

/**
 * @brief Convert skew-symmetric matrix to vector
 * @param S 3x3 skew-symmetric matrix
 * @return 3D vector
 */
Eigen::Vector3f Vee(const Eigen::Matrix3f& S);

} // namespace lio

#endif // LIE_UTILS_H