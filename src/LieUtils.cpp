/**
 * @file      LieUtils.cpp
 * @brief     Implementation of Lie group utilities for SO(3) and SE(3) operations
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "LieUtils.h"

namespace lio {

// ===== SO3 Implementation =====

SO3 SO3::Exp(const Eigen::Vector3f& omega) {
    const float theta = omega.norm();
    
    if (theta < kEpsilon) {
        // Small angle approximation: R ≈ I + [omega]_×
        return SO3(Eigen::Matrix3f::Identity() + Hat(omega));
    }
    
    const float theta_inv = 1.0f / theta;
    const Eigen::Vector3f k = omega * theta_inv;  // unit axis
    const Eigen::Matrix3f K = Hat(k);
    
    // Rodrigues' formula: R = I + sin(θ)[k]_× + (1-cos(θ))[k]_×²
    return SO3(Eigen::Matrix3f::Identity() + 
               std::sin(theta) * K + 
               (1.0f - std::cos(theta)) * K * K);
}

Eigen::Vector3f SO3::Log() const {
    const float trace = m_matrix.trace();
    const float cos_theta = (trace - 1.0f) * 0.5f;
    
    // Clamp to valid range
    const float cos_theta_clamped = std::max(-1.0f, std::min(1.0f, cos_theta));
    const float theta = std::acos(cos_theta_clamped);
    
    if (theta < kEpsilon) {
        // Small angle: ω ≈ vee(R - I)
        return Vee(m_matrix - Eigen::Matrix3f::Identity());
    }
    
    const float sin_theta = std::sin(theta);
    if (std::abs(sin_theta) < kEpsilon) {
        // θ ≈ π case: need special handling
        // Find the axis as the eigenvector with eigenvalue 1
        Eigen::Vector3f axis;
        
        // Check which diagonal element is largest
        int max_idx = 0;
        if (m_matrix(1, 1) > m_matrix(0, 0)) max_idx = 1;
        if (m_matrix(2, 2) > m_matrix(max_idx, max_idx)) max_idx = 2;
        
        axis[max_idx] = std::sqrt((m_matrix(max_idx, max_idx) + 1.0f) * 0.5f);
        
        for (int i = 0; i < 3; ++i) {
            if (i != max_idx) {
                axis[i] = m_matrix(max_idx, i) / (2.0f * axis[max_idx]);
            }
        }
        
        return axis * theta;
    }
    
    const float factor = theta / (2.0f * sin_theta);
    return factor * Vee(m_matrix - m_matrix.transpose());
}

SO3::SO3(const Eigen::Matrix3f& R) {
    // Use SVD to find the closest rotation matrix
    // R = U * Sigma * V^T, closest rotation is U * V^T
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    
    m_matrix = U * V.transpose();
    
    // Ensure det(R) = +1 (proper rotation, not reflection)
    if (m_matrix.determinant() < 0.0f) {
        // Flip the sign of the last column of U
        U.col(2) *= -1.0f;
        m_matrix = U * V.transpose();
    }
}

void SO3::Normalize() {
    *this = SO3(m_matrix);
}

// ===== SE3 Implementation =====

SE3::SE3(const Eigen::Matrix4f& matrix) {
    m_rotation = SO3(matrix.block<3, 3>(0, 0));
    m_translation = matrix.block<3, 1>(0, 3);
}

SE3 SE3::FromMatrix(const Eigen::Matrix4f& T) {
    return SE3(T);
}

SE3 SE3::Exp(const Eigen::Matrix<float, 6, 1>& xi) {
    const Eigen::Vector3f rho = xi.head<3>();     // translation part
    const Eigen::Vector3f phi = xi.tail<3>();     // rotation part
    
    const SO3 R = SO3::Exp(phi);
    
    const float theta = phi.norm();
    Eigen::Vector3f t;
    
    if (theta < kEpsilon) {
        // Small angle: V ≈ I
        t = rho;
    } else {
        // V = I + (1-cos(θ))/θ²[φ]_× + (θ-sin(θ))/θ³[φ]_×²
        const Eigen::Matrix3f phi_hat = Hat(phi);
        const float theta2 = theta * theta;
        const float sin_theta = std::sin(theta);
        const float cos_theta = std::cos(theta);
        
        const Eigen::Matrix3f V = Eigen::Matrix3f::Identity() +
            (1.0f - cos_theta) / theta2 * phi_hat +
            (theta - sin_theta) / (theta2 * theta) * phi_hat * phi_hat;
        
        t = V * rho;
    }
    
    return SE3(R, t);
}

Eigen::Matrix<float, 6, 1> SE3::Log() const {
    Eigen::Matrix<float, 6, 1> xi;
    
    const Eigen::Vector3f phi = m_rotation.Log();
    const float theta = phi.norm();
    
    if (theta < kEpsilon) {
        // Small angle: V^-1 ≈ I
        xi.head<3>() = m_translation;
    } else {
        // V^-1 = I - 1/2[φ]_× + (2sin(θ) - θ(1+cos(θ)))/(2θ²sin(θ))[φ]_×²
        const Eigen::Matrix3f phi_hat = Hat(phi);
        const float theta2 = theta * theta;
        const float sin_theta = std::sin(theta);
        const float cos_theta = std::cos(theta);
        
        const float coeff = (2.0f * sin_theta - theta * (1.0f + cos_theta)) / 
                           (2.0f * theta2 * sin_theta);
        
        const Eigen::Matrix3f V_inv = Eigen::Matrix3f::Identity() -
            0.5f * phi_hat +
            coeff * phi_hat * phi_hat;
        
        xi.head<3>() = V_inv * m_translation;
    }
    
    xi.tail<3>() = phi;
    return xi;
}

Eigen::Matrix4f SE3::Matrix() const {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3, 3>(0, 0) = m_rotation.Matrix();
    T.block<3, 1>(0, 3) = m_translation;
    return T;
}

// ===== Utility Functions =====

Eigen::Matrix3f Hat(const Eigen::Vector3f& v) {
    Eigen::Matrix3f S;
    S <<     0.0f, -v(2),  v(1),
          v(2),     0.0f, -v(0),
         -v(1),  v(0),     0.0f;
    return S;
}

Eigen::Vector3f Vee(const Eigen::Matrix3f& S) {
    return Eigen::Vector3f(S(2, 1), S(0, 2), S(1, 0));
}

} // namespace lio