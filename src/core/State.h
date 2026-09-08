#pragma once

#include "LieUtils.h"

#include <Eigen/Dense>
#include <memory>

namespace lio {

class State {
public:
	static constexpr int kStateDim = 17;
	static constexpr int kRotationIndex = 0;
	static constexpr int kPositionIndex = 3;
	static constexpr int kVelocityIndex = 6;
	static constexpr int kGyroBiasIndex = 9;
	static constexpr int kAccBiasIndex = 12;
	static constexpr int kGravityIndex = 15;
	static constexpr double kGravityNorm = 9.81;

	using Ptr = std::shared_ptr<State>;
	using ConstPtr = std::shared_ptr<const State>;
	using Vector = Eigen::Matrix<double, kStateDim, 1>;
	using Covariance = Eigen::Matrix<double, kStateDim, kStateDim>;

	State();
	State(const State&) = default;
	State& operator=(const State&) = default;

	State operator+(const Vector& delta) const;
	State& operator+=(const Vector& delta);
	Vector operator-(const State& other) const;

	void Reset();
	SE3 GetPose() const;
	void SetPose(const SE3& pose);
	void Print() const;

	static Eigen::Matrix<double, 3, 2> GravityTangentBasis(const Eigen::Vector3d& gravity);
	static Ptr Create() { return std::make_shared<State>(); }

	Eigen::Matrix3d m_rotation;
	Eigen::Vector3d m_position;
	Eigen::Vector3d m_velocity;
	Eigen::Vector3d m_gyro_bias;
	Eigen::Vector3d m_acc_bias;
	Eigen::Vector3d m_gravity;
	Covariance m_covariance;
};

} // namespace lio
