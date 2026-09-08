#include "State.h"

#include <cmath>
#include <spdlog/spdlog.h>

namespace lio {

State::State() {
	Reset();
}

void State::Reset() {
	m_rotation.setIdentity();
	m_position.setZero();
	m_velocity.setZero();
	m_gyro_bias.setZero();
	m_acc_bias.setZero();
	m_gravity = {0.0, 0.0, -kGravityNorm};
	m_covariance = Covariance::Identity() * 0.01;
	m_covariance.block<3, 3>(kRotationIndex, kRotationIndex) *= 1e-3;
	m_covariance.block<3, 3>(kGyroBiasIndex, kGyroBiasIndex) *= 1e-3;
	m_covariance.block<3, 3>(kAccBiasIndex, kAccBiasIndex) *= 1e-3;
	m_covariance.block<2, 2>(kGravityIndex, kGravityIndex) *= 1e-3;
}

Eigen::Matrix<double, 3, 2> State::GravityTangentBasis(const Eigen::Vector3d& gravity) {
	const Eigen::Vector3d direction = gravity.normalized();
	const Eigen::Vector3d reference = std::abs(direction.z()) < 0.9
		? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
	Eigen::Matrix<double, 3, 2> basis;
	basis.col(0) = direction.cross(reference).normalized();
	basis.col(1) = direction.cross(basis.col(0)).normalized();
	return basis;
}

State State::operator+(const Vector& delta) const {
	State result(*this);
	result.m_rotation = m_rotation * SO3::Exp(delta.segment<3>(kRotationIndex)).Matrix();
	result.m_position += delta.segment<3>(kPositionIndex);
	result.m_velocity += delta.segment<3>(kVelocityIndex);
	result.m_gyro_bias += delta.segment<3>(kGyroBiasIndex);
	result.m_acc_bias += delta.segment<3>(kAccBiasIndex);
	const Eigen::Vector3d gravity_update = m_gravity
		+ GravityTangentBasis(m_gravity) * delta.segment<2>(kGravityIndex);
	result.m_gravity = kGravityNorm * gravity_update.normalized();
	return result;
}

State& State::operator+=(const Vector& delta) {
	*this = *this + delta;
	return *this;
}

State::Vector State::operator-(const State& other) const {
	Vector delta = Vector::Zero();
	delta.segment<3>(kRotationIndex) = SO3(other.m_rotation.transpose() * m_rotation).Log();
	delta.segment<3>(kPositionIndex) = m_position - other.m_position;
	delta.segment<3>(kVelocityIndex) = m_velocity - other.m_velocity;
	delta.segment<3>(kGyroBiasIndex) = m_gyro_bias - other.m_gyro_bias;
	delta.segment<3>(kAccBiasIndex) = m_acc_bias - other.m_acc_bias;
	delta.segment<2>(kGravityIndex) = GravityTangentBasis(other.m_gravity).transpose()
		* (m_gravity - other.m_gravity);
	return delta;
}

SE3 State::GetPose() const {
	return SE3(m_rotation, m_position);
}

void State::SetPose(const SE3& pose) {
	m_rotation = pose.RotationMatrix();
	m_position = pose.Translation();
}

void State::Print() const {
	spdlog::info("Position: [{:.4f}, {:.4f}, {:.4f}]", m_position.x(), m_position.y(), m_position.z());
	spdlog::info("Velocity: [{:.4f}, {:.4f}, {:.4f}]", m_velocity.x(), m_velocity.y(), m_velocity.z());
	spdlog::info("Gyro bias: [{:.6f}, {:.6f}, {:.6f}]", m_gyro_bias.x(), m_gyro_bias.y(), m_gyro_bias.z());
	spdlog::info("Acc bias: [{:.6f}, {:.6f}, {:.6f}]", m_acc_bias.x(), m_acc_bias.y(), m_acc_bias.z());
	spdlog::info("Gravity: [{:.4f}, {:.4f}, {:.4f}]", m_gravity.x(), m_gravity.y(), m_gravity.z());
}

} // namespace lio
