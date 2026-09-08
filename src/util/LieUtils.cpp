#include "LieUtils.h"

#include <cmath>

namespace lio {

Eigen::Matrix3d Hat(const Eigen::Vector3d& vector) {
	Eigen::Matrix3d matrix;
	matrix << 0.0, -vector.z(), vector.y(),
		vector.z(), 0.0, -vector.x(),
		-vector.y(), vector.x(), 0.0;
	return matrix;
}

Eigen::Vector3d Vee(const Eigen::Matrix3d& matrix) {
	return {matrix(2, 1), matrix(0, 2), matrix(1, 0)};
}

SO3::SO3(const Eigen::Matrix3d& rotation) {
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d u = svd.matrixU();
	const Eigen::Matrix3d v = svd.matrixV();
	m_matrix = u * v.transpose();
	if (m_matrix.determinant() < 0.0) {
		u.col(2) *= -1.0;
		m_matrix = u * v.transpose();
	}
}

SO3 SO3::Exp(const Eigen::Vector3d& angle_axis) {
	const double angle = angle_axis.norm();
	const Eigen::Matrix3d skew = Hat(angle_axis);
	if (angle < kEpsilon) {
		return SO3(Eigen::Matrix3d::Identity() + skew + 0.5 * skew * skew);
	}
	const double angle_squared = angle * angle;
	return SO3(Eigen::Matrix3d::Identity()
		+ std::sin(angle) / angle * skew
		+ (1.0 - std::cos(angle)) / angle_squared * skew * skew);
}

Eigen::Vector3d SO3::Log() const {
	Eigen::AngleAxisd angle_axis(m_matrix);
	if (std::abs(angle_axis.angle()) < kEpsilon) {
		return Eigen::Vector3d::Zero();
	}
	return angle_axis.angle() * angle_axis.axis();
}

void SO3::Normalize() {
	*this = SO3(m_matrix);
}

SE3::SE3(const Eigen::Matrix4d& matrix)
	: m_rotation(matrix.block<3, 3>(0, 0)),
	  m_translation(matrix.block<3, 1>(0, 3)) {}

SE3 SE3::Exp(const Eigen::Matrix<double, 6, 1>& tangent) {
	const Eigen::Vector3d translation = tangent.head<3>();
	const Eigen::Vector3d rotation = tangent.tail<3>();
	const double angle = rotation.norm();
	const Eigen::Matrix3d skew = Hat(rotation);
	Eigen::Matrix3d jacobian = Eigen::Matrix3d::Identity();
	if (angle < kEpsilon) {
		jacobian += 0.5 * skew + skew * skew / 6.0;
	} else {
		const double angle_squared = angle * angle;
		jacobian += (1.0 - std::cos(angle)) / angle_squared * skew
			+ (angle - std::sin(angle)) / (angle_squared * angle) * skew * skew;
	}
	return SE3(SO3::Exp(rotation), jacobian * translation);
}

Eigen::Matrix<double, 6, 1> SE3::Log() const {
	Eigen::Matrix<double, 6, 1> tangent;
	const Eigen::Vector3d rotation = m_rotation.Log();
	const double angle = rotation.norm();
	const Eigen::Matrix3d skew = Hat(rotation);
	Eigen::Matrix3d inverse_jacobian = Eigen::Matrix3d::Identity() - 0.5 * skew;
	if (angle < kEpsilon) {
		inverse_jacobian += skew * skew / 12.0;
	} else {
		const double coefficient = 1.0 / (angle * angle)
			- (1.0 + std::cos(angle)) / (2.0 * angle * std::sin(angle));
		inverse_jacobian += coefficient * skew * skew;
	}
	tangent.head<3>() = inverse_jacobian * m_translation;
	tangent.tail<3>() = rotation;
	return tangent;
}

Eigen::Matrix4d SE3::Matrix() const {
	Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
	matrix.block<3, 3>(0, 0) = m_rotation.Matrix();
	matrix.block<3, 1>(0, 3) = m_translation;
	return matrix;
}

} // namespace lio
