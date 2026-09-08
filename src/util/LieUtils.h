#pragma once

#include <Eigen/Dense>

namespace lio {

constexpr double kEpsilon = 1e-12;

Eigen::Matrix3d Hat(const Eigen::Vector3d& v);
Eigen::Vector3d Vee(const Eigen::Matrix3d& matrix);

class SO3 {
public:
	SO3() = default;
	explicit SO3(const Eigen::Matrix3d& rotation);

	static SO3 Exp(const Eigen::Vector3d& angle_axis);
	static SO3 Identity() { return SO3(Eigen::Matrix3d::Identity()); }

	Eigen::Vector3d Log() const;
	void Normalize();

	const Eigen::Matrix3d& Matrix() const { return m_matrix; }
	Eigen::Matrix3d& Matrix() { return m_matrix; }
	SO3 Inverse() const { return SO3(m_matrix.transpose()); }
	SO3 operator*(const SO3& other) const { return SO3(m_matrix * other.m_matrix); }
	Eigen::Vector3d operator*(const Eigen::Vector3d& vector) const { return m_matrix * vector; }

private:
	Eigen::Matrix3d m_matrix = Eigen::Matrix3d::Identity();
};

class SE3 {
public:
	SE3() = default;
	SE3(const SO3& rotation, const Eigen::Vector3d& translation)
		: m_rotation(rotation), m_translation(translation) {}
	SE3(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation)
		: m_rotation(rotation), m_translation(translation) {}
	explicit SE3(const Eigen::Matrix4d& matrix);

	static SE3 FromMatrix(const Eigen::Matrix4d& matrix) { return SE3(matrix); }
	static SE3 Exp(const Eigen::Matrix<double, 6, 1>& tangent);
	static SE3 Identity() { return SE3(SO3::Identity(), Eigen::Vector3d::Zero()); }

	Eigen::Matrix<double, 6, 1> Log() const;
	Eigen::Matrix4d Matrix() const;
	Eigen::Matrix3d RotationMatrix() const { return m_rotation.Matrix(); }
	const Eigen::Vector3d& Translation() const { return m_translation; }

	SE3 operator*(const SE3& other) const {
		return SE3(m_rotation * other.m_rotation,
			m_translation + m_rotation * other.m_translation);
	}
	Eigen::Vector3d operator*(const Eigen::Vector3d& point) const {
		return m_rotation * point + m_translation;
	}
	SE3 Inverse() const {
		const SO3 inverse_rotation = m_rotation.Inverse();
		return SE3(inverse_rotation, inverse_rotation * (-m_translation));
	}

private:
	SO3 m_rotation;
	Eigen::Vector3d m_translation = Eigen::Vector3d::Zero();
};

} // namespace lio
