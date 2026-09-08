#pragma once

#include "PointCloudUtils.h"
#include "ProbabilisticKernelOptimizer.h"
#include "State.h"
#include "VoxelMap.h"

#include <Eigen/Dense>
#include <deque>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

namespace lio {

struct IMUData {
	double timestamp;
	Eigen::Vector3d acc;
	Eigen::Vector3d gyr;

	IMUData(double time, const Eigen::Vector3d& acceleration, const Eigen::Vector3d& angular_velocity)
		: timestamp(time), acc(acceleration), gyr(angular_velocity) {}
};

struct StateWithTimestamp {
	State state;
	double timestamp = 0.0;

	StateWithTimestamp() = default;
	StateWithTimestamp(const State& value, double time) : state(value), timestamp(time) {}
};

struct LidarData {
	double timestamp;
	PointCloudPtr cloud;

	LidarData(double time, PointCloudPtr points) : timestamp(time), cloud(std::move(points)) {}
};

struct MapPoint {
	Eigen::Vector3d position;
	Eigen::Vector3d normal;
	double timestamp;
	int frame_id;

	MapPoint(const Eigen::Vector3d& point, const Eigen::Vector3d& surface_normal, double time, int id)
		: position(point), normal(surface_normal), timestamp(time), frame_id(id) {}
};

struct EstimatorTestAccess;

class Estimator {
public:
	using Correspondence = std::tuple<Eigen::Vector3d, Eigen::Vector3d, double, size_t>;
	using NoiseJacobian = Eigen::Matrix<double, State::kStateDim, 12>;

	Estimator();
	~Estimator();

	void UpdateProcessNoise();
	void Initialize(const IMUData& first_imu);
	bool GravityInitialization(const std::vector<IMUData>& imu_buffer);
	void ProcessIMU(const IMUData& imu);
	void ProcessLidar(const LidarData& lidar);
	State GetCurrentState() const;
	std::vector<State> GetTrajectory() const;
	bool IsInitialized() const { return m_initialized; }

	struct Statistics {
		int total_frames = 0;
		int successful_registrations = 0;
		double avg_processing_time_ms = 0.0;
		double total_distance = 0.0;
		double avg_translation_error = 0.0;
		double avg_rotation_error = 0.0;
	};

	Statistics GetStatistics() const;
	std::shared_ptr<VoxelMap> GetVoxelMap() const;
	void PrintProcessingTimeStatistics() const;
	const std::vector<double>& GetProcessingTimes() const { return m_processing_times; }

	PointCloudPtr GetMapPointCloud() const {
		std::lock_guard<std::mutex> lock(m_map_mutex);
		auto result = std::make_shared<PointCloud>();
		if (!m_voxel_map) {
			return result;
		}
		for (const auto& centroid : m_voxel_map->GetL0Centroids()) {
			Point3D point;
			point.x = centroid.x();
			point.y = centroid.y();
			point.z = centroid.z();
			point.intensity = 1.0f;
			result->push_back(point);
		}
		return result;
	}

	PointCloudPtr GetProcessedCloud() const {
		std::lock_guard<std::mutex> lock(m_map_mutex);
		return m_processed_cloud;
	}

	struct Parameters {
		double acc_noise_std = 0.1;
		double gyr_noise_std = 0.01;
		double acc_bias_noise_std = 0.001;
		double gyr_bias_noise_std = 0.0001;
		double gravity_noise_std = 0.0;
		double lidar_noise_std = 0.05;
		int max_correspondences = 1000;
		double max_correspondence_distance = 1.0;
		int max_iterations = 10;
		double convergence_threshold = 1e-3;
		double scan_planarity_threshold = 0.1;
		double map_planarity_threshold = 0.01;
		double point_to_surfel_threshold = 0.1;
		int kdtree_knn = 5;
		double kdtree_planarity_threshold = 0.3;
		double kdtree_max_plane_residual = 0.5;
		int map_recovery_frames = 50;
		int min_surfel_inliers = 5;
		double min_linearity_ratio = 0.3;
		double voxel_size = 0.4;
		double map_voxel_size = 0.2;
		int max_map_points = 100000;
		double min_range = 0.5;
		double max_map_distance = 50.0;
		double map_box_multiplier = 2.0;
		int voxel_hierarchy_factor = 3;
		double min_plane_points = 5;
		double frustum_fov_horizontal = 90.0;
		double frustum_fov_vertical = 90.0;
		double frustum_max_range = 50.0;
		double keyframe_translation_threshold = 0.5;
		double keyframe_rotation_threshold = 10.0;
		Eigen::Matrix3d R_il = Eigen::Matrix3d::Identity();
		Eigen::Vector3d t_il = Eigen::Vector3d::Zero();
		Eigen::Vector3d gravity = {0.0, 0.0, -State::kGravityNorm};
		double min_motion_threshold = 0.1;
		int imu_buffer_size = 1000;
		bool enable_undistortion = true;
		int stride = 1;
		bool stride_then_voxel = true;
		double scan_duration = 0.1;
	} m_params;

private:
	friend struct EstimatorTestAccess;

	void PropagateState(const IMUData& imu);
	bool UpdateWithLidar(const LidarData& lidar);
	std::vector<Correspondence> FindCorrespondences(
		const PointCloudPtr scan, std::size_t maximum_correspondences = 0,
		VoxelMap::MatchDiagnostics* diagnostics = nullptr);
	void UpdateLocalMap(const PointCloudPtr scan);
	void CleanLocalMap();
	void ExtractPlanarFeatures(const PointCloudPtr cloud, std::vector<MapPoint>& features);
	void ComputeLidarJacobians(const std::vector<Correspondence>& correspondences,
		Eigen::MatrixXd& jacobian, Eigen::VectorXd& residual);
	PointCloudPtr UndistortPointCloud(const PointCloudPtr cloud,
		double scan_start_time, double scan_end_time);
	State InterpolateState(double timestamp) const;
	void BuildDiscreteImuModel(const Eigen::Vector3d& omega,
		const Eigen::Vector3d& acceleration, const Eigen::Matrix3d& rotation_mid, double dt);
	void ApplyStateCorrection(const State::Vector& correction);
	static void StabilizeCovariance(State::Covariance& covariance);
	bool ShouldUpdateMap(bool registered);

	State m_current_state;
	bool m_initialized = false;
	double m_last_update_time = 0.0;
	int m_frame_count = 0;
	unsigned int m_num_valid_correspondences = 0;
	IMUData m_previous_imu{0.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
	bool m_has_previous_imu = false;

	mutable std::mutex m_state_mutex;
	std::deque<StateWithTimestamp> m_state_history;
	std::deque<State> m_trajectory;
	std::vector<MapPoint> m_local_map;
	PointCloudPtr m_map_cloud;
	PointCloudPtr m_processed_cloud;
	std::shared_ptr<VoxelMap> m_voxel_map;
	mutable std::mutex m_map_mutex;
	std::vector<Correspondence> m_last_correspondences;
	mutable std::mutex m_stats_mutex;
	Statistics m_statistics;
	std::vector<double> m_processing_times;
	double m_sum_preprocess_time = 0.0;
	double m_sum_iekf_time = 0.0;
	double m_sum_map_time = 0.0;

	State::Covariance m_process_noise = State::Covariance::Zero();
	State::Covariance m_state_transition = State::Covariance::Identity();
	NoiseJacobian m_noise_jacobian = NoiseJacobian::Zero();
	Eigen::MatrixXd m_jacobian;
	Eigen::VectorXd m_residual_vector;
	Eigen::MatrixXd m_kalman_gain;

	bool m_first_lidar_frame = true;
	double m_last_lidar_time = 0.0;
	State m_last_lidar_state;
	Eigen::Vector3d m_last_keyframe_position = Eigen::Vector3d::Zero();
	Eigen::Matrix3d m_last_keyframe_rotation = Eigen::Matrix3d::Identity();
	bool m_first_keyframe = true;
	int m_map_recovery_frames_remaining = 0;
	std::shared_ptr<ProbabilisticKernelOptimizer> m_pko;
};

} // namespace lio
