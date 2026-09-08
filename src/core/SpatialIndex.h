/**
 * @file      SpatialIndex.h
 * @brief     Incremental spatial index for local map queries
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-06-25
 * @copyright  Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LIO_SPATIAL_INDEX_H
#define LIO_SPATIAL_INDEX_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

namespace lio
{

// Queries are read-only. The owner serializes updates against queries.
class SpatialIndex
{
public:
	struct Point
	{
		std::uint64_t id = 0;
		Eigen::Vector3f position = Eigen::Vector3f::Zero();
	};

	struct Aabb
	{
		Eigen::Vector3f minimum = Eigen::Vector3f::Zero();
		Eigen::Vector3f maximum = Eigen::Vector3f::Zero();
	};

	struct Statistics
	{
		std::size_t node_count = 0;
		std::size_t valid_count = 0;
		std::size_t deleted_count = 0;
		std::size_t max_depth = 0;
		std::size_t rebuild_count = 0;
	};

	void Clear();

	// IDs identify points; the last occurrence of an ID wins.
	void Build(const std::vector<Point>& points);

	void Add(const std::vector<Point>& points);

	void Upsert(const std::vector<Point>& points);

	bool Erase(std::uint64_t id);

	std::size_t DeleteAabbs(const std::vector<Aabb>& boxes);

	// Exact squared-distance order, then ascending ID, including ties at k.
	std::vector<Point> KNearest(const Eigen::Vector3f& query, std::size_t k) const;

	std::vector<Point> RadiusSearch(const Eigen::Vector3f& query, float radius) const;

	// Boxes and radii are closed: boundary points are included.
	std::vector<Point> BoxSearch(const Aabb& box) const;

	std::vector<Point> BoxSearch(const Eigen::Vector3f& minimum, const Eigen::Vector3f& maximum) const;

	std::size_t Size() const;
	std::size_t ValidSize() const;

	Statistics GetStatistics() const;

private:
	static constexpr std::size_t NONE = std::numeric_limits<std::size_t>::max();

	struct Node
	{
		Point point;
		Aabb bounds;
		std::size_t child[2] = {NONE, NONE};
		std::size_t parent = NONE;
		std::size_t live_count = 1;
		int axis = 0;
		bool live = true;
	};

	struct Neighbor
	{
		double distance;
		std::uint64_t id;
		std::size_t node;
	};

	static bool Nearer(const Neighbor& a, const Neighbor& b);

	static void ValidateVector(const Eigen::Vector3f& value);

	static void ValidateBox(const Aabb& box);

	static double SquaredDistance(const Eigen::Vector3f& a, const Eigen::Vector3f& b);

	static double BoundsDistance(const Eigen::Vector3f& query, const Aabb& bounds);

	static bool SplitLess(const Point& a, const Point& b, int axis);

	std::vector<Point> PointsFromNeighbors(const std::vector<Neighbor>& neighbors) const;

	void ResetTree(std::vector<Point>& points);

	std::size_t BuildRange(std::vector<Point>& points, std::size_t begin, std::size_t end,
		std::size_t parent, std::size_t depth);

	void Insert(const Point& point);

	void MarkDeleted(std::size_t index);

	void CompactIfNeeded();

	std::vector<Node> m_nodes;
	std::unordered_map<std::uint64_t, std::size_t> m_live;
	std::size_t m_depth = 0;
	std::size_t m_rebuilds = 0;
};

}  // namespace lio

#endif  // LIO_SPATIAL_INDEX_H
