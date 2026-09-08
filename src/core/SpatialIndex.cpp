/**
 * @file      SpatialIndex.cpp
 * @brief     Implementation of incremental spatial index for local map queries
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-06-25
 * @copyright  Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "SpatialIndex.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lio
{

void SpatialIndex::Clear()
{
	m_nodes.clear();
	m_live.clear();
	m_depth = 0;
	m_rebuilds = 0;
}

void SpatialIndex::Build(const std::vector<Point>& points)
{
	std::vector<Point> unique;
	std::unordered_map<std::uint64_t, std::size_t> slots;
	unique.reserve(points.size());
	slots.reserve(points.size());
	for (const Point& point : points)
	{
		ValidateVector(point.position);
		const auto entry = slots.emplace(point.id, unique.size());
		if (entry.second)
		{
			unique.push_back(point);
		}
		else
		{
			unique[entry.first->second] = point;
		}
	}
	ResetTree(unique);
	m_rebuilds = 0;
}

void SpatialIndex::Add(const std::vector<Point>& points)
{
	Upsert(points);
}

void SpatialIndex::Upsert(const std::vector<Point>& points)
{
	for (const Point& point : points)
	{
		ValidateVector(point.position);
	}
	if (m_live.empty() && points.size() >= 32)
	{
		Build(points);
		return;
	}
	for (const Point& point : points)
	{
		const auto found = m_live.find(point.id);
		if (found != m_live.end())
		{
			if ((m_nodes[found->second].point.position.array() == point.position.array()).all())
			{
				continue;
			}
			MarkDeleted(found->second);
		}
		Insert(point);
	}
	CompactIfNeeded();
}

bool SpatialIndex::Erase(std::uint64_t id)
{
	const auto found = m_live.find(id);
	if (found == m_live.end())
	{
		return false;
	}
	MarkDeleted(found->second);
	CompactIfNeeded();
	return true;
}

std::size_t SpatialIndex::DeleteAabbs(const std::vector<Aabb>& boxes)
{
	// Validate the whole operation before removing any point.
	for (const Aabb& box : boxes)
	{
		ValidateBox(box);
	}
	std::size_t removed = 0;
	for (const Aabb& box : boxes)
	{
		for (const Point& point : BoxSearch(box))
		{
			MarkDeleted(m_live.at(point.id));
			++removed;
		}
	}
	CompactIfNeeded();
	return removed;
}

std::vector<SpatialIndex::Point> SpatialIndex::KNearest(const Eigen::Vector3f& query, std::size_t k) const
{
	ValidateVector(query);
	k = std::min(k, ValidSize());
	if (k == 0)
	{
		return {};
	}
	std::vector<Neighbor> heap;
	heap.reserve(k);
	std::vector<std::size_t> pending{0};
	while (!pending.empty())
	{
		const std::size_t index = pending.back();
		pending.pop_back();
		const Node& node = m_nodes[index];
		if (node.live_count == 0
			|| (heap.size() == k && BoundsDistance(query, node.bounds) > heap.front().distance))
		{
			continue;
		}
		if (node.live)
		{
			const Neighbor candidate{SquaredDistance(query, node.point.position), node.point.id, index};
			if (heap.size() < k)
			{
				heap.push_back(candidate);
				std::push_heap(heap.begin(), heap.end(), Nearer);
			}
			else if (Nearer(candidate, heap.front()))
			{
				std::pop_heap(heap.begin(), heap.end(), Nearer);
				heap.back() = candidate;
				std::push_heap(heap.begin(), heap.end(), Nearer);
			}
		}
		const int near = query[node.axis] <= node.point.position[node.axis] ? 0 : 1;
		if (node.child[1 - near] != NONE)
		{
			pending.push_back(node.child[1 - near]);
		}
		if (node.child[near] != NONE)
		{
			pending.push_back(node.child[near]);
		}
	}
	std::sort_heap(heap.begin(), heap.end(), Nearer);
	return PointsFromNeighbors(heap);
}

std::vector<SpatialIndex::Point> SpatialIndex::RadiusSearch(const Eigen::Vector3f& query, float radius) const
{
	ValidateVector(query);
	if (!std::isfinite(radius) || radius < 0.0f)
	{
		throw std::invalid_argument("radius must be finite and non-negative");
	}
	if (m_live.empty())
	{
		return {};
	}
	const double limit = static_cast<double>(radius) * static_cast<double>(radius);
	std::vector<Neighbor> neighbors;
	std::vector<std::size_t> pending{0};
	while (!pending.empty())
	{
		const std::size_t index = pending.back();
		pending.pop_back();
		const Node& node = m_nodes[index];
		if (node.live_count == 0 || BoundsDistance(query, node.bounds) > limit)
		{
			continue;
		}
		const double distance = SquaredDistance(query, node.point.position);
		if (node.live && distance <= limit)
		{
			neighbors.push_back({distance, node.point.id, index});
		}
		for (std::size_t child : node.child)
		{
			if (child != NONE)
			{
				pending.push_back(child);
			}
		}
	}
	std::sort(neighbors.begin(), neighbors.end(), Nearer);
	return PointsFromNeighbors(neighbors);
}

std::vector<SpatialIndex::Point> SpatialIndex::BoxSearch(const Aabb& box) const
{
	ValidateBox(box);
	if (m_live.empty())
	{
		return {};
	}
	std::vector<Point> points;
	std::vector<std::size_t> pending{0};
	while (!pending.empty())
	{
		const Node& node = m_nodes[pending.back()];
		pending.pop_back();
		if (node.live_count == 0
			|| (box.minimum.array() > node.bounds.maximum.array()).any()
			|| (box.maximum.array() < node.bounds.minimum.array()).any())
		{
			continue;
		}
		if (node.live && (node.point.position.array() >= box.minimum.array()).all()
			&& (node.point.position.array() <= box.maximum.array()).all())
		{
			points.push_back(node.point);
		}
		for (std::size_t child : node.child)
		{
			if (child != NONE)
			{
				pending.push_back(child);
			}
		}
	}
	std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.id < b.id; });
	return points;
}

std::vector<SpatialIndex::Point> SpatialIndex::BoxSearch(const Eigen::Vector3f& minimum, const Eigen::Vector3f& maximum) const
{
	return BoxSearch({minimum, maximum});
}

std::size_t SpatialIndex::Size() const
{
	return m_nodes.size();
}

std::size_t SpatialIndex::ValidSize() const
{
	return m_live.size();
}

SpatialIndex::Statistics SpatialIndex::GetStatistics() const
{
	return {Size(), ValidSize(), Size() - ValidSize(), m_depth, m_rebuilds};
}

bool SpatialIndex::Nearer(const Neighbor& a, const Neighbor& b)
{
	return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}

void SpatialIndex::ValidateVector(const Eigen::Vector3f& value)
{
	if (!value.allFinite())
	{
		throw std::invalid_argument("coordinates must be finite");
	}
}

void SpatialIndex::ValidateBox(const Aabb& box)
{
	ValidateVector(box.minimum);
	ValidateVector(box.maximum);
	if ((box.minimum.array() > box.maximum.array()).any())
	{
		throw std::invalid_argument("box bounds must be ordered");
	}
}

double SpatialIndex::SquaredDistance(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
	double result = 0.0;
	for (int axis = 0; axis < 3; ++axis)
	{
		const double delta = static_cast<double>(a[axis]) - static_cast<double>(b[axis]);
		result += delta * delta;
	}
	return result;
}

double SpatialIndex::BoundsDistance(const Eigen::Vector3f& query, const Aabb& bounds)
{
	double result = 0.0;
	for (int axis = 0; axis < 3; ++axis)
	{
		const double coordinate = query[axis];
		const double closest = std::clamp(coordinate,
			static_cast<double>(bounds.minimum[axis]), static_cast<double>(bounds.maximum[axis]));
		const double delta = coordinate - closest;
		result += delta * delta;
	}
	return result;
}

bool SpatialIndex::SplitLess(const Point& a, const Point& b, int axis)
{
	for (int offset = 0; offset < 3; ++offset)
	{
		const int dimension = (axis + offset) % 3;
		if (a.position[dimension] != b.position[dimension])
		{
			return a.position[dimension] < b.position[dimension];
		}
	}
	return a.id < b.id;
}

std::vector<SpatialIndex::Point> SpatialIndex::PointsFromNeighbors(const std::vector<Neighbor>& neighbors) const
{
	std::vector<Point> points;
	points.reserve(neighbors.size());
	for (const Neighbor& neighbor : neighbors)
	{
		points.push_back(m_nodes[neighbor.node].point);
	}
	return points;
}

void SpatialIndex::ResetTree(std::vector<Point>& points)
{
	m_nodes.clear();
	m_nodes.reserve(points.size());
	m_live.clear();
	m_live.reserve(points.size());
	m_depth = 0;
	BuildRange(points, 0, points.size(), NONE, 0);
}

std::size_t SpatialIndex::BuildRange(std::vector<Point>& points, std::size_t begin, std::size_t end,
	std::size_t parent, std::size_t depth)
{
	if (begin == end)
	{
		return NONE;
	}
	const int axis = static_cast<int>(depth % 3);
	const std::size_t mid = begin + (end - begin) / 2;
	std::nth_element(points.begin() + begin, points.begin() + mid, points.begin() + end,
		[axis](const Point& a, const Point& b) { return SplitLess(a, b, axis); });
	const std::size_t index = m_nodes.size();
	Node node;
	node.point = points[mid];
	node.bounds = {node.point.position, node.point.position};
	node.parent = parent;
	node.axis = axis;
	m_nodes.push_back(node);
	m_live.emplace(node.point.id, index);
	m_depth = std::max(m_depth, depth + 1);
	const std::size_t left = BuildRange(points, begin, mid, index, depth + 1);
	const std::size_t right = BuildRange(points, mid + 1, end, index, depth + 1);
	m_nodes[index].child[0] = left;
	m_nodes[index].child[1] = right;
	for (std::size_t child : {left, right})
	{
		if (child != NONE)
		{
			m_nodes[index].bounds.minimum = m_nodes[index].bounds.minimum.cwiseMin(m_nodes[child].bounds.minimum);
			m_nodes[index].bounds.maximum = m_nodes[index].bounds.maximum.cwiseMax(m_nodes[child].bounds.maximum);
			m_nodes[index].live_count += m_nodes[child].live_count;
		}
	}
	return index;
}

void SpatialIndex::Insert(const Point& point)
{
	std::size_t parent = NONE;
	std::size_t cursor = m_nodes.empty() ? NONE : 0;
	std::size_t depth = 1;
	int side = 0;
	while (cursor != NONE)
	{
		parent = cursor;
		const Node& node = m_nodes[cursor];
		side = SplitLess(point, node.point, node.axis) ? 0 : 1;
		cursor = node.child[side];
		++depth;
	}
	Node node;
	node.point = point;
	node.bounds = {point.position, point.position};
	node.parent = parent;
	node.axis = static_cast<int>((depth - 1) % 3);
	const std::size_t index = m_nodes.size();
	m_nodes.push_back(node);
	m_live[point.id] = index;
	m_depth = std::max(m_depth, depth);
	if (parent != NONE)
	{
		m_nodes[parent].child[side] = index;
	}
	for (cursor = parent; cursor != NONE; cursor = m_nodes[cursor].parent)
	{
		Node& ancestor = m_nodes[cursor];
		++ancestor.live_count;
		ancestor.bounds.minimum = ancestor.bounds.minimum.cwiseMin(point.position);
		ancestor.bounds.maximum = ancestor.bounds.maximum.cwiseMax(point.position);
	}
}

void SpatialIndex::MarkDeleted(std::size_t index)
{
	m_nodes[index].live = false;
	m_live.erase(m_nodes[index].point.id);
	for (std::size_t cursor = index; cursor != NONE; cursor = m_nodes[cursor].parent)
	{
		--m_nodes[cursor].live_count;
	}
}

void SpatialIndex::CompactIfNeeded()
{
	if (Size() < 32)
	{
		return;
	}
	const std::size_t balanced_depth = static_cast<std::size_t>(
		std::ceil(std::log2(static_cast<double>(ValidSize() + 1))));
	if (Size() - ValidSize() <= ValidSize() && m_depth <= 2 * balanced_depth + 4)
	{
		return;
	}
	std::vector<Point> points;
	points.reserve(ValidSize());
	for (const Node& node : m_nodes)
	{
		if (node.live)
		{
			points.push_back(node.point);
		}
	}
	ResetTree(points);
	++m_rebuilds;
}

}  // namespace lio
