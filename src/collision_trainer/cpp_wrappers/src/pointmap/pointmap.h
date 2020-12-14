#pragma once

#include "../cloud/cloud.h"

#include <set>
#include <cstdint>
#include <cmath> 
#include <algorithm>

#include "../nanoflann/nanoflann.hpp"

using namespace std;

// KDTree type definition
typedef nanoflann::KDTreeSingleIndexAdaptorParams KDTree_Params;
typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> PointXYZ_KDTree;
typedef nanoflann::KDTreeSingleIndexDynamicAdaptor< nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> PointXYZ_Dynamic_KDTree;

//-------------------------------------------------------------------------------------------
//
// PointMapPython Class
// ********************
//
//	PointMap designed to be used in python. As it is hard to transfert unordered map to 
//	python dict structure, we rebuild the hashmap every update (not very efficient).
//
//-------------------------------------------------------------------------------------------

class MapVoxelData
{
public:

	// Elements
	// ********

	bool occupied;
	int count;
	PointXYZ centroid;
	PointXYZ normal;
	float score;


	// Methods
	// *******

	// Constructor
	MapVoxelData()
	{
		occupied = false;
		count = 0;
		score = -1.0f;
		centroid = PointXYZ();
		normal = PointXYZ();
	}
	MapVoxelData(const PointXYZ p0, const PointXYZ n0, const float s0, const int c0)
	{
		occupied = true;
		count = c0;
		score = s0;
		centroid = p0;
		normal = n0;
	}

	MapVoxelData(const PointXYZ p0)
	{
		// We initiate only the centroid
		count = 1;
		centroid = p0;

		// Other varaible are kept null
		occupied = false;
		score = -1.0f;
		normal = PointXYZ();
	}

	void update_centroid(const PointXYZ p0)
	{
		count += 1;
		centroid += p0;
	}

	void update_normal(const float s0, const PointXYZ n0)
	{
		// We keep the worst normal
		occupied = true;

		// Rule for normal update:
		// IF current_score=2 : normal was computed with planarity in the map, do not modify
		// IF s0 < score - 0.1 : Too bad score dont update (This includes the condition above)
		// IF s0 > score + 0.1 : Better score, use new normal
		// IF abs(s0 - score) < 0.1 : Similar score, avergae normal
		// When averaging be careful of orientation. Dont worry about norm, we renormalize every normal in the end

		if (s0 > score + 0.1)
		{
			score = s0;
			normal = n0;
		}
		else if (s0 > score - 0.1)
		{
			if (s0 > score)
				score = s0;
			if (normal.dot(n0) > 0)
				normal += n0;
			else
				normal -= n0;
		}
	}
	
};


class PointMapPython
{
public:

	// Elements
	// ********

	float dl;
	vector<PointXYZ> points;
	vector<PointXYZ> normals;
	vector<float> scores;
	vector<int> counts;


	// Methods
	// *******

	// Constructor
	PointMapPython()
	{
		dl = 1.0f;
	}
	PointMapPython(const float dl0)
	{
		dl = dl0;
	}


	// Methods
	void update(vector<PointXYZ>& points0, vector<PointXYZ>& normals0, vector<float>& scores0);

	void init_samples(const PointXYZ originCorner,
		const PointXYZ maxCorner,
		unordered_map<size_t, MapVoxelData>& samples);

	void add_samples(const vector<PointXYZ>& points0,
		const vector<PointXYZ>& normals0,
		const vector<float>& scores0,
		const PointXYZ originCorner,
		const PointXYZ maxCorner,
		unordered_map<size_t, MapVoxelData>& samples);

	size_t size() { return points.size(); }
};


//-------------------------------------------------------------------------------------------
//
// VoxKey
// ******
//
//	Here we define a struct that will be used as key in our hash map. It contains 3 integers.
//  Then we specialize the std::hash function for this class.
//
//-------------------------------------------------------------------------------------------

class VoxKey
{
public:
	int x;
	int y;
	int z;

	VoxKey() { x = 0; y = 0; z = 0; }
	VoxKey(int x0, int y0, int z0) { x = x0; y = y0; z = z0; }

	bool operator==(const VoxKey& other) const
	{
		return (x == other.x && y == other.y && z == other.z);
	}

};

inline VoxKey operator + (const VoxKey A, const VoxKey B)
{
	return VoxKey(A.x + B.x, A.y + B.y, A.z + B.z);
}

// Simple utility function to combine hashtables
template <typename T, typename... Rest>
void hash_combine(std::size_t& seed, const T& v, const Rest&... rest)
{
	seed ^= std::hash<T>{}(v)+0x9e3779b9 + (seed << 6) + (seed >> 2);
	(hash_combine(seed, rest), ...);
}

// Specialization of std:hash function
namespace std 
{
	template <>
	struct hash<VoxKey>
	{
		std::size_t operator()(const VoxKey& k) const
		{
			std::size_t ret = 0;
			hash_combine(ret, k.x, k.y, k.z);
			return ret;
		}
	};
}


//-------------------------------------------------------------------------------------------
//
// PointMap Class
// **************
//
//	PointMap designed to be used in C++. Everything should be more efficient here.
//
//-------------------------------------------------------------------------------------------


class PointMapOld
{
public:

	// Elements
	// ********

	// Voxel size
	float dl;

	// Containers for the data
	vector<PointXYZ> points;
	vector<PointXYZ> normals;
	vector<float> scores;

	// Containers for the data
	vector<PointXYZ> voxPoints;
	vector<PointXYZ> voxNormals;
	vector<float> voxScores;
	vector<int> voxCounts;

	// Sparse hashmap that contain voxels (each voxel data is in the contiguous vector containers)
	unordered_map<VoxKey, size_t> samples;


	// Methods
	// *******

	// Constructor
	PointMapOld()
	{
		dl = 1.0f;
	}
	PointMapOld(const float dl0)
	{
		dl = dl0;
	}
	PointMapOld(const float dl0,
		vector<PointXYZ>& init_points,
		vector<PointXYZ>& init_normals,
		vector<float>& init_scores)
	{
		dl = dl0;
		update(init_points, init_normals, init_scores);
	}

	// Size of the map (number of point/voxel in the map)
	size_t size() { return voxPoints.size(); }

	// Init of voxel centroid
	void init_sample_centroid(const VoxKey& k, const PointXYZ& p0)
	{
		// We place anew key in the hashmap
		samples.emplace(k, voxPoints.size());

		// We add new voxel data but initiate only the centroid
		voxPoints.push_back(p0);
		voxCounts.push_back(1);
		voxNormals.push_back(PointXYZ());
		voxScores.push_back(-1.0f);
	}

	// Update of voxel centroid
	void update_sample_centroid(const VoxKey& k, const PointXYZ& p0)
	{
		// Update count of points and centroid of the cell
		voxCounts[samples[k]] += 1;
		voxPoints[samples[k]] += p0;
	}

	// Update of voxel normal
	void update_sample_normal(PointXYZ& normal, float& score, const PointXYZ& n0, const float& s0)
	{
		// Rule for normal update:
		// IF current_score=2 : normal was computed with planarity in the map, do not modify
		// IF s0 < score - 0.1 : Too bad score dont update (This includes the condition above)
		// IF s0 > score + 0.1 : Better score, use new normal
		// IF abs(s0 - score) < 0.1 : Similar score, avergae normal
		// When averaging be careful of orientation. Dont worry about norm, we renormalize every normal in the end

		if (s0 > score + 0.1)
		{
			score = s0;
			normal = n0;
		}
		else if (s0 > score - 0.1)
		{
			if (s0 > score)
				score = s0;
			if (normal.dot(n0) > 0)
				normal += n0;
			else
				normal -= n0;
		}
	}

	// Update map with a set of new points
	void update(vector<PointXYZ>& points0, vector<PointXYZ>& normals0, vector<float>& scores0)
	{

		// Reserve new space if needed
		if (samples.size() < 1)
			samples.reserve(10 * points0.size());


		std::cout << std::endl << "--------------------------------------" << std::endl;
		std::cout << "current max_load_factor: " << samples.max_load_factor() << std::endl;
		std::cout << "current size: " << samples.size() << std::endl;
		std::cout << "current bucket_count: " << samples.bucket_count() << std::endl;
		std::cout << "current load_factor: " << samples.load_factor() << std::endl;
		std::cout << "--------------------------------------" << std::endl << std::endl;

		// Initialize variables
		float r = 1.5;
		float r2 = r * r;
		float inv_dl = 1 / dl;
		size_t i = 0;
		VoxKey k0, k;

		for (auto& p : points0)
		{
			// Position of point in sample map
			PointXYZ p_pos = p * inv_dl;

			// Corresponding key
			k0.x = (int)floor(p_pos.x);
			k0.y = (int)floor(p_pos.y);
			k0.z = (int)floor(p_pos.z);

			// Update the adjacent cells
			for (k.x = k0.x - 1; k.x < k0.x + 2; k.x++)
			{

				for (k.y = k0.y - 1; k.y < k0.y + 2; k.y++)
				{

					for (k.z = k0.z - 1; k.z < k0.z + 2; k.z++)
					{
						// Center of updated cell in grid coordinates
						PointXYZ cellCenter(k.x + 0.5, k.y + 0.5, k.z + 0.5);

						// Update barycenter if in range
						float d2 = (cellCenter - p_pos).sq_norm();
						if (d2 < r2)
						{
							if (samples.count(k) < 1)
								init_sample_centroid(k, p);
							else
								update_sample_centroid(k, p);
						}
					}
				}
			}

			// Update the point normal
			update_sample_normal(voxNormals[samples[k0]], voxScores[samples[k0]], normals0[i], scores0[i]);
			i++;
		}

		// Now update vector containers only with voxel whose centroid is in their voxel
		points.reserve(samples.size());
		normals.reserve(samples.size());
		scores.reserve(samples.size());
		i = 0;
		for (auto& v : samples)
		{
			// Check if centroid is in cell
			PointXYZ centroid = voxPoints[v.second] * (1.0 / voxCounts[v.second]);
			PointXYZ centroid_pos = centroid * inv_dl;
			k0.x = (int)floor(centroid_pos.x);
			k0.y = (int)floor(centroid_pos.y);
			k0.z = (int)floor(centroid_pos.z);

			float score = voxScores[v.second];
			if (score > -1.5 && k0 == v.first)
			{
				PointXYZ normal = voxNormals[v.second];
				normal *= 1.0 / (sqrt(normal.sq_norm()) + 1e-6);
				if (i < points.size())
				{
					points[i] = centroid;
					normals[i] = normal;
					scores[i] = score;
				}
				else
				{
					points.push_back(centroid);
					normals.push_back(normal);
					scores.push_back(score);
				}
				i++;
			}
		}
	}



};


class PointMap
{
public:

	// Elements
	// ********

	// Voxel size
	float dl;

	// Count the number of frames used tu update this map
	int update_idx;

	// Containers for the data
	PointCloud cloud;
	vector<PointXYZ> normals;
	vector<float> scores;
	vector<int> counts;

	// Sparse hashmap that contain voxels (each voxel data is in the contiguous vector containers)
	unordered_map<VoxKey, size_t> samples;

	// KDTree for neighbors query
	PointXYZ_Dynamic_KDTree tree;


	// Methods
	// *******

	// Constructor
	PointMap() : tree(3, cloud, KDTree_Params(10 /* max leaf */))
	{
		dl = 1.0f;
		update_idx = 0;
	}
	PointMap(const float dl0) : tree(3, cloud, KDTree_Params(10 /* max leaf */))
	{
		dl = dl0;
		update_idx = 0;
	}
	PointMap(const float dl0,
		const float max_dist0,
		vector<PointXYZ>& init_points,
		vector<PointXYZ>& init_normals,
		vector<float>& init_scores) : tree(3, cloud, KDTree_Params(10 /* max leaf */))
	{
		dl = dl0;
		update_idx = -1;
		update(init_points, init_normals, init_scores);
	}

	// Size of the map (number of point/voxel in the map)
	size_t size() { return cloud.pts.size(); }

	// Init of voxel centroid
	void init_sample(const VoxKey& k, const PointXYZ& p0, const PointXYZ& n0, const float& s0 , const int& c0)
	{
		// We place anew key in the hashmap
		samples.emplace(k, cloud.pts.size());

		// We add new voxel data but initiate only the centroid
		cloud.pts.push_back(p0);
		normals.push_back(n0);
		scores.push_back(s0);

		// Count is useless, instead save index of first frame placing a point in this cell
		counts.push_back(c0);
	}

	// Update of voxel centroid
	void update_sample(const size_t idx, const PointXYZ& p0, const PointXYZ& n0, const float& s0)
	{
		// Update count for optional removal count of points (USELESS see init_sample)
		//counts[idx] += 1;

		// Update normal if we have a clear view of it  and closer distance (see computation of score)
		if (s0 > scores[idx])
		{
			scores[idx] = s0;
			normals[idx] = n0;
		}
	}

	// Update map with a set of new points
	void update(vector<PointXYZ>& points0, vector<PointXYZ>& normals0, vector<float>& scores0)
	{

		// Reserve new space if needed
		if (samples.size() < 1)
			samples.reserve(10 * points0.size());
		if (cloud.pts.capacity() < cloud.pts.size() + points0.size())
		{
			cloud.pts.reserve(cloud.pts.capacity() + points0.size());
			counts.reserve(counts.capacity() + points0.size());
			normals.reserve(normals.capacity() + points0.size());
			scores.reserve(scores.capacity() + points0.size());
		}

		//std::cout << std::endl << "--------------------------------------" << std::endl;
		//std::cout << "current max_load_factor: " << samples.max_load_factor() << std::endl;
		//std::cout << "current size: " << samples.size() << std::endl;
		//std::cout << "current bucket_count: " << samples.bucket_count() << std::endl;
		//std::cout << "current load_factor: " << samples.load_factor() << std::endl;
		//std::cout << "--------------------------------------" << std::endl << std::endl;

		// Initialize variables
		float inv_dl = 1 / dl;
		size_t i = 0;
		VoxKey k0;
		size_t num_added = 0;

		for (auto& p : points0)
		{
			// Position of point in sample map
			PointXYZ p_pos = p * inv_dl;

			// Corresponding key
			k0.x = (int)floor(p_pos.x);
			k0.y = (int)floor(p_pos.y);
			k0.z = (int)floor(p_pos.z);

			// check angle

			// Update the point count
			if (samples.count(k0) < 1)
			{
				init_sample(k0, p, normals0[i], scores0[i], update_idx);
				num_added++;
			}
			else
			{
				update_sample(samples[k0], p, normals0[i], scores0[i]);
			}
			i++;
		}

		// Update tree
		tree.addPoints(cloud.pts.size() - num_added, cloud.pts.size() - 1);

		// Update frame count
		update_idx++;

	}

};











