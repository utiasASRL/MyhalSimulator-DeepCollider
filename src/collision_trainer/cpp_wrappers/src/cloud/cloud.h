//
//
//		0==========================0
//		|    Local feature test    |
//		0==========================0
//
//		version 1.0 : 
//			> 
//
//---------------------------------------------------
//
//		Cloud header
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//


# pragma once

#include "points.h"


//------------------------------------------------------------------------------------------------------------
// PointCloud class
// ****************
//
//------------------------------------------------------------------------------------------------------------

struct PointCloud
{

	std::vector<PointXYZ>  pts;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline float kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		if (dim == 0) return pts[idx].x;
		else if (dim == 1) return pts[idx].y;
		else return pts[idx].z;
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};


// Utility function for pointclouds
void filter_pointcloud(std::vector<PointXYZ>& pts, std::vector<float>& scores, float filter_value);
void filter_floatvector(std::vector<float>& vec, std::vector<float>& scores, float filter_value);
void filter_floatvector(std::vector<float>& vec, float filter_value);


// PLY reading/saving functions
void save_cloud(std::string dataPath, std::vector<PointXYZ>& points, std::vector<PointXYZ>& normals, std::vector<float>& features);
void save_cloud(std::string dataPath, std::vector<PointXYZ>& points, std::vector<float>& features);
void save_cloud(std::string dataPath, std::vector<PointXYZ>& points, std::vector<PointXYZ>& normals);
void save_cloud(std::string dataPath, std::vector<PointXYZ>& points);


void load_cloud(std::string& dataPath,
	std::vector<PointXYZ>& points);

void load_cloud(std::string& dataPath,
	std::vector<PointXYZ>& points,
	std::vector<float>& float_scalar,
	std::string& float_scalar_name,
	std::vector<int>& int_scalar,
	std::string& int_scalar_name);







