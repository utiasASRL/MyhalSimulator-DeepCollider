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
//		Cloud source :
//		Define usefull Functions/Methods
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//


#include "cloud.h"


// Getters
// *******

PointXYZ max_point(const std::vector<PointXYZ>& points)
{
	// Initialize limits
	PointXYZ maxP(points[0]);

	// Loop over all points
	for (auto p : points)
	{
		if (p.x > maxP.x)
			maxP.x = p.x;

		if (p.y > maxP.y)
			maxP.y = p.y;

		if (p.z > maxP.z)
			maxP.z = p.z;
	}

	return maxP;
}


PointXYZ min_point(const std::vector<PointXYZ>& points)
{
	// Initialize limits
	PointXYZ minP(points[0]);

	// Loop over all points
	for (auto p : points)
	{
		if (p.x < minP.x)
			minP.x = p.x;

		if (p.y < minP.y)
			minP.y = p.y;

		if (p.z < minP.z)
			minP.z = p.z;
	}

	return minP;
}


PointXYZ max_point(const PointXYZ A, const PointXYZ B)
{
	// Initialize limits
	PointXYZ maxP(A);
	if (B.x > maxP.x)
		maxP.x = B.x;
	if (B.y > maxP.y)
		maxP.y = B.y;
	if (B.z > maxP.z)
		maxP.z = B.z;
	return maxP;
}

PointXYZ min_point(const PointXYZ A, const PointXYZ B)
{
	// Initialize limits
	PointXYZ maxP(A);
	if (B.x < maxP.x)
		maxP.x = B.x;
	if (B.y < maxP.y)
		maxP.y = B.y;
	if (B.z < maxP.z)
		maxP.z = B.z;
	return maxP;
}


// Debug functions
// ***************

void save_cloud(std::string dataPath, std::vector<PointXYZ>& points)
{
	std::vector<float> none;
	save_cloud(dataPath, points, none);
}


void save_cloud(std::string dataPath, std::vector<PointXYZ>& points, std::vector<float>& features)
{
	// Variables
	uint64_t num_points = points.size();
	uint64_t num_features = features.size() / num_points;

	// Safe check
	if (num_features * num_points != features.size())
	{
		std::cout << "Warning: features dimension do not match point cloud" << std::endl;
		return;
	}


	// Open file
	npm::PLYFileOut file(dataPath);

	// Push fields
	file.pushField(num_points, 3, npm::PLY_FLOAT, { "x", "y", "z" }, points);

	std::vector<std::vector<float>> fields(num_features);
	for (size_t i = 0; i < num_features; i++)
	{
		char buffer[100];
		sprintf(buffer, "f%d", (int)i);
		fields[i] = std::vector<float>(features.begin() + i * num_points, features.begin() + (i + 1) * num_points);
		file.pushField(num_points, 1, npm::PLY_FLOAT, { std::string(buffer) }, fields[i]);
	}
	file.write();
}






