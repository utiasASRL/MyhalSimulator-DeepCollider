#include <Python.h>
#include <numpy/arrayobject.h>
#include "../src/polar_processing/polar_processing.h"
#include <string>




// docstrings for our module
// *************************

static char module_docstring[] = "This module provides polar coordinates related functions";

static char polar_normals_docstring[] = "Gets normals from a lidar pointcloud";

static char map_frame_comp_docstring[] = "Gets difference between map and frame points in polar coordinates";


// Declare the functions
// *********************

static PyObject* polar_normals(PyObject* self, PyObject* args, PyObject* keywds);
static PyObject* map_frame_comp(PyObject* self, PyObject* args, PyObject* keywds);


// Specify the members of the module
// *********************************

static PyMethodDef module_methods[] = 
{
	{ "polar_normals", (PyCFunction)polar_normals, METH_VARARGS | METH_KEYWORDS, polar_normals_docstring },
	{ "map_frame_comp", (PyCFunction)map_frame_comp, METH_VARARGS | METH_KEYWORDS, map_frame_comp_docstring },
	{NULL, NULL, 0, NULL}
};


// Initialize the module
// *********************

static struct PyModuleDef moduledef = 
{
    PyModuleDef_HEAD_INIT,
    "polar_processing",		// m_name
    module_docstring,       // m_doc
    -1,                     // m_size
    module_methods,         // m_methods
    NULL,                   // m_reload
    NULL,                   // m_traverse
    NULL,                   // m_clear
    NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_polar_processing(void)
{
    import_array();
	return PyModule_Create(&moduledef);
}


// Definition of the batch_subsample method
// **********************************

static PyObject* polar_normals(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	PyObject* queries_obj = NULL;

	// Keywords containers
	static char* kwlist[] = { "points", "radius", "lidar_n_lines", "h_scale", "r_scale", "verbose", NULL };
	float radius = 1.5;
	int lidar_n_lines = 32;
	float h_scale = 0.5f;
	float r_scale = 4.0f;
	int verbose = 0;


	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|$fiffi", kwlist, &queries_obj, &radius, &lidar_n_lines, &h_scale, &r_scale, &verbose))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	// Interpret the input objects as numpy arrays.
	PyObject* queries_array = PyArray_FROM_OTF(queries_obj, NPY_FLOAT, NPY_IN_ARRAY);

	// Verify data was load correctly.
	if (queries_array == NULL)
	{
		Py_XDECREF(queries_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting points to numpy arrays of type float32");
		return NULL;
	}
	// Check that the input array respect the dims
	if ((int)PyArray_NDIM(queries_array) != 2 || (int)PyArray_DIM(queries_array, 1) != 3)
	{
		Py_XDECREF(queries_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : points.shape is not (N, 3)");
		return NULL;
	}

	// Number of points
	int N = (int)PyArray_DIM(queries_array, 0);

	// Call the C++ function
	// *********************

	// Convert PyArray to Cloud C++ class
	vector<PointXYZ> queries;
	queries = vector<PointXYZ>((PointXYZ*)PyArray_DATA(queries_array), (PointXYZ*)PyArray_DATA(queries_array) + N);

	// Create result containers
	vector<PointXYZ> normals;
	vector<float> planarity;
	vector<float> linearity;

	// Compute results
	extract_features_multi_thread(queries, normals, planarity, linearity, lidar_n_lines, h_scale, r_scale, verbose);

	// Manage outputs
	// **************

	// Dimension of input containers
	npy_intp* normals_dims = new npy_intp[2];
	normals_dims[0] = normals.size();
	normals_dims[1] = 3;
	npy_intp* scores_dims = new npy_intp[1];
	scores_dims[0] = planarity.size();

	// Create output array
	PyObject* res_normals_obj = PyArray_SimpleNew(2, normals_dims, NPY_FLOAT);
	PyObject* res_plan_obj = PyArray_SimpleNew(1, scores_dims, NPY_FLOAT);
	PyObject* res_lin_obj = PyArray_SimpleNew(1, scores_dims, NPY_FLOAT);

	// Fill normals array with values
	size_t size_in_bytes = normals.size() * 3 * sizeof(float);
	memcpy(PyArray_DATA(res_normals_obj), normals.data(), size_in_bytes);

	// Fill scores array with values
	size_t size_in_bytes2 = planarity.size() * sizeof(float);
	memcpy(PyArray_DATA(res_plan_obj), planarity.data(), size_in_bytes2);
	memcpy(PyArray_DATA(res_lin_obj), linearity.data(), size_in_bytes2);

	// Merge results
	PyObject* ret = Py_BuildValue("NNN", res_normals_obj, res_plan_obj, res_lin_obj);

	// Clean up
	// ********

	Py_XDECREF(queries_array);

	return ret;
}


static PyObject* map_frame_comp(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	float map_dl = 0.1;
	float theta_dl = 0.1;
	float phi_dl = 1.0;
	bool motion_distortion = false;
	char* fnames_str;
	PyObject* map_p_obj = NULL;
	PyObject* map_n_obj = NULL;
	PyObject* H_obj = NULL;

	// Keywords containers
	static char* kwlist[] = { "frame_names", "map_points",  "map_normals", "H_frames", "map_dl", "theta_dl", "phi_dl", NULL };

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "zOOO|$fff", kwlist, 
		&fnames_str,
		&map_p_obj, 
		&map_n_obj, 
		&H_obj, 
		&map_dl, 
		&theta_dl, 
		&phi_dl))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	// Interpret the input objects as numpy arrays.
	PyObject* map_p_array = PyArray_FROM_OTF(map_p_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* map_n_array = PyArray_FROM_OTF(map_n_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* H_array = PyArray_FROM_OTF(H_obj, NPY_DOUBLE, NPY_IN_ARRAY);

	// Data verification
	// *****************

	vector<bool> conditions;
	vector<string> error_messages;

	// Verify data was load correctly.
	conditions.push_back(map_p_array == NULL);
	error_messages.push_back("Error converting map points to numpy arrays of type float32");
	conditions.push_back(map_n_array == NULL);
	error_messages.push_back("Error converting map normals to numpy arrays of type float32");
	conditions.push_back(H_array == NULL);
	error_messages.push_back("Error converting R to numpy arrays of type double");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(map_p_array);
			Py_XDECREF(map_n_array);
			Py_XDECREF(H_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Check that the input array respect the dims
	conditions.push_back((int)PyArray_NDIM(map_p_array) != 2 || (int)PyArray_DIM(map_p_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : map_points.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(map_n_array) != 2 || (int)PyArray_DIM(map_n_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : map_normals.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(H_array) != 3 || (int)PyArray_DIM(H_array, 1) != 4 || (int)PyArray_DIM(H_array, 2) != 4);
	error_messages.push_back("Error, wrong dimensions : R.shape is not (N, 4, 4)");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(map_p_array);
			Py_XDECREF(map_n_array);
			Py_XDECREF(H_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Check number of points
	size_t Nm = (size_t)PyArray_DIM(map_p_array, 0);
	size_t N_frames = (size_t)PyArray_DIM(H_array, 0);


	// Init variables
	// **************

	// Convert frame names to string
	string frame_names(fnames_str);

	// Convert PyArray to Cloud C++ class
	vector<PointXYZ> map_points, map_normals;
	map_points = vector<PointXYZ>((PointXYZ*)PyArray_DATA(map_p_array), (PointXYZ*)PyArray_DATA(map_p_array) + Nm);
	map_normals = vector<PointXYZ>((PointXYZ*)PyArray_DATA(map_n_array), (PointXYZ*)PyArray_DATA(map_n_array) + Nm);

	// Convert H to Eigen matrices
	vector<double> H_vec((double*)PyArray_DATA(H_array), (double*)PyArray_DATA(H_array) + N_frames * 4 * 4);
	Eigen::Map<Eigen::MatrixXd> all_H_t((double*)H_vec.data(), 4, N_frames * 4);
	Eigen::MatrixXd all_H = all_H_t.transpose();


	// Init point map
	// **************
	
	// Create the pointmap voxels
	unordered_map<VoxKey, size_t> map_samples;
	map_samples.reserve(map_points.size());
	float inv_map_dl = 1.0 / map_dl;
	VoxKey k0;
	size_t p_i = 0;

	for (auto &p : map_points)
	{

		//cout << p << endl;

		// Corresponding key
		k0.x = (int)floor(p.x * inv_map_dl);
		k0.y = (int)floor(p.y * inv_map_dl);
		k0.z = (int)floor(p.z * inv_map_dl);

		//cout << k0.x << ", " << k0.y << ", " << k0.z << endl;

		// Update the sample map
		if (map_samples.count(k0) < 1)
		{
			map_samples.emplace(k0, p_i);
		}
		else
		{
			int a;
			//cout << "WARNING: multiple points in a single map voxel" << endl;
			//return NULL;
		}
			
		p_i++;
	}
	//cout << "++++++++++++++++++++++++++++++++++++++" << endl;

	// Init map movable probabilities and counts
	vector<float> movable_probs(map_points.size(), 0);
	vector<int> movable_counts(map_points.size(), 0);


	// Start movable detection
	// ***********************

	// Loop on the lines of "frame_names" string
	int verbose = 1;
	istringstream iss(frame_names);
	size_t frame_i = 0;
	clock_t t0 = std::clock();
	for (string line; getline(iss, line);)
	{

		// Load frame
		// **********

		// Load ply file
		vector<PointXYZ> f_pts;
		load_cloud(line, f_pts);

		// Get the corresponding pose
		Eigen::Matrix3d R = all_H.block(frame_i * 4, 0, 3, 3);
		Eigen::Vector3d T = all_H.block(frame_i * 4, 3, 3, 1);

		// Handle motion distortion by slices
		if (motion_distortion)
		{
			for (int s = 0; s < 12; s++)
			{
				Eigen::Matrix3d slice_R;
				Eigen::Vector3d slice_T;
				vector<PointXYZ> slice_pts;

				// DO STUFF
			}
		}
		else
		{
			// Compute results
			compare_map_to_frame(f_pts, map_points, map_normals, map_samples, R, T, theta_dl, phi_dl, map_dl, movable_probs, movable_counts);


		}
		frame_i++;

		if (verbose > 0)
		{
			clock_t t1 = std::clock();
			double duration = 1000 * (t1 - t0) / (double)CLOCKS_PER_SEC;
			cout << "Annotation step " << frame_i << "/" << N_frames << " done in " << duration << " ms " << endl;
			t0 = t1;
		}
	}

	// Manage outputs
	// **************

	// Dimension of input containers
	npy_intp* res_dims = new npy_intp[1];
	res_dims[0] = movable_probs.size();

	// Create output array
	PyObject* movable_probs_obj = PyArray_SimpleNew(1, res_dims, NPY_FLOAT);
	PyObject* movable_counts_obj = PyArray_SimpleNew(1, res_dims, NPY_INT);

	// Fill normals array with values
	size_t size_in_bytes = movable_probs.size() * sizeof(float);
	memcpy(PyArray_DATA(movable_probs_obj), movable_probs.data(), size_in_bytes);

	// Fill scores array with values
	size_t size_in_bytes2 = movable_counts.size() * sizeof(int);
	memcpy(PyArray_DATA(movable_counts_obj), movable_counts.data(), size_in_bytes2);

	// Merge results
	PyObject* ret = Py_BuildValue("NN", movable_probs_obj, movable_counts_obj);

	// Clean up
	// ********

	Py_XDECREF(map_p_array);
	Py_XDECREF(map_n_array);
	Py_XDECREF(H_array);

	return ret;
}
