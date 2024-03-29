#!/usr/bin/env python3

#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Path setup
import os
from os.path import exists, join
import sys
ENV_USER = os.getenv('USER')
ENV_HOME = os.getenv('HOME')
sys.path.insert(0, join(ENV_HOME, "catkin_ws/src/ros_numpy"))
sys.path.insert(0, join(ENV_HOME, "catkin_ws/src/collision_checker/src"))
sys.path.insert(0, join(ENV_HOME, "catkin_ws/src/collision_checker/src/utils"))
sys.path.insert(0, join(ENV_HOME, "catkin_ws/src/collision_checker/src/models"))
sys.path.insert(0, join(ENV_HOME, "catkin_ws/src/collision_checker/src/kernels"))
sys.path.insert(0, join(ENV_HOME, "catkin_ws/src/collision_checker/src/cpp_wrappers"))

# Common libs
import torch
import pickle
import time
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np
import torch.multiprocessing as mp

# Useful classes
from utils.config import Config
from utils.ply import read_ply, write_ply
from models.architectures import KPCollider
from kernels.kernel_points import create_3D_rotations
from torch.utils.data import DataLoader, Sampler
from scipy.spatial.transform import Rotation as scipyR

# ROS
import rospy
import ros_numpy
import rosgraph
from ros_numpy import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from collision_checker.msg import VoxGrid
import tf2_ros
from tf2_msgs.msg import TFMessage

# for pausing gazebo during computation:
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)


def batch_grid_subsampling(points, batches_len, features=None, labels=None,
                           sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=False):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Subsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                                batches_len,
                                                                                features=features,
                                                                                classes=labels,
                                                                                sampleDl=sampleDl,
                                                                                max_p=max_p,
                                                                                verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels


def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Online tester class
#       \*************************/
#


class OnlineDataset:

    def __init__(self, config):

        # Dict from labels to names
        self.label_to_names = {0: 'uncertain',
                               1: 'ground',
                               2: 'still',
                               3: 'longT',
                               4: 'shortT'}

        # Initiate a bunch of variables concerning class labels
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.sort([0])

        # Load neighb_limits dictionary
        neighb_lim_file = join(ENV_HOME, 'Data/MyhalSim/neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            raise ValueError('No neighbors limit file found')

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(config.num_layers):

            dl = config.first_subsampling_dl * (2**layer_ind)
            if config.deform_layers[layer_ind]:
                r = dl * config.deform_radius
            else:
                r = dl * config.conv_radius

            key = '{:s}_{:d}_{:d}_{:.3f}_{:.3f}'.format('random',
                                                        config.n_frames,
                                                        config.max_val_points,
                                                        dl,
                                                        r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if len(neighb_limits) == config.num_layers:
            self.neighborhood_limits = neighb_limits
        else:
            raise ValueError('The neighbors limits were not initialized')

        # Setup varaibles for parallelised input queue
        self.manager = mp.Manager()
        self.frame_queue = self.manager.list()
        self.pose_queue = self.manager.list()

        #self.frame_queue = mp.Queue(maxsize=config.n_frames)
        self.config = config
        self.worker_lock = mp.Lock()

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return 0

    def __getitem__(self, batch_i):

        ################
        # Init Variables
        ################

        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        pl2D_list = []
        img_list = []
        f_list = []
        l_list = []
        fi_list = []
        p0_list = []
        s_list = []
        R_list = []
        r_inds_list = []
        r_mask_list = []
        val_labels_list = []
        batch_n = 0

        ############################
        # Get input points and poses
        ############################

        # Get the last three frames returned by the lidar call_back
        current_frames = []
        current_poses = []
        print(35*' ', '{:^35s}'.format('CPU 1 : Waiting for 3 frames'), 35*' ')
        while len(self.frame_queue) < self.config.n_frames:
            time.sleep(0.01)
        print(35*' ', '{:^35s}'.format('CPU 1 : Waiting for not None'), 35*' ')
        while self.pose_queue[-1] is None:
            time.sleep(0.001)
        with self.worker_lock:
            current_frames = list(self.frame_queue)
            current_poses = list(self.pose_queue)

        print(35*' ', '{:^35s}'.format('CPU 1 : OK got 3 frames'), 35*' ')


        # print('------------------------------------')
        # print('Hello you')
        # for f_i, (stamp, xyz) in enumerate(current_frames):
        #     print(stamp.to_sec(), xyz.shape, xyz[18])
        #     print(current_poses[f_i])
        # print('------------------------------------')

        # Safe check
        if np.any([pose is None for pose in current_poses]):
            return self.dummy_batch()

        if np.any([frame_pts.shape[0] < 100 for stamp, frame_pts in current_frames]):
            return self.dummy_batch()

        print(35*' ', '{:^35s}'.format('CPU 1 : Processing batch'), 35*' ')

        t += [time.time()]




    
        #########################
        # Merge n_frames together
        #########################

        # Initiate merged points
        merged_points = np.zeros((0, 3), dtype=np.float32)
        merged_feats = np.zeros((0, self.config.n_frames), dtype=np.float32)
        p_origin = np.zeros((1, 3))
        p0 = np.zeros((0,))
        q0 = np.zeros((0,))
        
        # Loop in inverse order to start with most recent frame
        for f_i, (pose, (stamp, frame_pts)) in enumerate(zip(current_poses[::-1], current_frames[::-1])):

            # Get translation and rotation matrices
            T = np.array([pose.transform.translation.x,
                          pose.transform.translation.y,
                          pose.transform.translation.z])
            q = np.array([pose.transform.rotation.x,
                          pose.transform.rotation.y,
                          pose.transform.rotation.z,
                          pose.transform.rotation.w])
            R = scipyR.from_quat(q).as_matrix()

            # Update p0 for the most recent frame
            if p0.shape[0] < 1:
                p0 = np.copy(T)
                q0 = np.copy(q)

            # Apply tranformation to align input
            aligned_pts = np.dot(frame_pts, R.T) + T
        
            # Eliminate points further than config.val_radius
            mask = np.sum(np.square(aligned_pts - p0), axis=1) < self.config.in_radius**2
            aligned_pts = aligned_pts[mask, :]

            # # Shuffle points
            # mask_inds = np.where(mask)[0].astype(np.int32)
            # rand_order = np.random.permutation(mask_inds)
            # aligned_pts = aligned_pts[rand_order, :3]
            # sem_labels = sem_labels[rand_order]

            # Stack features
            features = np.zeros((aligned_pts.shape[0], self.config.n_frames), dtype=np.float32)
            features[:, f_i] = 1

            # Merge points
            merged_points = np.vstack((merged_points, aligned_pts))
            merged_feats = np.vstack((merged_feats, features))

        t += [time.time()]

        # # DEBUG: Save input frames
        # plyname = join(ENV_HOME, 'catkin_ws/test_frame_{:.3f}.ply'.format(current_frames[0][0].to_sec()))
        # write_ply(plyname,
        #           [merged_points, merged_feats],
        #           ['x', 'y', 'z', 'f1', 'f2', 'f3'])

        #################
        # Subsample input
        #################

        # Then center on p0
        merged_points_c = (merged_points - p0).astype(np.float32)

        # Subsample merged frames
        in_pts, in_fts = grid_subsampling(merged_points_c,
                                          features=merged_feats,
                                          sampleDl=self.config.first_subsampling_dl)

        # # Randomly drop some points (augmentation process and safety for GPU memory consumption)
        # n = in_pts.shape[0]
        # if n > self.max_in_p:
        #     input_inds = np.random.choice(n,
        #                                   size=self.max_in_p,
        #                                   replace=False)
        #     in_pts = in_pts[input_inds, :]
        #     in_fts = in_fts[input_inds, :]
        #     n = input_inds.shape[0]

        t += [time.time()]

        ##########################
        # Compute 3D-2D projection
        ##########################
        # C++ wrappers to get the projections indexes (with shadow pools)
        # Temporarily use the 3D neighbors wrappers

        # Max number of points pooled to a grid cell
        max_2D_pools = 20

        # Project points on 2D plane
        support_pts = np.copy(in_pts)
        support_pts[:, 2] *= 0

        # Create grid
        grid_ticks = np.arange(-self.config.in_radius / np.sqrt(2),
                               self.config.in_radius / np.sqrt(2),
                               self.config.dl_2D)
        xx, yy = np.meshgrid(grid_ticks, grid_ticks)
        L_2D = xx.shape[0]
        pool_points = np.vstack((np.ravel(xx), np.ravel(yy), np.ravel(yy) * 0)).astype(np.float32).T

        # Get pooling indices
        pool_inds = batch_neighbors(pool_points,
                                    support_pts,
                                    [pool_points.shape[0]],
                                    [support_pts.shape[0]],
                                    self.config.dl_2D / np.sqrt(2))

        # Remove excedent => get to shape [L_2D*L_2D, max_2D_pools]
        if pool_inds.shape[1] < max_2D_pools:
            diff_d = max_2D_pools - pool_inds.shape[1]
            pool_inds = np.pad(pool_inds,
                               ((0, 0), (0, diff_d)),
                               'constant',
                               constant_values=support_pts.shape[0])
        else:
            pool_inds = pool_inds[:, :max_2D_pools]

        # Reshape into 2D grid
        pools_2D = np.reshape(pool_inds, (L_2D, L_2D, max_2D_pools)).astype(np.int64)
        
        # # Adjust pools_2D for batch pooling
        # pl2D_list = [pools_2D]
        # p_list = [in_pts]
        # batch_N = np.sum([p.shape[0] for p in p_list])
        # batch_n = 0
        # for b_i, pl2D in enumerate(pl2D_list):
        #     mask = pl2D == p_list[b_i].shape[0]
        #     pl2D[mask] = batch_N
        #     pl2D[np.logical_not(mask)] += batch_n
        #     batch_n += p_list[b_i].shape[0]
        # stacked_pools_2D = np.stack(pl2D_list, axis=0)
        stacked_pools_2D = np.expand_dims(pools_2D, 0)

        t += [time.time()]

        ################
        # Input features
        ################

        # Input features (Use reflectance, input height or all coordinates)
        stacked_features = np.ones_like(in_pts[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 3:
            # Use only the three frame indicators
            stacked_features = in_fts
        elif self.config.in_features_dim == 4:
            # Use the ones + the three frame indicators
            stacked_features = np.hstack((stacked_features, in_fts))
        else:
            raise ValueError('Only accepted input dimensions are 1, 2 and 4 (without and with XYZ)')
            
        t += [time.time()]

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #
        
        stack_lengths = np.array([pp.shape[0] for pp in p_list],
                                 dtype=np.int32)

        # Get the whole input list
        input_list = self.build_segmentation_input_list(in_pts, np.array([in_pts.shape[0]], dtype=np.int32))

        # Add additionnal inputs
        input_list += [stacked_pools_2D]
        input_list += [stacked_features]
        input_list += [p0, q0, current_frames[-1][0].to_sec()]

        # # Fake sleeping time
        # time.sleep(8.0)

        t += [time.time()]

        ###############
        # Debug Timings
        ###############
        
        debugT = True
        if debugT:
            
            print()
            print(35*' ', '{:^35s}'.format('CPU 1 : Timings'), 35*' ')
            ti = 0
            print(35*' ', '{:^35s}'.format('Waiting ....... {:5.1f}ms'.format(1000 * (t[ti + 1] - t[ti]))), 35*' ')
            ti += 1
            print(35*' ', '{:^35s}'.format('Merging ....... {:5.1f}ms'.format(1000 * (t[ti + 1] - t[ti]))), 35*' ')
            ti += 1
            print(35*' ', '{:^35s}'.format('Subsampling ... {:5.1f}ms'.format(1000 * (t[ti + 1] - t[ti]))), 35*' ')
            ti += 1
            print(35*' ', '{:^35s}'.format('2D proj ....... {:5.1f}ms'.format(1000 * (t[ti + 1] - t[ti]))), 35*' ')
            ti += 1
            print(35*' ', '{:^35s}'.format('Features ...... {:5.1f}ms'.format(1000 * (t[ti + 1] - t[ti]))), 35*' ')
            ti += 1
            print(35*' ', '{:^35s}'.format('Inputs ........ {:5.1f}ms'.format(1000 * (t[ti + 1] - t[ti]))), 35*' ')
            print()
            print(35*' ', '{:^35s}'.format('Total ......... {:5.1f}ms'.format(1000 * (t[-1] - t[0]))), 35*' ')

        print(35*' ', '{:^35s}'.format('CPU 1 : Done!'), 35*' ')
        print()

        return [self.config.num_layers] + input_list

    def dummy_batch(self):

        dummy_list = [0]

        for i in range(11):
            dummy_list.append(np.zeros((1,)))

        return dummy_list

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, :self.neighborhood_limits[layer]]
        else:
            return neighbors

    def build_segmentation_input_list(self, stacked_points, stack_lengths):

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors(stacked_points, pool_p, stack_lengths, pool_b, 2 * r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 3), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)
                up_i = np.zeros((0, 1), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            if up_i.shape[0] > 0:
                up_i = self.big_neighborhood_filter(up_i, len(input_points) + 1)

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_upsamples += [up_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths

        return li


class OnlineSampler(Sampler):
    """Sampler for MyhalCollision"""

    def __init__(self, dataset: OnlineDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        # Dummy Generator loop
        # for i in range(10000000):
        #     yield i

        # Generator loop
        while True:
            yield 0

    def __len__(self):
        return 0


class OnlineColliderBatch:
    """Custom batch definition with memory pinning for MyhalCollision"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = int(input_list[0])

        # Extract input tensors from the list of numpy array
        ind = 1
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.pools_2D = torch.from_numpy(input_list[ind])
        ind += 1
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.p0 = input_list[ind]
        ind += 1
        self.q0 = input_list[ind]
        ind += 1
        self.t0 = input_list[ind]

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.pools_2D = self.pools_2D.pin_memory()
        self.features = self.features.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.pools_2D = self.pools_2D.to(device)
        self.features = self.features.to(device)

        return self


def MyhalCollisionCollate(batch_data):
    return OnlineColliderBatch(batch_data)

