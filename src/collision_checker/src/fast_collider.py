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
import numpy as np

# Useful classes
from utils.config import Config
from utils.ply import read_ply, write_ply
from models.architectures import KPCollider
from kernels.kernel_points import create_3D_rotations

# ROS
import rospy
import ros_numpy
import rosgraph
from ros_numpy import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from collision_checker.msg import VoxGrid

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


class OnlineData:

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

    def build_segmentation_input_list(self, config, stacked_points, stack_lengths):

        # Starting radius of convolutions
        r_normal = config.first_subsampling_dl * config.conv_radius

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

        arch = config.architecture

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
                    r = r_normal * config.deform_radius / config.conv_radius
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
                dl = 2 * r_normal / config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * config.deform_radius / config.conv_radius
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


class OnlineBatch:
    """Custom batch definition for online frame processing"""

    def __init__(self, frame_points, config, data_handler):
        """
        Function creating the batch structure from frame points.
        :param frame_points: matrix of the frame points
        :param config: Configuration class
        :param data_handler: Data handling class
        """

        # TODO: Speed up this CPU preprocessing
        #           > Use OMP for neighbors processing
        #           > Use the polar coordinates to get neighbors???? (avoiding tree building time)

        # First subsample points

        in_pts = grid_subsampling(frame_points, sampleDl=config.first_subsampling_dl)

        # Randomly drop some points (safety for GPU memory consumption)
        if in_pts.shape[0] > config.max_val_points:
            input_inds = np.random.choice(in_pts.shape[0], size=config.max_val_points, replace=False)
            in_pts = in_pts[input_inds, :]

        # Length of the point list (here not really useful but the network need that value)
        in_lengths = np.array([in_pts.shape[0]], dtype=np.int32)

        # Features the network was trained with
        in_features = np.ones_like(in_pts[:, :1], dtype=np.float32)
        if config.in_features_dim == 1:
            pass
        elif config.in_features_dim == 2:
            # Use height coordinate
            in_features = np.hstack((in_features, in_pts[:, 2:3]))
        elif config.in_features_dim == 4:
            # Use all coordinates
            in_features = np.hstack((in_features, in_pts[:3]))

        # Get the whole input list
        input_list = data_handler.build_segmentation_input_list(config, in_pts, in_lengths)

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+config.num_layers]]
        ind += config.num_layers
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+config.num_layers]]
        ind += config.num_layers
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+config.num_layers]]
        ind += config.num_layers
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+config.num_layers]]
        ind += config.num_layers
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+config.num_layers]]
        ind += config.num_layers
        self.features = torch.from_numpy(in_features)

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
        self.features = self.features.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)

        return self


class OnlineCollider:

    def __init__(self, in_topic):

        ####################
        # Init environment #
        ####################

        # Set which gpu is going to be used
        on_gpu = True
        GPU_ID = rospy.get_param('/gpu_id')

        # Set GPU visible device
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

        # Get the GPU for PyTorch
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        ######################
        # Load trained model #
        ######################

        print('\nModel Preparation')
        print('*****************')
        t1 = time.time()

        # Choose which training checkpoints to use
        log_name = rospy.get_param('/log_name')
        training_path = join('../../collision_trainer/results', log_name)
        
        chkp_name = rospy.get_param('/chkp_name')
        chkp_path = os.path.join(training_path, 'checkpoints', chkp_name)

        # Load configuration class used at training
        self.config = Config()
        self.config.load(training_path)

        # Init data class
        self.data_handler = OnlineData(self.config)

        # Define network model
        self.net = KPCollider(self.config, self.data_handler.label_values, self.data_handler.ignored_labels)
        self.net.to(self.device)
        self.softmax = torch.nn.Softmax(1)
        self.sigmoid_2D = torch.nn.Sigmoid()

        # Load the pretrained weigths
        if on_gpu and torch.cuda.is_available():
            checkpoint = torch.load(chkp_path)
        else:
            checkpoint = torch.load(chkp_path, map_location={'cuda:0': 'cpu'})
        self.net.load_state_dict(checkpoint['model_state_dict'])

        # Switch network from training to evaluation mode
        self.net.eval()

        print("\nModel and training state restored from " + chkp_path)
        print('Done in {:.1f}s\n'.format(time.time() - t1))

        ############
        # Init ROS #
        ############

        rospy.init_node('fast_collider', anonymous=True)
        # obtaining/defining the parameters for the output of the laserscan data
        try:
            self.gmapping_status = rospy.get_param('gmapping_status')
        except KeyError:
            self.gmapping_status = True
        
        self.template_scan = LaserScan()

        self.template_scan.angle_max = np.pi
        self.template_scan.angle_min = -np.pi
        self.template_scan.angle_increment = 0.01
        self.template_scan.time_increment = 0.0
        self.template_scan.scan_time = 0.01
        self.template_scan.range_min = 0.0
        self.template_scan.range_max = 30.0
        self.min_height = 0.01
        self.max_height = 1
        self.ranges_size = int(np.ceil((self.template_scan.angle_max - self.template_scan.angle_min)/self.template_scan.angle_increment))

        self.collision_pub = rospy.Publisher('/collision_preds', VoxGrid, queue_size=10)
        self.time_resolution = self.config.T_2D / self.config.n_2D_layers
        #self.tf_listener = tf.TransformListener()
                                                                                    
        self.pub_funcs = []
        if (PUBLISH_POINTCLOUD):
            self.pub = rospy.Publisher('/classified_points', PointCloud2, queue_size=10)
            self.pub_funcs.append(self.publish_as_pointcloud)
        if (PUBLISH_LASERSCAN):
            # scan for local planner (the first element of the tuple denotes the classes alloted to that scan)
            self.pub_list = [([0, 1, 2, 3, 4], rospy.Publisher('/local_planner_points2', LaserScan, queue_size=10))]
            self.pub_funcs.append(self.publish_as_laserscan)
            if (self.gmapping_status):
                self.pub_list.append(([0, 2, 3], rospy.Publisher('/gmapping_points2', LaserScan, queue_size=10)))  # scan for gmapping
            else:
                self.pub_list.append(([2], rospy.Publisher('/amcl_points2', LaserScan, queue_size=10)))  # scan for amcl localization
                self.pub_list.append(([0, 2, 3], rospy.Publisher('/global_planner_points2', LaserScan, queue_size=10)))  # scan for global planner

        rospy.Subscriber(in_topic, PointCloud2, self.lidar_callback)
        rospy.spin()

    def network_inference(self, points):
        """
        Function simulating a network inference.
        :param points: The input list of points as a numpy array (type float32, size [N,3])
        :return: predictions : The output of the network. Class for each point as a numpy array (type int32, size [N])
        """

        # Ensure no gradient is computed
        with torch.no_grad():

            #####################
            # Input preparation #
            #####################

            # t = [time.time()]

            # Create batch from the frame points
            batch = OnlineBatch(points, self.config, self.data_handler)

            # t += [time.time()]

            # Convert batch to a cuda
            batch.to(self.device)
            # t += [time.time()]
            torch.cuda.synchronize(self.device)

            #####################
            # Network inference #
            #####################

            # Forward pass
            outputs, preds_init, preds_future = self.net(batch, self.config)
            torch.cuda.synchronize(self.device)
            # t += [time.time()]

            # Get probs and labels
            predicted_point_probs = self.softmax(outputs).cpu().detach().numpy()
            torch.cuda.synchronize(self.device)
            # t += [time.time()]

            ##################
            # Handle outputs #
            ##################

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(self.data_handler.label_values):
                if label_value in self.data_handler.ignored_labels:
                    predicted_point_probs = np.insert(predicted_point_probs, l_ind, 0, axis=1)

            # Get predicted labels
            point_predictions = self.data_handler.label_values[np.argmax(predicted_point_probs, axis=1)].astype(np.int32)
            # t += [time.time()]

            # Get future collisions [1, T, W, H, 3] (only one element in the batch)
            stck_future_preds = self.sigmoid_2D(preds_future).cpu().detach().numpy()
            stck_future_preds = stck_future_preds[0, :, :, :, :]

            # Use walls and obstacles from init preds
            # stck_init_preds = sigmoid_2D(preds_init).cpu().detach().numpy()
            # stck_init_preds = stck_init_preds[0, 0, :, :, :]
            # stck_future_preds[:, :, :, :2] = np.expand_dims(stck_init_preds[:, :, :2], 0)

            # Convert to uint8 for message 0-254 = prob, 255 = fixed obstacle
            fixed_preds = np.logical_or(stck_future_preds[:, :, :, :2] > 0.5, axis=-1).astype(np.uint8) * 255
            moving_preds = (stck_future_preds[:, :, :, 2] * 255).astype(np.uint8)
            collision_preds = np.maximum(fixed_preds, moving_preds)

            # print('\n************************\n')
            # print('Timings:')
            # i = 0
            # print('Batch ...... {:7.1f} ms'.format(1000*(t[i+1] - t[i])))
            # i += 1
            # print('ToGPU ...... {:7.1f} ms'.format(1000*(t[i+1] - t[i])))
            # i += 1
            # print('Forward .... {:7.1f} ms'.format(1000*(t[i+1] - t[i])))
            # i += 1
            # print('Softmax .... {:7.1f} ms'.format(1000*(t[i+1] - t[i])))
            # i += 1
            # print('Preds ...... {:7.1f} ms'.format(1000*(t[i+1] - t[i])))
            # print('-----------------------')
            # print('TOTAL  ..... {:7.1f} ms'.format(1000*(t[-1] - t[0])))
            # print('\n************************\n')

            return point_predictions, batch.points[0].cpu().numpy(), collision_preds

    def lidar_callback(self, cloud):
        
        #pause the simulation
        if (PAUSE_SIM):
            pause.call()
        rospy.loginfo("Received Point Cloud")
        
        t1 = time.time()

        # convert PointCloud2 message to structured numpy array
        labeled_points = pc2.pointcloud2_to_array(cloud)

        # convert numpy array to Nx3 sized numpy array of float32
        xyz_points = pc2.get_xyz_points(labeled_points, remove_nans=True, dtype=np.float32)

        # obtain 1xN numpy array of predictions and Nx3 numpy array of sampled points
        predictions, new_points, collision_preds = self.network_inference(xyz_points)

        self.publish_collisions(collision_preds, cloud)

        for func in self.pub_funcs:
            func(new_points, predictions, cloud)
        t2 = time.time()
        print("Computation time: {:.3f}s\n".format(t2-t1))
        rospy.loginfo("Sent Pointcloud")

        #unpause the simulation
        if (PAUSE_SIM):
            unpause.call()

    def publish_as_laserscan(self, cloud_arr, predictions, original_msg):
        t1 = time.time()

        # stores the laser scan output messages (depending on the filtering we are using), where the index in outputs
        # corresponds to the publisher at self.pub_list[i][1]
        outputs = []
        for k in self.pub_list:
            scan = self.template_scan
            scan.ranges = self.ranges_size * [np.inf]
            scan.header = original_msg.header
            scan.header.frame_id = "base_link"
            outputs.append(scan)

        # Here we assume cloud_arr is a numpy array of shape (N, 3)

        # First remove any NaN value form the array
        valid_mask = np.logical_not(np.any(np.isnan(cloud_arr), axis=1))
        cloud_arr = cloud_arr[valid_mask, :]
        predictions = predictions[valid_mask]

        # Compute 2d polar coordinate
        r_arr = np.sqrt(np.sum(cloud_arr[:, :2] ** 2, axis=1))
        angle_arr = np.arctan2(cloud_arr[:, 1], cloud_arr[:, 0])

        # Then remove points according to height/range/angle limits
        valid_mask = cloud_arr[:, 2] > self.min_height
        valid_mask = np.logical_and(valid_mask, cloud_arr[:, 2] < self.max_height)
        valid_mask = np.logical_and(valid_mask, r_arr > self.template_scan.range_min ** 2)
        valid_mask = np.logical_and(valid_mask, r_arr < self.template_scan.range_max ** 2)
        valid_mask = np.logical_and(valid_mask, angle_arr > self.template_scan.angle_min)
        valid_mask = np.logical_and(valid_mask, angle_arr <= self.template_scan.angle_max)
        angle_arr = angle_arr[valid_mask]
        r_arr = r_arr[valid_mask]
        predictions = predictions[valid_mask]

        # Compute angle index for all remaining points
        indexes = np.floor(((angle_arr - self.template_scan.angle_min) / self.template_scan.angle_increment))
        indexes = indexes.astype(np.int32)

        # Loop over outputs
        for j in range(len(outputs)):

            # Mask of the points of the category we need
            prediction_mask = np.isin(predictions, self.pub_list[j][0])

            # Update ranges only with points of the right category
            np_ranges = np.array(outputs[j].ranges)
            np_ranges[indexes[prediction_mask]] = r_arr[prediction_mask]
            outputs[j].ranges = list(np_ranges)

            # Publish output
            self.pub_list[j][1].publish(outputs[j])

        print("publish as laserscan time: {:.7f}s".format(time.time() - t1))

    def publish_as_pointcloud(self, new_points, predictions, cloud):
        # data structure of binary blob output for PointCloud2 data type

        output_dtype = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring'], 'formats': ['<f4', '<f4', '<f4', '<f4', '<u2'], 'offsets': [0, 4, 8, 16, 20], 'itemsize': 32})

        # fill structured numpy array with points and classes (in the intensity field). Fill ring with zeros to maintain Pointcloud2 structure
        new_points = np.c_[new_points, predictions, np.zeros(len(predictions))]

        new_points = np.core.records.fromarrays(new_points.transpose(), output_dtype)

         
        # convert to Pointcloud2 message and publish
        msg = pc2.array_to_pointcloud2(new_points, cloud.header.stamp, cloud.header.frame_id)
        
        self.pub.publish(msg)
 
    def publish_collisions(self, collision_preds, cloud):

        cloud_origin = np.array([0, 0], dtype=np.float32)
        cloud_quat = np.array([0, 0, 0, 0], dtype=np.float32)

        # Get origin and orientation
        origin = cloud_origin - self.config.in_radius / np.sqrt(2)

        # Define header
        msg = VoxGrid()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = cloud.header.frame_id

        # Define message
        msg.depth = collision_preds.shape[0]
        msg.width = collision_preds.shape[1]
        msg.height = collision_preds.shape[2]
        msg.dl = self.config.dl_2D
        msg.dt = self.time_resolution
        msg.origin.x = origin[0]
        msg.origin.y = origin[1]
        msg.origin.z = cloud.header.stamp
        msg.theta = cloud_quat[0]
        msg.data = collision_preds

        self.collision_pub.publish(msg)
 

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    # setup testing parameters

    try:  # ensure that the service is available
        rosgraph.Master.lookupService('/gazebo/pause_physics')
        PAUSE_SIM = True  # modify this if we want to pause the simulation during computation
        pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    except:
        PAUSE_SIM = False

    PAUSE_SIM = False

    PUBLISH_POINTCLOUD = True
    PUBLISH_LASERSCAN = False

    #########
    # Start #
    #########

    tester = OnlineCollider("/velodyne_points")




