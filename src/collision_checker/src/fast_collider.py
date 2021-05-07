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
from fast_collider_data import MyhalCollisionCollate, OnlineDataset, OnlineSampler
from scipy import ndimage

# ROS
import rospy
import ros_numpy
import rosgraph
from ros_numpy import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from collision_checker.msg import VoxGrid
import tf2_ros
from tf2_msgs.msg import TFMessage

# for pausing gazebo during computation:
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan




class OnlineCollider():

    def __init__(self):

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
            
        ############
        # Init ROS #
        ############

        self.obstacle_range = 2.0

        rospy.init_node('fast_collider', anonymous=True)

        ######################
        # Load trained model #
        ######################

        print('\nModel Preparation')
        print('*****************')
        t1 = time.time()

        # Choose which training checkpoints to use
        log_name = rospy.get_param('/log_name')
        training_path = join(ENV_HOME, 'catkin_ws/src/collision_trainer/results', log_name)
        chkp_name = rospy.get_param('/chkp_name')
        chkp_path = os.path.join(training_path, 'checkpoints', chkp_name)

        # Load configuration class used at training
        self.config = Config()
        self.config.load(training_path)

        # Init data class
        self.online_dataset = OnlineDataset(self.config)
        self.online_sampler = OnlineSampler(self.online_dataset)
        self.online_loader = DataLoader(self.online_dataset,
                                        batch_size=1,
                                        sampler=self.online_sampler,
                                        collate_fn=MyhalCollisionCollate,
                                        num_workers=1,
                                        pin_memory=True)

        # Define network model
        self.net = KPCollider(self.config, self.online_dataset.label_values, self.online_dataset.ignored_labels)
        self.net.to(self.device)
        self.softmax = torch.nn.Softmax(1)
        self.sigmoid_2D = torch.nn.Sigmoid()

        # Load the pretrained weigths
        if on_gpu and torch.cuda.is_available():
            checkpoint = torch.load(chkp_path, map_location=self.device)
        else:
            checkpoint = torch.load(chkp_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(checkpoint['model_state_dict'])

        # Switch network from training to evaluation mode
        self.net.eval()

        print("\nModel and training state restored from " + chkp_path)
        print('Done in {:.1f}s\n'.format(time.time() - t1))

        ###############
        # ROS sub/pub #
        ###############

        # Subscribe to the lidar topic
        print('\nSubscribe to /velodyne_points')
        rospy.Subscriber("/velodyne_points", PointCloud2, self.lidar_callback)
        print('OK\n')

        # Subsrcibe
        print('\nSubscribe to tf messages')
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        rospy.Subscriber("/tf", TFMessage, self.tf_callback)
        print('OK\n')

        # Init collision publisher
        self.collision_pub = rospy.Publisher('/plan_costmap_3D', VoxGrid, queue_size=10)
        self.visu_pub = rospy.Publisher('/collision_visu', OccupancyGrid, queue_size=10)
        self.time_resolution = self.config.T_2D / self.config.n_2D_layers

        # # obtaining/defining the parameters for the output of the laserscan data
        # try:
        #     self.gmapping_status = rospy.get_param('gmapping_status')
        # except KeyError:
        #     self.gmapping_status = True
        
        # self.template_scan = LaserScan()

        # self.template_scan.angle_max = np.pi
        # self.template_scan.angle_min = -np.pi
        # self.template_scan.angle_increment = 0.01
        # self.template_scan.time_increment = 0.0
        # self.template_scan.scan_time = 0.01
        # self.template_scan.range_min = 0.0
        # self.template_scan.range_max = 30.0
        # self.min_height = 0.01
        # self.max_height = 1
        # self.ranges_size = int(np.ceil((self.template_scan.angle_max - self.template_scan.angle_min)/self.template_scan.angle_increment))

                                                                                    
        # self.pub_funcs = []
        # if (PUBLISH_POINTCLOUD):
        #     self.pub = rospy.Publisher('/classified_points', PointCloud2, queue_size=10)
        #     self.pub_funcs.append(self.publish_as_pointcloud)
        # if (PUBLISH_LASERSCAN):
        #     # scan for local planner (the first element of the tuple denotes the classes alloted to that scan)
        #     self.pub_list = [([0, 1, 2, 3, 4], rospy.Publisher('/local_planner_points2', LaserScan, queue_size=10))]
        #     self.pub_funcs.append(self.publish_as_laserscan)
        #     if (self.gmapping_status):
        #         self.pub_list.append(([0, 2, 3], rospy.Publisher('/gmapping_points2', LaserScan, queue_size=10)))  # scan for gmapping
        #     else:
        #         self.pub_list.append(([2], rospy.Publisher('/amcl_points2', LaserScan, queue_size=10)))  # scan for amcl localization
        #         self.pub_list.append(([0, 2, 3], rospy.Publisher('/global_planner_points2', LaserScan, queue_size=10)))  # scan for global planner

        return


    def tf_callback(self, msg):

        frame_recieved = False
        for transform in msg.transforms:
            if transform.header.frame_id == 'map' and transform.child_frame_id == 'odom':
                frame_recieved = True


        if frame_recieved:

            # Get the time stamps for all frames
            for f_i, data in enumerate(self.online_dataset.frame_queue):

                if self.online_dataset.pose_queue[f_i] is None:
                    pose = None
                    look_i = 0
                    while pose is None and (rospy.get_rostime() < data[0] + rospy.Duration(1.0)) and look_i < 50:
                        try:
                            pose = self.tfBuffer.lookup_transform('map', 'velodyne', data[0])
                            print('{:^35s}'.format('stamp {:.3f} read at {:.3f}'.format(pose.header.stamp.to_sec(), rospy.get_rostime().to_sec())), 35*' ', 35*' ')
                        except (tf2_ros.InvalidArgumentException, tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                            #print(e)
                            #print(rospy.get_rostime(), data[0] + rospy.Duration(1.0))
                            time.sleep(0.001)
                            pass
                        look_i += 1

                    with self.online_dataset.worker_lock:
                        self.online_dataset.pose_queue[f_i] = pose
        return


    def lidar_callback(self, cloud):

        #t1 = time.time()
        #print('Starting lidar_callback after {:.1f}'.format(1000*(t1 - self.last_t)))

        # convert PointCloud2 message to structured numpy array
        labeled_points = pc2.pointcloud2_to_array(cloud)

        # convert numpy array to Nx3 sized numpy array of float32
        xyz_points = pc2.get_xyz_points(labeled_points, remove_nans=True, dtype=np.float32)

        # Safe check
        if xyz_points.shape[0] < 100:
            print('{:^35s}'.format('CPU 0 : Corrupted frame not added'), 35*' ', 35*' ')
            return

        # # Convert to torch tensor and share memory
        # xyz_tensor = torch.from_numpy(xyz_points)
        # xyz_tensor.share_memory_()

        # # Convert header info to shared tensor
        # frame_t = torch.from_numpy(np.array([cloud.header.stamp.to_sec()], dtype=np.float64))
        # frame_t.share_memory_()
            
        # Update the frame list
        with self.online_dataset.worker_lock:

            if len(self.online_dataset.frame_queue) >= self.config.n_frames:
                self.online_dataset.frame_queue.pop(0)
                self.online_dataset.pose_queue.pop(0)
            self.online_dataset.frame_queue.append((cloud.header.stamp, xyz_points))
            self.online_dataset.pose_queue.append(None)

        print('{:^35s}'.format('CPU 0 : New frame added'), 35*' ', 35*' ')

        #t2 = time.time()
        #print('Finished lidar_callback in {:.1f}'.format(1000*(t2 - t1)))
        #self.last_t = time.time()

        return


    def inference_loop(self):

        # No gradient computation here
        with torch.no_grad():

            # When starting this for loop, one thread will be spawned and creating network 
            # input while the loop is performing GPU operations.
            for i, batch in enumerate(self.online_loader):

                #####################
                # Input preparation #
                #####################

                t = [time.time()]

                # Check that ros master is still up
                try:
                    topics = rospy.get_published_topics()
                except ConnectionRefusedError as e:
                    print('Lost connection to master. Terminate collider')
                    break

                # Check if batch is a dummy
                if len(batch.points) < 1:
                    print(35*' ', 35*' ', '{:^35s}'.format('GPU : Corrupted batch skipped'))
                    time.sleep(0.5)
                    continue
                
                print(35*' ', 35*' ', '{:^35s}'.format('GPU : Got batch, start inference'))

                # Convert batch to a cuda tensors
                if 'cuda' in self.device.type:
                    batch.to(self.device)
                torch.cuda.synchronize(self.device)

                t += [time.time()]
                    
                #####################
                # Network inference #
                #####################

                # Forward pass
                outputs, preds_init, preds_future = self.net(batch, self.config)
                torch.cuda.synchronize(self.device)
                
                t += [time.time()]

                ###########
                # Outputs #
                ###########
                    
                # Get future collisions [1, T, W, H, 3] (only one element in the batch)
                stck_future_preds = self.sigmoid_2D(preds_future).cpu().detach().numpy()
                stck_future_preds = stck_future_preds[0, :, :, :, :]

                # Use walls and obstacles from init preds
                # stck_init_preds = sigmoid_2D(preds_init).cpu().detach().numpy()
                # stck_init_preds = stck_init_preds[0, 0, :, :, :]
                # stck_future_preds[:, :, :, :2] = np.expand_dims(stck_init_preds[:, :, :2], 0)
                
                # Diffuse the collision risk to create a simili-signed-distance-function
                fixed_obstacles = np.sum(stck_future_preds[:, :, :, :3], axis=-1) < 0.5
                fixed_obstacles_dist = []
                for i in range(fixed_obstacles.shape[0]):
                    slice_dist = ndimage.distance_transform_edt(fixed_obstacles[i])
                    fixed_obstacles_dist.append(slice_dist)
                fixed_obstacles_dist = np.stack(fixed_obstacles_dist, 0)

                fixed_obstacles_func = np.clip(1.0 - fixed_obstacles_dist * self.config.dl_2D / self.obstacle_range, 0, 1)

                # Convert to uint8 for message 0-254 = prob, 255 = fixed obstacle
                fixed_preds = (fixed_obstacles_func * 255).astype(np.uint8)
                moving_preds = (stck_future_preds[:, :, :, 2] * 255).astype(np.uint8)
                collision_preds = np.maximum(fixed_preds, moving_preds)


                # Publish collision in a custom message
                self.publish_collisions(collision_preds, batch.t0, batch.p0, batch.q0)
                self.publish_collisions_visu(collision_preds, batch.t0, batch.p0, batch.q0)

                t += [time.time()]


                # Fake slowing pause
                time.sleep(2.5)
                print(35*' ', 35*' ', '{:^35s}'.format('GPU : Inference Done'))


        return

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
 
    def publish_collisions(self, collision_preds, t0, p0, q0):

        # Get origin and orientation
        origin0 = p0 - self.config.in_radius / np.sqrt(2)

        # Define header
        msg = VoxGrid()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = 'map'

        # Define message
        msg.depth = collision_preds.shape[0]
        msg.width = collision_preds.shape[1]
        msg.height = collision_preds.shape[2]
        msg.dl = self.config.dl_2D
        msg.dt = self.time_resolution
        msg.origin.x = origin0[0]
        msg.origin.y = origin0[1]
        msg.origin.z = t0

        #msg.theta = q0[0]
        msg.theta = 0

        msg.data = collision_preds.ravel().tolist()


        # Publish
        self.collision_pub.publish(msg)

        return

    def publish_collisions_visu(self, collision_preds, t0, p0, q0):
        '''
        0 = invisible
        1 -> 98 = blue to red
        99 = cyan
        100 = yellow
        101 -> 127 = green
        128 -> 254 = red to yellow
        255 = vert/gris
        '''

        # Get origin and orientation
        origin0 = p0 - self.config.in_radius / np.sqrt(2)

        # Define header
        msg = OccupancyGrid()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = 'map'


        # Define message meta data
        msg.info.map_load_time = rospy.Time.from_sec(t0)
        msg.info.resolution = self.config.dl_2D
        msg.info.width = collision_preds.shape[1]
        msg.info.height = collision_preds.shape[2]
        msg.info.origin.position.x = origin0[0]
        msg.info.origin.position.y = origin0[1]
        msg.info.origin.position.z = 0.1
        #msg.info.origin.orientation.x = q0[0]
        #msg.info.origin.orientation.y = q0[1]
        #msg.info.origin.orientation.z = q0[2]
        #msg.info.origin.orientation.w = q0[3]


        # Define message data
        T = 15
        data_array = collision_preds[T, :, :].astype(np.float32)
        mask = collision_preds[T, :, :] > 254
        mask2 = np.logical_not(mask)
        data_array[mask2] = data_array[mask2] * 98 / 254
        data_array[mask2] = np.maximum(1, np.minimum(98, data_array[mask2] * 2.0))
        data_array[mask] = 101
        data_array = data_array.astype(np.int8)
        msg.data = data_array.ravel()

        # Publish
        self.visu_pub.publish(msg)

        return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':
    #mp.set_start_method('spawn')

    # q = mp.Queue()

    # queue_loader_proc = multiprocessing.Process(name='queue_loader', target=queue_loader, args=(q,))
    # #d.daemon = True

    # net_proc = multiprocessing.Process(name='net_inf', target=net_inf, args=(q,))
    # #n.daemon = False

    # queue_loader_proc.start()
    # print(q.get())    # prints "[42, None, 'hello']"
    # p.join()


    # a = 1/0

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


    # Setup the collider Class
    print('\n\n\n\n        ------ Init Collider ------')
    tester = OnlineCollider()
    print('OK')

    # Start network process
    print('Start inference loop')
    tester.inference_loop()
    print('OK')

