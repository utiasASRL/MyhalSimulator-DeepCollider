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
import torch.multiprocessing as max_in_p
from utils.mayavi_visu import zoom_collisions

# Useful classes
from utils.config import Config
from utils.ply import read_ply, write_ply
from models.architectures import KPCollider
from kernels.kernel_points import create_3D_rotations
from torch.utils.data import DataLoader, Sampler
from fast_collider_data import MyhalCollisionCollate, OnlineDataset, OnlineSampler
from scipy import ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as scipyR

# ROS
import rospy
import ros_numpy
import rosgraph
from ros_numpy import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, PolygonStamped, Point32
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from collision_checker.msg import VoxGrid
import tf2_ros
from tf2_msgs.msg import TFMessage
import imageio

# for pausing gazebo during computation:
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan




class OnlineCollider():

    def __init__(self):

        ####################
        # Init environment #
        ####################

        # Set which gpu is going to be used (auto for automatic choice)
        on_gpu = True
        GPU_ID = 'auto'

        # Automatic choice (need pynvml to be installed)
        if GPU_ID == 'auto':
            print('\nSearching a free GPU:')
            for i in range(torch.cuda.device_count()):
                a = torch.cuda.list_gpu_processes(i)
                print(torch.cuda.list_gpu_processes(i))
                a = a.split()
                if a[1] == 'no':
                    GPU_ID = a[0][-1:]

        # Safe check no free GPU
        if GPU_ID == 'auto':
            print('\nNo free GPU found!\n')
            a = 1/0

        else:
            print('\nUsing GPU:', GPU_ID, '\n')

        # Get the GPU for PyTorch
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:{:d}".format(int(GPU_ID)))
        else:
            self.device = torch.device("cpu")
            
        ############
        # Init ROS #
        ############

        self.obstacle_range = 1.9
        self.norm_p = 3
        self.norm_invp = 1 / self.norm_p

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

        # Collision risk diffusion
        k_range = int(np.ceil(self.obstacle_range / self.config.dl_2D))
        k = 2 * k_range + 1
        dist_kernel = np.zeros((k, k))
        for i, vv in enumerate(dist_kernel):
            for j, v in enumerate(vv):
                dist_kernel[i, j] = np.sqrt((i - k_range) ** 2 + (j - k_range) ** 2)
        dist_kernel = np.clip(1.0 - dist_kernel * self.config.dl_2D / self.obstacle_range, 0, 1) ** self.norm_p
        self.fixed_conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k_range, bias=False)
        self.fixed_conv.weight.requires_grad = False
        self.fixed_conv.weight *= 0
        self.fixed_conv.weight += torch.from_numpy(dist_kernel)
        self.fixed_conv.to(self.device)

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
        self.velo_frame_id = 'velodyne'
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

        # Init obstacle publisher
        self.obstacle_pub = rospy.Publisher('/move_base/TebLocalPlannerROS/obstacles', ObstacleArrayMsg, queue_size=10)

        # Init point cloud publisher
        self.pointcloud_pub = rospy.Publisher('/classified_points', PointCloud2, queue_size=10)

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
        self.velo_frame_id = cloud.header.frame_id

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
                outputs_3D, preds_init, preds_future = self.net(batch, self.config)
                torch.cuda.synchronize(self.device)
                
                t += [time.time()]

                ###########
                # Outputs #
                ###########

                # Get collision predictions [1, T, W, H, 3] -> [T, W, H, 3]
                collision_preds = self.sigmoid_2D(preds_future)[0]

                # Get the diffused risk
                diffused_risk, obst_pos = self.get_diffused_risk(collision_preds)

                # Get obstacles in world coordinates
                origin0 = batch.p0 - self.config.in_radius / np.sqrt(2)

                world_obst = []
                for obst_i, pos in enumerate(obst_pos):
                    world_obst.append(origin0[:2] + pos * self.config.dl_2D)

                # Publish collision risk in a custom message
                self.publish_collisions(diffused_risk, batch.t0, batch.p0, batch.q0)
                self.publish_collisions_visu(diffused_risk, batch.t0, batch.p0, batch.q0, visu_T=15)

                # Publish obstacles
                self.publish_obstacles(world_obst, batch.t0, batch.p0, batch.q0)

                ##############
                # Outputs 3D #
                ##############

                # Get predictions
                predicted_probs = self.softmax(outputs_3D).cpu().detach().numpy()
                for l_ind, label_value in enumerate(self.online_dataset.label_values):
                    if label_value in self.online_dataset.ignored_labels:
                        predicted_probs = np.insert(predicted_probs, l_ind, 0, axis=1)
                predictions = self.online_dataset.label_values[np.argmax(predicted_probs, axis=1)].astype(np.int32)

                # Get frame points re-aligned in the velodyne coordinates
                pred_points = batch.points[0].cpu().detach().numpy() + batch.p0

                R0 = scipyR.from_quat(batch.q0).as_matrix()
                pred_points = np.dot(pred_points - batch.p0, R0)
                
                # Publish pointcloud
                self.publish_pointcloud(pred_points, predictions, batch.t0)

                t += [time.time()]

                # Fake slowing pause
                time.sleep(2.5)
                
                t += [time.time()]

                print(35*' ', 35*' ', '{:^35s}'.format('GPU : Inference Done in {:.3f}s (+ {:.3f}s fake)'.format(t[-2] - t[0], t[-1] - t[-2])))


        return

    def get_diffused_risk(self, collision_preds):
                                    
        # # Remove residual preds (hard hysteresis)
        # collision_risk *= (collision_risk > 0.06).type(collision_risk.dtype)
                    
        # Remove residual preds (soft hysteresis)
        lim1 = 0.06
        lim2 = 0.09
        dlim = lim2 - lim1
        mask0 = collision_preds <= lim1
        mask1 = torch.logical_and(collision_preds < lim2, collision_preds > lim1)
        collision_preds[mask0] *= 0
        collision_preds[mask1] *= (1 - ((collision_preds[mask1] - lim2) / dlim) ** 2) ** 2

        # Get risk from static objects, [1, 1, W, H]
        static_preds = torch.unsqueeze(torch.max(collision_preds[:1, :, :, :2], dim=-1)[0], 1)
        #static_preds = (static_risk > 0.3).type(collision_preds.dtype)

        # Normalize risk values between 0 and 1 depending on density
        static_risk = static_preds / (self.fixed_conv(static_preds) + 1e-6)

        # Diffuse the risk from normalized static objects
        diffused_0 = self.fixed_conv(static_risk).cpu().detach().numpy()

        # Repeat for all the future steps [1, 1, W, H] -> [T, W, H]
        diffused_0 = np.squeeze(np.tile(diffused_0, (collision_preds.shape[0], 1, 1, 1)))

        # Diffuse the risk from moving obstacles , [T, 1, W, H] -> [T, W, H]
        moving_risk = torch.unsqueeze(collision_preds[..., 2], 1)
        diffused_1 = np.squeeze(self.fixed_conv(moving_risk).cpu().detach().numpy())
        
        # Inverse power for p-norm
        diffused_0 = np.power(np.maximum(0, diffused_0), self.norm_invp)
        diffused_1 = np.power(np.maximum(0, diffused_1), self.norm_invp)

        # Merge the two risk after rescaling
        diffused_0 *= 1.0 / (np.max(diffused_0) + 1e-6)
        diffused_1 *= 1.0 / (np.max(diffused_1) + 1e-6)
        diffused_risk = np.maximum(diffused_0, diffused_1)

        # Convert to uint8 for message 0-254 = prob, 255 = fixed obstacle
        diffused_risk = np.minimum(diffused_risk * 255, 255).astype(np.uint8)
        
        # # Save walls for debug
        # debug_walls = np.minimum(diffused_risk[10] * 255, 255).astype(np.uint8)
        # cm = plt.get_cmap('viridis')
        # print(batch.t0)
        # print(type(batch.t0))
        # im_name = join(ENV_HOME, 'catkin_ws/src/collision_trainer/results/debug_walls_{:.3f}.png'.format(batch.t0))
        # imageio.imwrite(im_name, zoom_collisions(cm(debug_walls), 5))

        # Get local maxima in moving obstacles
        obst_mask = self.get_local_maxima(diffused_1[15])

        # Create obstacles in walls (one cell over 2 to have arround 1 obstacle every 25 cm)
        static_mask = np.squeeze(static_preds.cpu().detach().numpy() > 0.3)
        static_mask[::2, :] = 0
        static_mask[:, ::2] = 0

        # Merge obstacles
        obst_mask[static_mask] = 1

        # Convert to pixel positions
        obst_pos = self.mask_to_pix(obst_mask)

        return diffused_risk, obst_pos

    def get_local_maxima(self, data, neighborhood_size=5, threshold=0.1):
        
        # Get maxima positions as a mask
        data_max = filters.maximum_filter(data, neighborhood_size)
        max_mask = (data == data_max)

        # Remove maxima if their peak is not higher than threshold in the neighborhood
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        max_mask[diff == 0] = 0

        return max_mask

    def mask_to_pix(self, mask):
        
        # Get positions in world coordinates
        labeled, num_objects = ndimage.label(mask)
        slices = ndimage.find_objects(labeled)
        x, y = [], []

        mask_pos = []
        for dy, dx in slices:

            x_center = (dx.start + dx.stop - 1) / 2
            y_center = (dy.start + dy.stop - 1) / 2
            mask_pos.append(np.array([x_center, y_center], dtype=np.float32))

        return mask_pos

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

    def publish_collisions_visu(self, collision_preds, t0, p0, q0, visu_T=15):
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
        msg.info.origin.position.z = -0.01
        #msg.info.origin.orientation.x = q0[0]
        #msg.info.origin.orientation.y = q0[1]
        #msg.info.origin.orientation.z = q0[2]
        #msg.info.origin.orientation.w = q0[3]


        # Define message data
        data_array = collision_preds[visu_T, :, :].astype(np.float32)
        mask = collision_preds[visu_T, :, :] > 253
        mask2 = np.logical_not(mask)
        data_array[mask2] = data_array[mask2] * 98 / 253
        data_array[mask2] = np.maximum(1, np.minimum(98, data_array[mask2] * 1.0))
        data_array[mask] = 98  # 101
        data_array = data_array.astype(np.int8)
        msg.data = data_array.ravel()

        # Publish
        self.visu_pub.publish(msg)

        return

    def publish_obstacles(self, obstacle_list, t0, p0, q0):


        msg = ObstacleArrayMsg()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        
        # Add point obstacles
        for obst_i, pos in enumerate(obstacle_list):

            obstacle_msg = ObstacleMsg()
            obstacle_msg.id = obst_i
            obstacle_msg.polygon.points = [Point32(x=pos[0], y=pos[1], z=0)]

            # obstacle_msg.polygon.points[0].x = 1.5
            # obstacle_msg.polygon.points[0].y = 0
            # obstacle_msg.polygon.points[0].z = 0

            msg.obstacles.append(obstacle_msg)

        self.obstacle_pub.publish(msg)

        return

    def publish_pointcloud(self, new_points, predictions, t0):

        # data structure of binary blob output for PointCloud2 data type
        output_dtype = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring'],
                                 'formats': ['<f4', '<f4', '<f4', '<f4', '<u2'],
                                 'offsets': [0, 4, 8, 16, 20],
                                 'itemsize': 32})

        # fill structured numpy array with points and classes (in the intensity field). Fill ring with zeros to maintain Pointcloud2 structure
        c_points = np.c_[new_points, predictions, np.zeros(len(predictions))]
        c_points = np.core.records.fromarrays(c_points.transpose(), output_dtype)

        # convert to Pointcloud2 message and publish
        msg = pc2.array_to_pointcloud2(c_points, rospy.Time.from_sec(t0), self.velo_frame_id)
        
        self.pointcloud_pub.publish(msg)

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



