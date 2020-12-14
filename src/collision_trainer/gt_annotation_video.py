#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on MyhalSim dataset
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

# Common libs
import numpy as np
from utils.ply import read_ply

from mayavi import mlab
import imageio
import pickle
import time
from os import listdir
from os.path import join

import open3d as o3d
import matplotlib.pyplot as plt
from datasets.MyhalSim import *


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def pcd_update_from_ply(ply_name, pcd, H_frame, scalar_field='classif'):

    # Load first frame
    data = read_ply(ply_name)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    classif = data[scalar_field]

    if np.sum(np.abs(H_frame)) > 1e-3:
        points = np.hstack((points, np.ones_like(points[:, :1])))
        points = np.matmul(points, H_frame.T).astype(np.float32)[:, :3]

    # Get colors
    np_colors = colormap[classif, :]

    # Save as pcd
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)

    if 'nx' in data.dtype.names:
        normals = np.vstack((data['nx'], data['ny'], data['nz'])).T
        if np.sum(np.abs(H_frame)) > 1e-3:
            world_normals = np.matmul(normals, H_frame[:3, :3].T).astype(np.float32)
        pcd.normals = o3d.utility.Vector3dVector(normals)

if __name__ == '__main__':



    # List of days we want to create video from
    # my_days = ['2020-07-06-21-04-50',
    #            '2020-07-06-21-04-57',
    #            '2020-07-06-21-55-08',
    #            '2020-07-06-21-55-16',
    #            '2020-07-07-12-58-21',
    #            '2020-07-07-12-58-23']

    my_days = ['2020-11-05-22-09-11']


    #day_list = day_list[1:]

    # path of the classified frames
    #path = '../../Myhal_Simulation/annotated_frames'
    #path = '../../Myhal_Simulation/predicted_frames'
    path = '../../Myhal_Simulation/simulated_runs/'

    # Scalar field we want to show
    #scalar_field = 'classif'
    #scalar_field = 'labels'
    #scalar_field = 'pred'
    scalar_field = 'cat'

    # Should the cam be static or follow the robot
    following_cam = True

    # Are frame localized in world coordinates?
    localized_frames = False

    # Colormap
    colormap = np.array([[209, 209, 209],
                         [122, 122, 122],
                         [255, 255, 0],
                         [0, 98, 255],
                         [255, 0, 0]], dtype=np.float32) / 255

    # Colormap
    # colormap = np.array([[122, 122, 122],
    #                      [0, 251, 251],
    #                      [255, 0, 0],
    #                      [89, 248, 123],
    #                      [0, 0, 255],
    #                      [255, 255, 0],
    #                      [0, 190, 0]], dtype=np.float32) / 255

    ##################
    # Mayavi animation
    ##################

    # Window for headless visu
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)

    for day in my_days:

        ########
        # Init #
        ########

        # Get annotated lidar frames
        #frames_folder = join(path, day, 'classified_frames')
        #frames_folder = join(path, day, 'sim_frames')
        #frames_folder = join(path, day, 'classif1_frames')
        frames_folder = join(path, day, 'classif2_frames')
        f_names = [f for f in listdir(frames_folder) if f[-4:] == '.ply']
        f_times = np.array([float(f[:-4]) for f in f_names], dtype=np.float64)
        f_names = np.array([join(frames_folder, f) for f in f_names])
        ordering = np.argsort(f_times)
        f_names = f_names[ordering]
        f_times = f_times[ordering]

        # Load mapping poses
        map_traj_file = join(path, day, 'logs-'+day, 'map_traj.ply')
        data = read_ply(map_traj_file)
        map_T = np.vstack([data['pos_x'], data['pos_y'], data['pos_z']]).T
        map_Q = np.vstack([data['rot_x'], data['rot_y'], data['rot_z'], data['rot_w']]).T

        # Times
        day_map_t = data['time']

        # Convert map to homogenous rotation/translation matrix
        map_R = scipyR.from_quat(map_Q)
        map_R = map_R.as_matrix()
        day_map_H = np.zeros((len(day_map_t), 4, 4))
        day_map_H[:, :3, :3] = map_R
        day_map_H[:, :3, 3] = map_T
        day_map_H[:, 3, 3] = 1

        print(len(f_names), len(day_map_t))

        # Verify which frames we need:
        frame_names = []
        f_name_i = 0
        for i, t in enumerate(day_map_t):

            f_name = '{:.6f}.ply'.format(t)
            while f_name_i < len(f_names) and not (f_names[f_name_i].endswith(f_name)):
                print(f_names[f_name_i], ' skipped for ', f_name)
                f_name_i += 1

            if f_name_i >= len(f_names):
                break

            frame_names.append(f_names[f_name_i])
            f_name_i += 1

        print(len(frame_names), len(day_map_t))

        for fffn, ttt in zip(frame_names, day_map_t):
            print(fffn, ttt)


        print(len(f_names), len(day_map_t))
        ######
        # Go #
        ######

        # Load the first frame in the window
        vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        if localized_frames:
            H_frame = np.zeros((4, 4))
        else:
            # H_frame = day_map_H[0]
            # Only rotate frame to have a hoverer mode
            H_frame = day_map_H[0]
            H_frame[:3, 3] *= 0

        pcd_update_from_ply(frame_names[0], pcd, H_frame, scalar_field)
        vis.add_geometry(pcd)

        # Apply render options
        render_option = vis.get_render_option()
        render_option.light_on = False
        render_option.point_size = 5
        render_option.show_coordinate_frame = True

        # Prepare view point
        view_control = vis.get_view_control()
        if following_cam:
            target = day_map_H[0][:3, 3]
            front = target + np.array([0.0, -10.0, 15.0])
            view_control.set_front(front)
            view_control.set_lookat(target)
            view_control.set_up([0.0, 0.0, 1.0])
            view_control.set_zoom(0.2)
            pinhole0 = view_control.convert_to_pinhole_camera_parameters()
            follow_H0 = np.copy(pinhole0.extrinsic)
        else:
            traj_points = np.vstack([H[:3, 3] for H in day_map_H])
            target = np.mean(traj_points, axis=0)
            front = target + np.array([10.0, 10.0, 10.0])
            view_control.set_front(front)
            view_control.set_lookat(target)
            view_control.set_up([0.0, 0.0, 1.0])
            view_control.set_zoom(0.4)
            follow_H0 = None
            pinhole0 = None

        # Advanced display
        N = len(frame_names)
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nGenerating Open3D screenshots for ' + day)
        video_list = []
        for i, f_name in enumerate(frame_names):

            if i > len(day_map_H) - 1:
                break

            # New frame
            if localized_frames:
                H_frame = np.zeros((4, 4))
            else:
                # H_frame = day_map_H[i]
                # Only rotate frame to have a hoverer mode
                H_frame = day_map_H[i]
                H_frame[:3, 3] *= 0
            pcd_update_from_ply(f_name, pcd, H_frame, scalar_field=scalar_field)
            vis.update_geometry(pcd)

            # New view point
            if following_cam:
                # third person mode
                follow_H = np.dot(follow_H0, np.linalg.inv(day_map_H[i]))
                # pinhole0.extrinsic = follow_H
                # view_control.convert_from_pinhole_camera_parameters(pinhole0)

            # Render
            vis.poll_events()
            vis.update_renderer()

            # Screenshot
            image = vis.capture_screen_float_buffer(False)
            video_list.append((np.asarray(image) * 255).astype(np.uint8))
            #plt.imsave('test_{:d}.png'.format(i), image, dpi=1)

            print('', end='\r')
            print(fmt_str.format('#' * (((i + 1) * progress_n) // N), 100 * (i + 1) / N), end='', flush=True)

        # Show a nice 100% progress bar
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\n')

        # Path for saving
        video_path = join(path, day, 'video_{:s}_{:s}.mp4'.format(scalar_field, day))

        # Write video file
        print('\nWriting video file for ' + day)
        kargs = {'macro_block_size': None}
        with imageio.get_writer(video_path, mode='I', fps=30, quality=10, **kargs) as writer:
            N = len(video_list)
            for i, frame in enumerate(video_list):
                writer.append_data(frame)

                print('', end='\r')
                print(fmt_str.format('#' * (((i + 1) * progress_n) // N), 100 * (i + 1) / N), end='', flush=True)

        # Show a nice 100% progress bar
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\n')

    vis.destroy_window()