#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Script for various visualization with mayavi
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import numpy as np
from sklearn.neighbors import KDTree
from os import makedirs, remove, rename, listdir
from os.path import exists, join
import time

import sys

# PLY reader
from utils.ply import write_ply, read_ply
from utils.metrics import fast_confusion, IoU_from_confusions

# Configuration class
from utils.config import Config


def show_ModelNet_models(all_points):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Indices
    global file_i
    file_i = 0

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Plot new data feature
        points = all_points[file_i]

        # Rescale points for visu
        points = (points * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        # Show point clouds colorized with activations
        activations = mlab.points3d(points[:, 0],
                                    points[:, 1],
                                    points[:, 2],
                                    points[:, 2],
                                    scale_factor=3.0,
                                    scale_mode='none',
                                    figure=fig1)

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i

        if vtk_obj.GetKeyCode() in ['g', 'G']:

            file_i = (file_i - 1) % len(all_points)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:

            file_i = (file_i + 1) % len(all_points)
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_ModelNet_examples(clouds, cloud_normals=None, cloud_labels=None):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    if cloud_labels is None:
        cloud_labels = [points[:, 2] for points in clouds]

    # Indices
    global file_i, show_normals
    file_i = 0
    show_normals = True

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Plot new data feature
        points = clouds[file_i]
        labels = cloud_labels[file_i]
        if cloud_normals is not None:
            normals = cloud_normals[file_i]
        else:
            normals = None

        # Rescale points for visu
        points = (points * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        # Show point clouds colorized with activations
        activations = mlab.points3d(points[:, 0],
                                    points[:, 1],
                                    points[:, 2],
                                    scale_factor=2.0,
                                    scale_mode='none',
                                    figure=fig1)
        if normals is not None and show_normals:
            # Dont show all normals or we cant see well
            # random_N = points.shape[0] // 4
            # random_inds = np.random.permutation(points.shape[0])[:random_N]
            random_inds = np.arange(points.shape[0])
            activations = mlab.quiver3d(points[random_inds, 0],
                                        points[random_inds, 1],
                                        points[random_inds, 2],
                                        normals[random_inds, 0],
                                        normals[random_inds, 1],
                                        normals[random_inds, 2],
                                        scale_factor=10.0,
                                        line_width=1.0,
                                        scale_mode='none',
                                        figure=fig1)

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i, show_normals

        if vtk_obj.GetKeyCode() in ['g', 'G']:
            file_i = (file_i - 1) % len(clouds)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:
            file_i = (file_i + 1) % len(clouds)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            show_normals = not show_normals
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_ModelNet_lrf(clouds, clouds_lrf):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Indices
    global file_i, show_normals
    file_i = 0
    show_normals = True

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Plot new data feature
        points = clouds[file_i]
        lrf = clouds_lrf[file_i]

        # Rescale points for visu
        points = (points * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        # Show point clouds colorized with activations
        activations = mlab.points3d(points[:, 0],
                                    points[:, 1],
                                    points[:, 2],
                                    scale_factor=2.0,
                                    scale_mode='none',
                                    figure=fig1)

        if show_normals:

            # Dont show all normals or we cant see well
            for i in range(3):
                color = [0.0, 0.0, 0.0]
                color[i] = 1.0
                activations = mlab.quiver3d(points[::100, 0],
                                            points[::100, 1],
                                            points[::100, 2],
                                            lrf[::100, 0, i],
                                            lrf[::100, 1, i],
                                            lrf[::100, 2, i],
                                            scale_factor=10.0,
                                            line_width=1.0,
                                            scale_mode='none',
                                            color=tuple(color),
                                            figure=fig1)

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i, show_normals

        if vtk_obj.GetKeyCode() in ['g', 'G']:
            file_i = (file_i - 1) % len(clouds)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:
            file_i = (file_i + 1) % len(clouds)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            show_normals = not show_normals
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_neighbors(query, supports, neighbors):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Indices
    global file_i
    file_i = 0

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Rescale points for visu
        p1 = (query * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
        p2 = (supports * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        l1 = p1[:, 2]*0
        l1[file_i] = 1

        l2 = p2[:, 2]*0 + 2
        l2[neighbors[file_i]] = 3

        # Show point clouds colorized with activations
        activations = mlab.points3d(p1[:, 0],
                                    p1[:, 1],
                                    p1[:, 2],
                                    l1,
                                    scale_factor=2.0,
                                    scale_mode='none',
                                    vmin=0.0,
                                    vmax=3.0,
                                    figure=fig1)

        activations = mlab.points3d(p2[:, 0],
                                    p2[:, 1],
                                    p2[:, 2],
                                    l2,
                                    scale_factor=3.0,
                                    scale_mode='none',
                                    vmin=0.0,
                                    vmax=3.0,
                                    figure=fig1)

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i

        if vtk_obj.GetKeyCode() in ['g', 'G']:

            file_i = (file_i - 1) % len(query)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:

            file_i = (file_i + 1) % len(query)
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_input_batch(batch):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Input', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Unstack batch
    all_points = batch.unstack_points()
    all_neighbors = batch.unstack_neighbors()
    all_pools = batch.unstack_pools()

    # Indices
    global b_i, l_i, neighb_i, show_pools
    b_i = 0
    l_i = 0
    neighb_i = 0
    show_pools = False

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Rescale points for visu
        p = (all_points[l_i][b_i] * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
        labels = p[:, 2] * 0

        if show_pools:
            p2 = (all_points[l_i + 1][b_i][neighb_i:neighb_i + 1] * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
            p = np.vstack((p, p2))
            labels = np.hstack((labels, np.ones((1,), dtype=np.int32) * 3))
            pool_inds = all_pools[l_i][b_i][neighb_i]
            pool_inds = pool_inds[pool_inds >= 0]
            labels[pool_inds] = 2
        else:
            neighb_inds = all_neighbors[l_i][b_i][neighb_i]
            neighb_inds = neighb_inds[neighb_inds >= 0]
            labels[neighb_inds] = 2
            labels[neighb_i] = 3

        # Show point clouds colorized with activations
        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      labels,
                      scale_factor=2.0,
                      scale_mode='none',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)

        """
        mlab.points3d(p[-2:, 0],
                      p[-2:, 1],
                      p[-2:, 2],
                      labels[-2:]*0 + 3,
                      scale_factor=0.16 * 1.5 * 50,
                      scale_mode='none',
                      mode='cube',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)
        mlab.points3d(p[-1:, 0],
                      p[-1:, 1],
                      p[-1:, 2],
                      labels[-1:]*0 + 2,
                      scale_factor=0.16 * 2 * 2.5 * 1.5 * 50,
                      scale_mode='none',
                      mode='sphere',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)

        """

        # New title
        title_str = '<([) b_i={:d} (])>    <(,) l_i={:d} (.)>    <(N) n_i={:d} (M)>'.format(b_i, l_i, neighb_i)
        mlab.title(title_str, color=(0, 0, 0), size=0.3, height=0.90)
        if show_pools:
            text = 'pools (switch with G)'
        else:
            text = 'neighbors (switch with G)'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.3)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global b_i, l_i, neighb_i, show_pools

        if vtk_obj.GetKeyCode() in ['[', '{']:
            b_i = (b_i - 1) % len(all_points[l_i])
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in [']', '}']:
            b_i = (b_i + 1) % len(all_points[l_i])
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in [',', '<']:
            if show_pools:
                l_i = (l_i - 1) % (len(all_points) - 1)
            else:
                l_i = (l_i - 1) % len(all_points)
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in ['.', '>']:
            if show_pools:
                l_i = (l_i + 1) % (len(all_points) - 1)
            else:
                l_i = (l_i + 1) % len(all_points)
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            neighb_i = (neighb_i - 1) % all_points[l_i][b_i].shape[0]
            update_scene()

        elif vtk_obj.GetKeyCode() in ['m', 'M']:
            neighb_i = (neighb_i + 1) % all_points[l_i][b_i].shape[0]
            update_scene()

        elif vtk_obj.GetKeyCode() in ['g', 'G']:
            if l_i < len(all_points) - 1:
                show_pools = not show_pools
                neighb_i = 0
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_input_normals(batch):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Input', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Unstack batch
    all_points = batch.unstack_points()
    all_neighbors = batch.unstack_neighbors()
    all_normals = batch.unstack_pools()

    # Indices
    global b_i, l_i, neighb_i, show_pools
    b_i = 0
    l_i = 0
    neighb_i = 0
    show_pools = False

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Rescale points for visu
        p = (all_points[l_i][b_i] * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
        labels = p[:, 2] * 0

        if show_pools:
            p2 = (all_points[l_i + 1][b_i][neighb_i:neighb_i + 1] * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
            p = np.vstack((p, p2))
            labels = np.hstack((labels, np.ones((1,), dtype=np.int32) * 3))
            pool_inds = all_pools[l_i][b_i][neighb_i]
            pool_inds = pool_inds[pool_inds >= 0]
            labels[pool_inds] = 2
        else:
            neighb_inds = all_neighbors[l_i][b_i][neighb_i]
            neighb_inds = neighb_inds[neighb_inds >= 0]
            labels[neighb_inds] = 2
            labels[neighb_i] = 3

        # Show point clouds colorized with activations
        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      labels,
                      scale_factor=2.0,
                      scale_mode='none',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)

        """
        mlab.points3d(p[-2:, 0],
                      p[-2:, 1],
                      p[-2:, 2],
                      labels[-2:]*0 + 3,
                      scale_factor=0.16 * 1.5 * 50,
                      scale_mode='none',
                      mode='cube',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)
        mlab.points3d(p[-1:, 0],
                      p[-1:, 1],
                      p[-1:, 2],
                      labels[-1:]*0 + 2,
                      scale_factor=0.16 * 2 * 2.5 * 1.5 * 50,
                      scale_mode='none',
                      mode='sphere',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)

        """

        # New title
        title_str = '<([) b_i={:d} (])>    <(,) l_i={:d} (.)>    <(N) n_i={:d} (M)>'.format(b_i, l_i, neighb_i)
        mlab.title(title_str, color=(0, 0, 0), size=0.3, height=0.90)
        if show_pools:
            text = 'pools (switch with G)'
        else:
            text = 'neighbors (switch with G)'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.3)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global b_i, l_i, neighb_i, show_pools

        if vtk_obj.GetKeyCode() in ['[', '{']:
            b_i = (b_i - 1) % len(all_points[l_i])
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in [']', '}']:
            b_i = (b_i + 1) % len(all_points[l_i])
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in [',', '<']:
            if show_pools:
                l_i = (l_i - 1) % (len(all_points) - 1)
            else:
                l_i = (l_i - 1) % len(all_points)
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in ['.', '>']:
            if show_pools:
                l_i = (l_i + 1) % (len(all_points) - 1)
            else:
                l_i = (l_i + 1) % len(all_points)
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            neighb_i = (neighb_i - 1) % all_points[l_i][b_i].shape[0]
            update_scene()

        elif vtk_obj.GetKeyCode() in ['m', 'M']:
            neighb_i = (neighb_i + 1) % all_points[l_i][b_i].shape[0]
            update_scene()

        elif vtk_obj.GetKeyCode() in ['g', 'G']:
            if l_i < len(all_points) - 1:
                show_pools = not show_pools
                neighb_i = 0
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_point_cloud(points):
    from mayavi import mlab


    # Create figure for features
    #fig1 = mlab.figure('Deformations', bgcolor=(1.0, 1.0, 1.0), size=(1280, 920))
    #fig1.scene.parallel_projection = False

    mlab.points3d(points[:, 0],
                  points[:, 1],
                  points[:, 2],
                  resolution=8,
                  scale_factor=1,
                  scale_mode='none',
                  color=(0, 1, 1))
    #mlab.show()

    #TODO: mayavi interactive mode?

    input('press enter to resume script')


    a = 1/0


def show_bundle_adjustment(bundle_frames_path):
    from mayavi import mlab

    ##################################
    # Load ply file with bundle frames
    ##################################

    # Load ply
    data = read_ply(bundle_frames_path)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    bundle_inds = data['b']
    steps = data['s']

    # Get steps and bundli indices
    B = int(np.max(bundle_inds)) + 1
    S = int(np.max(steps)) + 1

    # Adjust bundle inds for color
    bundle_inds[bundle_inds < 0.1] -= 2 * B

    # Reshape points
    points = points.reshape(S, B, -1, 3)
    bundle_inds = bundle_inds.reshape(S, B, -1)

    # Get original frame
    size = np.linalg.norm(points[0, 0, -1] - points[0, 0, 0])
    x = np.linspace(0, size, 50, dtype=np.float32)
    p0 = np.hstack((np.vstack((x, x * 0, x * 0)), np.vstack((x * 0, x, x * 0)), np.vstack((x * 0, x * 0, x)))).T


    ###############
    # Visualization
    ###############

    # Create figure for features
    fig1 = mlab.figure('Bundle', bgcolor=(1.0, 1.0, 1.0), size=(1280, 920))
    fig1.scene.parallel_projection = False

    # Indices
    global s, plots, p_scale
    p_scale = 0.003
    s = 0
    plots = {}

    def update_scene():
        global s, plots, p_scale

        # Get the current view
        v = mlab.view()
        roll = mlab.roll()

        #  clear figure
        for key in plots.keys():
            plots[key].remove()

        plots = {}

        # Get points we want to show
        p = points[s].reshape(-1, 3)
        b = bundle_inds[s].reshape(-1)

        plots['points'] = mlab.points3d(p[:, 0],
                                        p[:, 1],
                                        p[:, 2],
                                        b,
                                        resolution=8,
                                        scale_factor=p_scale,
                                        scale_mode='none',
                                        figure=fig1)

        # Show original frame
        plots['points0'] = mlab.points3d(p0[:, 0],
                                         p0[:, 1],
                                         p0[:, 2],
                                         resolution=8,
                                         scale_factor=p_scale * 2,
                                         scale_mode='none',
                                         color=(0, 0, 0),
                                         figure=fig1)

        # New title
        plots['title'] = mlab.title(str(s), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        plots['text'] = mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        # plots['orient'] = mlab.orientation_axes()

        # Set the saved view
        mlab.view(*v)
        mlab.roll(roll)

        return

    def keyboard_callback(vtk_obj, event):
        global s, plots, p_scale

        if vtk_obj.GetKeyCode() in ['b', 'B']:
            p_scale /= 1.5
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            p_scale *= 1.5
            update_scene()

        if vtk_obj.GetKeyCode() in ['g', 'G']:
            s = (s - 1) % S
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:
            s = (s + 1) % S
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def compare_Shapenet_results(logs, log_names):
    from mayavi import mlab

    ######
    # Init
    ######

    # dataset parameters
    n_parts = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    label_to_names = {0: 'Airplane',
                      1: 'Bag',
                      2: 'Cap',
                      3: 'Car',
                      4: 'Chair',
                      5: 'Earphone',
                      6: 'Guitar',
                      7: 'Knife',
                      8: 'Lamp',
                      9: 'Laptop',
                      10: 'Motorbike',
                      11: 'Mug',
                      12: 'Pistol',
                      13: 'Rocket',
                      14: 'Skateboard',
                      15: 'Table'}
    name_to_label = {v: k for k, v in label_to_names.items()}

    # Get filenames that are common to all tests
    example_paths = [join(f, '_Examples') for f in logs]
    n_log = len(logs)
    file_names = np.sort([f for f in listdir(example_paths[0]) if f.endswith('.ply')])
    for log_path in example_paths:
        file_names = np.sort([f for f in listdir(log_path) if f in file_names])

    logs_clouds = []
    logs_labels = []
    logs_preds = []
    logs_IoUs = []
    logs_objs = []
    for log_path in example_paths:
        logs_clouds += [[]]
        logs_labels += [[]]
        logs_preds += [[]]
        logs_IoUs += [[]]
        logs_objs += [[]]
        for file_name in file_names:
            file_path = join(log_path, file_name)
            obj_lbl = name_to_label[file_name.split('_')[0]]
            logs_objs[-1] += [obj_lbl]

            data = read_ply(file_path)
            logs_clouds[-1] += [np.vstack((data['x'], -data['y'], data['z'])).T]

            lbls = data['gt']
            preds = data['pre']
            logs_labels[-1] += [lbls]
            logs_preds[-1] += [preds]

            # Compute IoUs
            parts = [j for j in range(n_parts[obj_lbl])]
            C = fast_confusion(lbls, preds, np.array(parts, dtype=np.int32))
            IoU = np.mean(IoU_from_confusions(C))

            logs_IoUs[-1] += [IoU]

    logs_IoUs = np.array(logs_IoUs, dtype=np.float32).T

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1300, 700))
    fig1.scene.parallel_projection = False

    # Indices
    global file_i, v, roll
    file_i = 0
    v = None
    roll = None

    def update_scene():
        global v, roll

        # Get current view
        if v is not None:
            v = mlab.view()
            roll = mlab.roll()

        #  clear figure
        mlab.clf(fig1)

        # Plot new data feature
        print(logs_IoUs[file_i])

        n_p = n_parts[logs_objs[0][file_i]]

        # Show point clouds colorized with activations
        activations = mlab.points3d(logs_clouds[0][file_i][:, 0],
                                    logs_clouds[0][file_i][:, 1],
                                    logs_clouds[0][file_i][:, 2],
                                    logs_labels[0][file_i],
                                    scale_factor=0.03,
                                    scale_mode='none',
                                    vmin=0,
                                    vmax=n_p-1,
                                    figure=fig1)

        for i in range(n_log):
            # Rescale points for visu
            pts = logs_clouds[i][file_i] + np.array([0.0, 2.5, 0.0]) * (i + 1)

            # Show point clouds colorized with activations
            activations = mlab.points3d(pts[:, 0],
                                        pts[:, 1],
                                        pts[:, 2],
                                        logs_preds[i][file_i],
                                        scale_factor=0.03,
                                        scale_mode='none',
                                        vmin=0,
                                        vmax=n_p-1,
                                        figure=fig1)

        # New title

        s = '{:s}: GT'.format(file_names[file_i])
        for IoU in logs_IoUs[file_i]:
            s += ' / {:.1f}'.format(100 * IoU)

        mlab.title(s, color=(0, 0, 0), size=0.2, height=0.01)
        mlab.orientation_axes()

        # Set/Get the current view
        if v is None:
            v = mlab.view()
            roll = mlab.roll()
        else:
            mlab.view(*v)
            mlab.roll(roll)

        return

    def keyboard_callback(vtk_obj, event):
        global file_i, points, labels, predictions

        if vtk_obj.GetKeyCode() in ['g', 'G']:
            file_i = (file_i - 1) % len(logs_labels[0])
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:
            file_i = (file_i + 1) % len(logs_labels[0])
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()




















