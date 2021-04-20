#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to test any model on any dataset
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

# Common libs
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir, remove, getcwd, makedirs
from sklearn.metrics import confusion_matrix
import time
import pickle
from torch.utils.data import DataLoader

# My libs
from utils.config import Config
from utils.metrics import IoU_from_confusions, smooth_metrics, fast_confusion
from utils.ply import read_ply
from models.architectures import FakeColliderLoss, KPCollider
from utils.tester import ModelTester
from utils.mayavi_visu import fast_save_future_anim, save_zoom_img

# Datasets
from datasets.MyhalCollision import MyhalCollisionDataset, MyhalCollisionSampler, MyhalCollisionCollate, MyhalCollisionSamplerTest

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def running_mean(signal, n, axis=0, stride=1):
    signal = np.array(signal)
    torch_conv = torch.nn.Conv1d(1, 1, kernel_size=2*n+1, stride=stride, bias=False)
    torch_conv.weight.requires_grad_(False)
    torch_conv.weight *= 0
    torch_conv.weight += 1 / (2*n+1)
    if signal.ndim == 1:
        torch_signal = torch.from_numpy(signal.reshape([1, 1, -1]).astype(np.float32))
        return torch_conv(torch_signal).squeeze().numpy()

    elif signal.ndim == 2:
        print('TODO implement with torch and stride here')
        smoothed = np.empty(signal.shape)
        if axis == 0:
            for i, sig in enumerate(signal):
                sig_sum = np.convolve(sig, np.ones((2*n+1,)), mode='same')
                sig_num = np.convolve(sig*0+1, np.ones((2*n+1,)), mode='same')
                smoothed[i, :] = sig_sum / sig_num
        elif axis == 1:
            for i, sig in enumerate(signal.T):
                sig_sum = np.convolve(sig, np.ones((2*n+1,)), mode='same')
                sig_num = np.convolve(sig*0+1, np.ones((2*n+1,)), mode='same')
                smoothed[:, i] = sig_sum / sig_num
        else:
            print('wrong axis')
        return smoothed

    else:
        print('wrong dimensions')
        return None


def IoU_multi_metrics(all_IoUs, smooth_n):

    # Get mean IoU for consecutive epochs to directly get a mean
    all_mIoUs = [np.hstack([np.mean(obj_IoUs, axis=1) for obj_IoUs in epoch_IoUs]) for epoch_IoUs in all_IoUs]
    smoothed_mIoUs = []
    for epoch in range(len(all_mIoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_mIoUs))
        smoothed_mIoUs += [np.mean(np.hstack(all_mIoUs[i0:i1]))]

    # Get mean for each class
    all_objs_mIoUs = [[np.mean(obj_IoUs, axis=1) for obj_IoUs in epoch_IoUs] for epoch_IoUs in all_IoUs]
    smoothed_obj_mIoUs = []
    for epoch in range(len(all_objs_mIoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_objs_mIoUs))

        epoch_obj_mIoUs = []
        for obj in range(len(all_objs_mIoUs[0])):
            epoch_obj_mIoUs += [np.mean(np.hstack([objs_mIoUs[obj] for objs_mIoUs in all_objs_mIoUs[i0:i1]]))]

        smoothed_obj_mIoUs += [epoch_obj_mIoUs]

    return np.array(smoothed_mIoUs), np.array(smoothed_obj_mIoUs)


def IoU_class_metrics(all_IoUs, smooth_n):

    # Get mean IoU per class for consecutive epochs to directly get a mean without further smoothing
    smoothed_IoUs = []
    for epoch in range(len(all_IoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_IoUs))
        smoothed_IoUs += [np.mean(np.vstack(all_IoUs[i0:i1]), axis=0)]
    smoothed_IoUs = np.vstack(smoothed_IoUs)
    smoothed_mIoUs = np.mean(smoothed_IoUs, axis=1)

    return smoothed_IoUs, smoothed_mIoUs


def load_confusions(filename, n_class):

    with open(filename, 'r') as f:
        lines = f.readlines()

    confs = np.zeros((len(lines), n_class, n_class))
    for i, line in enumerate(lines):
        C = np.array([int(value) for value in line.split()])
        confs[i, :, :] = C.reshape((n_class, n_class))

    return confs


def load_training_results(path):

    filename = join(path, 'training.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()

    epochs = []
    steps = []
    L_out = []
    L_p = []
    acc = []
    t = []
    L_2D_init = []
    L_2D_prop = []
    for line in lines[1:]:
        line_info = line.split()
        if (len(line) > 0):
            epochs += [int(line_info[0])]
            steps += [int(line_info[1])]
            L_out += [float(line_info[2])]
            L_p += [float(line_info[3])]
            acc += [float(line_info[4])]
            t += [float(line_info[5])]
            if len(line_info) > 6:
                L_2D_init += [float(line_info[6])]
                L_2D_prop += [float(line_info[7])]

        else:
            break

    ret_list = [epochs, steps, L_out, L_p, acc, t]

    if L_2D_init:
        ret_list.append(L_2D_init)
    if L_2D_prop:
        ret_list.append(L_2D_prop)

    return ret_list


def load_single_IoU(filename, n_parts):

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load all IoUs
    all_IoUs = []
    for i, line in enumerate(lines):
        all_IoUs += [np.reshape([float(IoU) for IoU in line.split()], [-1, n_parts])]
    return all_IoUs


def load_snap_clouds(path, dataset, only_last=False):

    cloud_folders = np.array([join(path, str(f, 'utf-8')) for f in listdir(path)
                              if str(f, 'utf-8').startswith('val_preds')])
    cloud_epochs = np.array([int(f.split('_')[-1]) for f in cloud_folders])
    epoch_order = np.argsort(cloud_epochs)
    cloud_epochs = cloud_epochs[epoch_order]
    cloud_folders = cloud_folders[epoch_order]

    Confs = np.zeros((len(cloud_epochs), dataset.num_classes, dataset.num_classes), dtype=np.int32)
    for c_i, cloud_folder in enumerate(cloud_folders):
        if only_last and c_i < len(cloud_epochs) - 1:
            continue

        # Load confusion if previously saved
        conf_file = join(cloud_folder, 'conf.txt')
        if isfile(conf_file):
            Confs[c_i] += np.loadtxt(conf_file, dtype=np.int32)

        else:
            for f in listdir(cloud_folder):
                f = str(f, 'utf-8')
                if f.endswith('.ply') and not f.endswith('sub.ply'):
                    data = read_ply(join(cloud_folder, f))
                    labels = data['class']
                    preds = data['preds']
                    Confs[c_i] += fast_confusion(labels, preds, dataset.label_values).astype(np.int32)

            np.savetxt(conf_file, Confs[c_i], '%12d')

        # Erase ply to save disk memory
        if c_i < len(cloud_folders) - 1:
            for f in listdir(cloud_folder):
                f = str(f, 'utf-8')
                if f.endswith('.ply'):
                    remove(join(cloud_folder, f))

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
        if label_value in dataset.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return cloud_epochs, IoU_from_confusions(Confs)


def load_multi_snap_clouds(path, dataset, file_i, only_last=False):

    cloud_folders = np.array([join(path, f) for f in listdir(path) if f.startswith('val_preds')])
    cloud_epochs = np.array([int(f.split('_')[-1]) for f in cloud_folders])
    epoch_order = np.argsort(cloud_epochs)
    cloud_epochs = cloud_epochs[epoch_order]
    cloud_folders = cloud_folders[epoch_order]

    if len(cloud_folders) > 0:
        dataset_folders = [f for f in listdir(cloud_folders[0]) if dataset.name in f]
        cloud_folders = [join(f, dataset_folders[file_i]) for f in cloud_folders]

    Confs = np.zeros((len(cloud_epochs), dataset.num_classes, dataset.num_classes), dtype=np.int32)
    for c_i, cloud_folder in enumerate(cloud_folders):
        if only_last and c_i < len(cloud_epochs) - 1:
            continue

        # Load confusion if previously saved
        conf_file = join(cloud_folder, 'conf_{:s}.txt'.format(dataset.name))
        if isfile(conf_file):
            Confs[c_i] += np.loadtxt(conf_file, dtype=np.int32)

        else:
            for f in listdir(cloud_folder):
                if f.endswith('.ply') and not f.endswith('sub.ply'):
                    if np.any([cloud_path.endswith(f) for cloud_path in dataset.files]):
                        data = read_ply(join(cloud_folder, f))
                        labels = data['class']
                        preds = data['preds']
                        Confs[c_i] += confusion_matrix(labels, preds, dataset.label_values).astype(np.int32)

            np.savetxt(conf_file, Confs[c_i], '%12d')

        # Erase ply to save disk memory
        if c_i < len(cloud_folders) - 1:
            for f in listdir(cloud_folder):
                if f.endswith('.ply'):
                    remove(join(cloud_folder, f))

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
        if label_value in dataset.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return cloud_epochs, IoU_from_confusions(Confs)


def load_multi_IoU(filename, n_parts):

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load all IoUs
    all_IoUs = []
    for i, line in enumerate(lines):
        obj_IoUs = [[float(IoU) for IoU in s.split()] for s in line.split('/')]
        obj_IoUs = [np.reshape(IoUs, [-1, n_parts[obj]]) for obj, IoUs in enumerate(obj_IoUs)]
        all_IoUs += [obj_IoUs]
    return all_IoUs


# ----------------------------------------------------------------------------------------------------------------------
#
#           Plot functions
#       \********************/
#


def compare_trainings(list_of_paths, list_of_labels=None):

    # Parameters
    # **********

    plot_lr = False
    smooth_epochs = 0.5
    stride = 2

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Training Logs
    # ******************

    all_epochs = []
    all_loss = []
    all_loss1 = []
    all_loss2 = []
    all_loss3 = []
    all_lr = []
    all_times = []
    all_RAMs = []

    for path in list_of_paths:

        # Check if log contains stuff
        check = 'val_IoUs.txt' in [str(f, 'utf-8') for f in listdir(path)]
        check = check or ('val_confs.txt' in [str(f, 'utf-8') for f in listdir(path)])
        check = check or ('val_RMSEs.txt' in [str(f, 'utf-8') for f in listdir(path)])

        if check:
            config = Config()
            config.load(path)
        else:
            continue


        # Load results
        training_res_list = load_training_results(path)
        if len(training_res_list) > 6:
            epochs, steps, L_out, L_p, acc, t, L_2D_init, L_2D_prop = training_res_list
        else:
            epochs, steps, L_out, L_p, acc, t = training_res_list
            L_2D_init = []
            L_2D_prop = []

        epochs = np.array(epochs, dtype=np.int32)
        epochs_d = np.array(epochs, dtype=np.float32)
        steps = np.array(steps, dtype=np.float32)

        # Compute number of steps per epoch
        max_e = np.max(epochs)
        first_e = np.min(epochs)
        epoch_n = []
        for i in range(first_e, max_e):
            bool0 = epochs == i
            e_n = np.sum(bool0)
            epoch_n.append(e_n)
            epochs_d[bool0] += steps[bool0] / e_n
        smooth_n = int(np.mean(epoch_n) * smooth_epochs)
        smooth_loss = running_mean(L_out, smooth_n, stride=stride)
        all_loss += [smooth_loss]
        if L_2D_init:
            all_loss2 += [running_mean(L_2D_init, smooth_n, stride=stride)]
            all_loss3 += [running_mean(L_2D_prop, smooth_n, stride=stride)]
            all_loss1 += [all_loss[-1] - all_loss2[-1] - all_loss3[-1]]
        all_epochs += [epochs_d[smooth_n:-smooth_n:stride]]
        all_times += [t[smooth_n:-smooth_n:stride]]

        # Learning rate
        if plot_lr:
            lr_decay_v = np.array([lr_d for ep, lr_d in config.lr_decays.items()])
            lr_decay_e = np.array([ep for ep, lr_d in config.lr_decays.items()])
            max_e = max(np.max(all_epochs[-1]) + 1, np.max(lr_decay_e) + 1)
            lr_decays = np.ones(int(np.ceil(max_e)), dtype=np.float32)
            lr_decays[0] = float(config.learning_rate)
            lr_decays[lr_decay_e] = lr_decay_v
            lr = np.cumprod(lr_decays)
            all_lr += [lr[np.floor(all_epochs[-1]).astype(np.int32)]]

        # Rescale losses
        rescale_losses = True
        if L_2D_init and rescale_losses:
            all_loss2[-1] *= 1 / config.power_2D_init_loss
            all_loss3[-1] *= 1 / config.power_2D_prop_loss

    # Plots learning rate
    # *******************

    if plot_lr:
        # Figure
        fig = plt.figure('lr')
        for i, label in enumerate(list_of_labels):
            plt.plot(all_epochs[i], all_lr[i], linewidth=1, label=label)

        # Set names for axes
        plt.xlabel('epochs')
        plt.ylabel('lr')
        plt.yscale('log')

        # Display legends and title
        plt.legend(loc=1)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Plots loss
    # **********

    if all_loss2:

        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 5))
        
        for i, label in enumerate(list_of_labels):
            axes[0].plot(all_epochs[i], all_loss1[i], linewidth=1, label=label)
            axes[1].plot(all_epochs[i], all_loss2[i], linewidth=1, label=label)
            axes[2].plot(all_epochs[i], all_loss3[i], linewidth=1, label=label)

        # Set names for axes
        for ax in axes:
            ax.set_xlabel('epochs')
        axes[0].set_ylabel('loss')
        axes[0].set_yscale('log')

        # Display legends and title
        axes[2].legend(loc=1)
        axes[0].set_title('3D_net loss')
        axes[1].set_title('2D_init loss')
        axes[2].set_title('2D_prop loss')

        # Customize the graph
        for ax in axes:
            ax.grid(linestyle='-.', which='both')



    else:

        # Figure
        fig = plt.figure('loss')
        for i, label in enumerate(list_of_labels):
            plt.plot(all_epochs[i], all_loss[i], linewidth=1, label=label)

        # Set names for axes
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.yscale('log')

        # Display legends and title
        plt.legend(loc=1)
        plt.title('Losses compare')

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Plot Times
    # **********

    # Figure
    fig = plt.figure('time')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_epochs[i], np.array(all_times[i]) / 3600, linewidth=1, label=label)

    # Set names for axes
    plt.xlabel('epochs')
    plt.ylabel('time')
    # plt.yscale('log')

    # Display legends and title
    plt.legend(loc=0)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all
    plt.show()


def compare_convergences_collision2D(list_of_paths, list_of_names=None):

    # Parameters
    # **********

    smooth_n = 4

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_mean_fe = []
    all_last_fe = []

    # Load parameters
    config = Config()
    config.load(list_of_paths[0])
    
    for path in list_of_paths:

        # Load config and saved results
        future_errors = np.loadtxt(join(path, 'future_error.txt'))
        print(future_errors.shape, future_errors.dtype)

        max_epoch = future_errors.shape[0]

        # Aggregate results
        epochs_d = np.array([i for i in range(max_epoch)])
        all_pred_epochs += [epochs_d[smooth_n:-smooth_n]]
        all_mean_fe += [running_mean(np.mean(future_errors, axis=1), smooth_n)]
        all_last_fe += [running_mean(future_errors[:, -1], smooth_n)]


    # Plots
    # *****

    # Figure
    fig = plt.figure('mean FE')
    for i, name in enumerate(list_of_names):
        p = plt.plot(all_pred_epochs[i], all_mean_fe[i], linewidth=1, label=name)
    plt.xlabel('epochs')
    plt.ylabel('mean_fe')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend()

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Figure
    fig = plt.figure('last FE')
    for i, name in enumerate(list_of_names):
        p = plt.plot(all_pred_epochs[i], all_last_fe[i], linewidth=1, label=name)
    plt.xlabel('epochs')
    plt.ylabel('last_fe')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend()

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all
    plt.show()

    a = 1/0


    for path in list_of_paths:

        # Load or compute the gif results
        folder_stats_file = join(path, 'log_stats_{:d}.pkl'.format(max_epoch))
        if exists(folder_stats_file):
            with open(folder_stats_file, 'rb') as file:
                all_p, all_gt = pickle.load(file)

        else:

            folder = join(path, 'future_preds')

            file_names = [f[:-4] for f in os.listdir(folder) if f.endswith('.npy')]
            all_p = []
            all_gt = []
            for f_i, f_name in enumerate(file_names):

                # read gt gif
                gt_name = join(folder, f_name + '.npy')
                im = imageio.get_reader(gt_name)
                frames = []
                for frame in im:
                    frames.append(frame[::zoom, ::zoom, :3])
                imgs = np.stack(frames)

                # Get gt mask of moving objects
                gt_mask = imgs[:, :, :, 0] > 1

                # Morpho closing to reduce noise
                struct2 = ndimage.generate_binary_structure(3, 2)
                struct2[[0, 2], :, :] = False
                gt_closed = ndimage.binary_closing(gt_mask, structure=struct2, iterations=3, border_value=1)

                # Debug
                #fig, anim = show_future_anim(gt_mask.astype(np.uint8) * 255)
                #fig2, anim2 = show_future_anim(gt_closed.astype(np.uint8) * 255)
                #plt.show()

                # Load predictions
                pre_name = join(folder, f_name + '_f_pre.gif')
                im = imageio.get_reader(pre_name)
                frames = []
                for frame in im:
                    frames.append(frame[::zoom, ::zoom, :3])
                preds = np.stack(frames)

                # Get moving predictions (with red/yellow colormap)
                r = np.copy(preds[1:, :, :, 0]).astype(np.float32)
                g = np.copy(preds[1:, :, :, 1]).astype(np.float32)
                no_red_mask = r < 1
                g[no_red_mask] = 0
                p = (r + g) / (255 + 255)
                gt = gt_closed[1:, :, :]

                all_p.append(p)
                all_gt.append(gt)

                print(f_i, len(file_names))

            # Stack and save these stats
            all_p = np.stack(all_p)
            all_gt = np.stack(all_gt)

            with open(folder_stats_file, 'wb') as file:
                pickle.dump([all_p, all_gt], file)






        # Get validation IoUs
        file = join(path, 'val_IoUs.txt')
        val_IoUs = load_single_IoU(file, nc_model)

        # Get Subpart IoUs
        file = join(path, 'subpart_IoUs.txt')
        subpart_IoUs = load_single_IoU(file, nc_model)

        # Get mean IoU
        val_class_IoUs, val_mIoUs = IoU_class_metrics(val_IoUs, smooth_n)
        subpart_class_IoUs, subpart_mIoUs = IoU_class_metrics(subpart_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_val_mIoUs += [val_mIoUs]
        all_val_class_IoUs += [val_class_IoUs]
        all_subpart_mIoUs += [subpart_mIoUs]
        all_subpart_class_IoUs += [subpart_class_IoUs]

        s = '{:^6.1f}|'.format(100*subpart_mIoUs[-1])
        for IoU in subpart_class_IoUs[-1]:
            s += '{:^6.1f}'.format(100*IoU)
        print(s)

    print(6*'-' + '|' + 6*config.num_classes*'-')
    for snap_IoUs in all_val_class_IoUs:
        if len(snap_IoUs) > 0:
            s = '{:^6.1f}|'.format(100*np.mean(snap_IoUs[-1]))
            for IoU in snap_IoUs[-1]:
                s += '{:^6.1f}'.format(100*IoU)
        else:
            s = '{:^6s}'.format('-')
            for _ in range(config.num_classes):
                s += '{:^6s}'.format('-')
        print(s)

    # Plots
    # *****

    # Figure
    fig = plt.figure('mIoUs')
    for i, name in enumerate(list_of_names):
        p = plt.plot(all_pred_epochs[i], all_subpart_mIoUs[i], '--', linewidth=1, label=name)
        plt.plot(all_pred_epochs[i], all_val_mIoUs[i], linewidth=1, color=p[-1].get_color())
    plt.xlabel('epochs')
    plt.ylabel('IoU')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    for c_i, c_name in enumerate(class_list):
        if c_i in displayed_classes:

            # Figure
            fig = plt.figure(c_name + ' IoU')
            for i, name in enumerate(list_of_names):
                plt.plot(all_pred_epochs[i], all_val_class_IoUs[i][:, c_i], linewidth=1, label=name)
            plt.xlabel('epochs')
            plt.ylabel('IoU')

            # Set limits for y axis
            #plt.ylim(0.8, 1)

            # Display legends and title
            plt.legend(loc=4)

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            #ax.set_yticks(np.arange(0.8, 1.02, 0.02))



    # Show all
    plt.show()

    return


def prediction_evolution_plot(chosen_log):

    ############
    # Parameters
    ############

    # Load parameters
    config = Config()
    config.load(chosen_log)

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    
    # Find all checkpoints in the chosen training folder
    chkp_path = join(chosen_log, 'checkpoints')
    chkps = np.sort([join(chkp_path, f) for f in listdir(chkp_path) if f[:4] == 'chkp'])

    print()
    print(chkps)
    print()
    print()

    # Get training and validation days
    val_path = join(chosen_log, 'val_preds')
    val_days = np.unique([f.split('_')[0] for f in listdir(val_path) if f.endswith('pots.ply')])

    # Util ops
    softmax = torch.nn.Softmax(1)
    sigmoid_2D = torch.nn.Sigmoid()
    fake_loss = FakeColliderLoss(config)

    # Result folder
    if not exists(join(config.saving_path, 'test_visu')):
        makedirs(join(config.saving_path, 'test_visu'))

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.
    config.augment_noise = 0
    augment_scale_min = 1.0
    augment_scale_max = 1.0
    config.augment_symmetries = [False, False, False]
    config.augment_rotation = 'none'
    config.validation_size = 100
    
    ##########################################
    # Choice of the image we want to visualize
    ##########################################

    # TODO
    wanted_inds = [50, 100, 150]
    
    ###########################
    # Initialize model and data
    ###########################

    # Dataset
    test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', balance_classes=False)

    # Specific sampler with pred inds
    test_sampler = MyhalCollisionSamplerTest(test_dataset, wanted_inds)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=MyhalCollisionCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    if config.max_val_points < 0:
        config.max_val_points = 1e9
        test_loader.dataset.max_in_p = 1e9
        test_sampler.calib_max_in(config, test_loader, untouched_ratio=0.95, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    # Init model
    net = KPCollider(config, test_dataset.label_values, test_dataset.ignored_labels)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chkps[-1])

    # Choose to train on CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        net.to(device)
    else:
        device = torch.device("cpu")

    # Force batch size to be 1
    test_dataset.batch_limit = 1

    ######################################
    # Start predictions with ckpts weights
    ######################################

    all_preds = []

    for chkp_i, chkp in enumerate(chkps[:5]):

        # Load new checkpoint weights
        if torch.cuda.is_available():
            checkpoint = torch.load(chkp)
        else:
            checkpoint = torch.load(chkp, map_location={'cuda:0': 'cpu'})
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch_i = checkpoint['epoch'] + 1
        net.eval()
        print("\nModel and training state restored from " + chkp)

        chkp_preds = []

        # Predict wanted inds with this chkp
        for i, batch in enumerate(test_loader):

            print(i, len(batch.scales), batch.scales)

            if 'cuda' in device.type:
                batch.to(device)

            # Forward pass
            outputs, preds_init_2D, preds_2D = net(batch, config)

            # Get probs and labels
            f_inds = batch.frame_inds.cpu().numpy()
            stck_init_preds = sigmoid_2D(preds_init_2D).cpu().detach().numpy()
            stck_future_logits = preds_2D.cpu().detach().numpy()
            stck_future_preds = sigmoid_2D(preds_2D).cpu().detach().numpy()
            stck_future_gts = batch.future_2D.cpu().detach().numpy()
            torch.cuda.synchronize(device)

            # Get the 2D predictions and gt (init_2D)
            b_i = 0
            img0 = stck_init_preds[b_i, 0, :, :, :]
            gt_im0 = np.copy(stck_future_gts[b_i, 0, :, :, :])
            gt_im1 = stck_future_gts[b_i, 0, :, :, :]
            gt_im1[:, :, 2] = np.max(stck_future_gts[b_i, 1:, :, :, 2], axis=0)
            img1 = stck_init_preds[b_i, 1, :, :, :]

            # Get the 2D predictions and gt (prop_2D)
            img = stck_future_preds[b_i, :, :, :, :]
            gt_im = stck_future_gts[b_i, 1:, :, :, :]

            # Future errors defined the same as the loss
            future_errors_bce = fake_loss.apply(gt_im, stck_future_logits[b_i, :, :, :, :], error='bce')
            future_errors = fake_loss.apply(gt_im, stck_future_logits[b_i, :, :, :, :], error='linear')
            future_errors = np.concatenate((future_errors_bce, future_errors), axis=0)

            # # Save prediction too in gif format
            # s_ind = f_inds[b_i, 0]
            # f_ind = f_inds[b_i, 1]
            # filename = '{:s}_{:07d}_e{:04d}.npy'.format(test_dataset.sequences[s_ind], f_ind, epoch_i)
            # gifpath = join(config.saving_path, 'test_visu', filename)
            # fast_save_future_anim(gifpath[:-4] + '_f_gt.gif', gt_im, zoom=5, correction=True)
            # fast_save_future_anim(gifpath[:-4] + '_f_pre.gif', img, zoom=5, correction=True)

            # Store all predictions
            chkp_preds.append(img)

        # Store all predictions
        chkp_preds = np.stack(chkp_preds, axis=0)
        all_preds.append(chkp_preds)
        
    # All predictions shape: [chkp_n, frames_n, T, H, W, 3]
    all_preds = np.stack(all_preds, axis=0)

    ################
    # Visualizations
    ################

    # First idea: evolution of the gif
    

    a = 1/0

    


    return











# ----------------------------------------------------------------------------------------------------------------------
#
#           Experiments
#       \*****************/
#


def collider_tests_1(old_result_limit):
    """
    A lot have been going on, we know come back to basics and experiment with bouncers. In this experiment, bouncers have various speeds, and we try different loss weights
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-04-14_18-48-28'
    end = 'Log_2021-09-15_17-03-17'
    

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['Bouncer-Loss=1/1',
                  'Bouncer-Loss=0.5/4',
                  'Bouncer-Loss=0.5/4-NEW_METRIC',
                  'Bouncer-Loss=1/2-NEW_METRIC',
                  'Bouncer-Loss=1/2-NEW_METRIC',
                  'test'
                  ]

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main call
#       \***************/
#


if __name__ == '__main__':

    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # Old result limit
    old_res_lim = 'Log_2020-05-04_19-17-59'

    # My logs: choose the logs to show
    logs, logs_names = collider_tests_1(old_res_lim)
    #os.environ['QT_DEBUG_PLUGINS'] = '1'


    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # Check that all logs are of the same dataset. Different object can be compared
    plot_dataset = None
    config = None
    for log in logs:
        config = Config()
        config.load(log)
        this_dataset = config.dataset
        if plot_dataset:
            if plot_dataset == this_dataset:
                continue
            else:
                raise ValueError('All logs must share the same dataset to be compared')
        else:
            plot_dataset = this_dataset

    ################
    # Plot functions
    ################

    # Evolution of predictions from checkpoints to checkpoints
    prediction_evolution_plot(logs[-1])

    # # Plot the training loss and accuracy
    # compare_trainings(logs, logs_names)

    # # Plot the validation
    # if config.dataset_task == 'collision_prediction':
    #     if config.dataset.startswith('MyhalCollision'):
    #         compare_convergences_collision2D(logs, logs_names)
    # else:
    #     raise ValueError('Unsupported dataset : ' + plot_dataset)




