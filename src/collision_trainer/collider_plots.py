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
os.environ.update(OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',)
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir, remove, getcwd, makedirs
from sklearn.metrics import confusion_matrix
import time
import pickle
from torch.utils.data import DataLoader
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import imageio
from scipy import ndimage

# My libs
from utils.config import Config
from utils.metrics import IoU_from_confusions, smooth_metrics, fast_confusion
from utils.ply import read_ply
from models.architectures import FakeColliderLoss, KPCollider
from utils.tester import ModelTester
from utils.mayavi_visu import fast_save_future_anim, save_zoom_img, colorize_collisions, zoom_collisions, superpose_gt

# Datasets
from datasets.MyhalCollision import MyhalCollisionDataset, MyhalCollisionSampler, MyhalCollisionCollate, MyhalCollisionSamplerTest

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def running_mean(signal, n, axis=0, stride=1):

    # Create the smoothing convolution
    torch_conv = torch.nn.Conv1d(1, 1, kernel_size=2 * n + 1, stride=stride, bias=False)
    torch_conv.weight.requires_grad_(False)
    torch_conv.weight *= 0
    torch_conv.weight += 1 / (2 * n + 1)

    signal = np.array(signal)
    if signal.ndim == 1:

        # Reshape signal to torch Tensor
        signal = np.expand_dims(np.expand_dims(signal, 0), 1).astype(np.float32)
        torch_signal = torch.from_numpy(signal)

        # Get result
        smoothed = torch_conv(torch_signal).squeeze().numpy()

        return smoothed

    elif signal.ndim == 2:

        # transpose if we want axis 0
        if axis == 0:
            signal = signal.T

        # Reshape signal to torch Tensor
        signal = np.expand_dims(signal, 1).astype(np.float32)
        torch_signal = torch.from_numpy(signal)

        # Get result
        smoothed = torch_conv(torch_signal).squeeze().numpy()

        # transpose if we want axis 0
        if axis == 0:
            smoothed = smoothed.T

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


def compare_trainings(list_of_paths, list_of_labels=None, smooth_epochs=3.0):

    # Parameters
    # **********

    plot_lr = False
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


def compare_convergences_collision2D(list_of_paths, list_of_names=None, smooth_n=20):

    # Parameters
    # **********

    

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_fe = []
    all_bce = []
    all_fp = []
    all_fp_bce = []
    all_fn = []
    all_fn_bce = []

    # Load parameters
    config = Config()
    config.load(list_of_paths[0])

    for path in list_of_paths:

        # Load config and saved results
        metric_list = []
        file_list = ['subpart_IoUs.txt',
                     'val_IoUs.txt',
                     'reconstruction_error.txt',
                     'future_error.txt',
                     'future_error_bce.txt',
                     'future_FP.txt',
                     'future_FN.txt',
                     'future_FP_bce.txt',
                     'future_FN_bce.txt']
        max_epoch = 0
        for filename in file_list:
            try:
                metric = np.loadtxt(join(path, filename))
                max_epoch = max(max_epoch, metric.shape[0])
                smoothed = running_mean(metric, smooth_n)
            except OSError as e:
                smoothed = np.zeros((0, 0), dtype=np.float64)
            metric_list.append(smoothed)
        (IoUs,
         val_IoUs,
         mean_recons_e,
         mean_future_e,
         mean_future_bce,
         mean_future_FP,
         mean_future_FN,
         mean_future_FP_bce,
         mean_future_FN_bce) = metric_list

        # Epoch count
        epochs_d = np.array([i for i in range(max_epoch)])

        # Aggregate results
        all_pred_epochs += [epochs_d[smooth_n:-smooth_n]]
        all_fe += [mean_future_e]
        all_bce += [mean_future_bce]
        all_fp += [mean_future_FP]
        all_fp_bce += [mean_future_FP_bce]
        all_fn += [mean_future_FN]
        all_fn_bce += [mean_future_FN_bce]

    # Plots
    # *****

    # create plots

    for reduc in ['mean']:
        for error, error_name in zip([all_fe, all_bce, all_fp, all_fp_bce, all_fn, all_fn_bce],
                                     ['all_fe', 'all_bce', 'all_fp', 'all_fp_bce', 'all_fn', 'all_fn_bce']):
            fig = plt.figure(reduc + ' ' + error_name[4:])
            for i, name in enumerate(list_of_names):
                if error[i].shape[0] > 0:
                    if reduc == 'last':
                        plotted_e = error[i][:, -1]
                    else:
                        plotted_e = np.mean(error[i], axis=1)
                else:
                    plotted_e = all_pred_epochs[i] * 0
                p = plt.plot(all_pred_epochs[i], plotted_e, linewidth=1, label=name)

            plt.xlabel('epochs')
            plt.ylabel(reduc + ' ' + error_name[4:])

            # Set limits for y axis
            #plt.ylim(0.55, 0.95)

            # Display legends and title
            plt.legend()

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all -------------------------------------------------------------------
    plt.show()

    return


def evolution_gifs(chosen_log):

    ############
    # Parameters
    ############

    # Load parameters
    config = Config()
    config.load(chosen_log)

    # Find all checkpoints in the chosen training folder
    chkp_path = join(chosen_log, 'checkpoints')
    chkps = np.sort([join(chkp_path, f) for f in listdir(chkp_path) if f[:4] == 'chkp'])

    # Get training and validation days
    val_path = join(chosen_log, 'val_preds')
    val_days = np.unique([f.split('_')[0] for f in listdir(val_path) if f.endswith('pots.ply')])

    # Util ops
    softmax = torch.nn.Softmax(1)
    sigmoid_2D = torch.nn.Sigmoid()
    fake_loss = FakeColliderLoss(config)

    # Result folder
    visu_path = join(config.saving_path, 'test_visu')
    if not exists(visu_path):
        makedirs(visu_path)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.
    config.augment_noise = 0
    config.augment_scale_min = 1.0
    config.augment_scale_max = 1.0
    config.augment_symmetries = [False, False, False]
    config.augment_rotation = 'none'
    config.validation_size = 100

    ##########################################
    # Choice of the image we want to visualize
    ##########################################

    # Dataset
    test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', balance_classes=False)

    wanted_inds = [700, 100, 150, 800]
    wanted_s_inds = [test_dataset.all_inds[ind][0] for ind in wanted_inds]
    wanted_f_inds = [test_dataset.all_inds[ind][1] for ind in wanted_inds]
    sf_to_i = {tuple(test_dataset.all_inds[ind]): i for i, ind in enumerate(wanted_inds)}

    ####################################
    # Preload to avoid long computations
    ####################################

    # List all precomputed preds:
    saved_preds = np.sort([f for f in listdir(visu_path) if f.endswith('.pkl')])
    saved_pred_inds = [int(f[:-4].split('_')[-1]) for f in saved_preds]

    # Load if available
    if np.all([ind in saved_pred_inds for ind in wanted_inds]):

        print('\nFound previous predictions, loading them')

        all_preds = []
        all_gts = []
        for ind in wanted_inds:
            wanted_ind_file = join(visu_path, 'preds_{:08d}.pkl'.format(ind))
            with open(wanted_ind_file, 'rb') as wfile:
                ind_preds, ind_gts = pickle.load(wfile)
            all_preds.append(ind_preds)
            all_gts.append(ind_gts)
        all_preds = np.stack(all_preds, axis=1)
        all_gts = np.stack(all_gts, axis=0)

    ########
    # Or ...
    ########

    else:

        ############
        # Choose GPU
        ############

        # Set which gpu is going to be used (auto for automatic choice)
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
            a = 1 / 0

        else:
            print('\nUsing GPU:', GPU_ID, '\n')

        # Set GPU visible device
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
        chosen_gpu = int(GPU_ID)

        ###########################
        # Initialize model and data
        ###########################

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

        # Choose to train on CPU or GPU
        if torch.cuda.is_available():
            device = torch.device("cuda:{:d}".format(chosen_gpu))
            net.to(device)
        else:
            device = torch.device("cpu")

        ######################################
        # Start predictions with ckpts weights
        ######################################

        all_preds = []
        all_gts = [None for _ in wanted_inds]

        for chkp_i, chkp in enumerate(chkps):

            # Load new checkpoint weights
            if torch.cuda.is_available():
                checkpoint = torch.load(chkp, map_location=device)
            else:
                checkpoint = torch.load(chkp, map_location=torch.device('cpu'))
            net.load_state_dict(checkpoint['model_state_dict'])
            epoch_i = checkpoint['epoch'] + 1
            net.eval()
            print("\nModel and training state restored from " + chkp)

            chkp_preds = [None for _ in wanted_inds]

            # Predict wanted inds with this chkp
            for i, batch in enumerate(test_loader):

                if 'cuda' in device.type:
                    batch.to(device)

                # Forward pass
                outputs, preds_init_2D, preds_2D = net(batch, config)

                # Get probs and labels
                f_inds = batch.frame_inds.cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                stck_init_preds = sigmoid_2D(preds_init_2D).cpu().detach().numpy()
                stck_future_logits = preds_2D.cpu().detach().numpy()
                stck_future_preds = sigmoid_2D(preds_2D).cpu().detach().numpy()
                stck_future_gts = batch.future_2D.cpu().detach().numpy()
                torch.cuda.synchronize(device)

                # Loop on batch
                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get the 2D predictions and gt (init_2D)
                    img0 = stck_init_preds[b_i, 0, :, :, :]
                    gt_im0 = np.copy(stck_future_gts[b_i, config.n_frames - 1, :, :, :])
                    gt_im1 = stck_future_gts[b_i, config.n_frames - 1, :, :, :]
                    gt_im1[:, :, 2] = np.max(stck_future_gts[b_i, :, :, :, 2], axis=0)
                    img1 = stck_init_preds[b_i, 1, :, :, :]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Get the 2D predictions and gt (prop_2D)
                    img = stck_future_preds[b_i, :, :, :, :]
                    gt_im = stck_future_gts[b_i, config.n_frames:, :, :, :]

                    # # Future errors defined the same as the loss
                    if sf_to_i[(s_ind, f_ind)] == 0:
                        future_errors_bce = fake_loss.apply(gt_im, stck_future_logits[b_i, :, :, :, :], error='bce')
                        a = 1/0
                    # future_errors = fake_loss.apply(gt_im, stck_future_logits[b_i, :, :, :, :], error='linear')
                    # future_errors = np.concatenate((future_errors_bce, future_errors), axis=0)

                    # # Save prediction too in gif format
                    # s_ind = f_inds[b_i, 0]
                    # f_ind = f_inds[b_i, 1]
                    # filename = '{:s}_{:07d}_e{:04d}.npy'.format(test_dataset.sequences[s_ind], f_ind, epoch_i)
                    # gifpath = join(config.saving_path, 'test_visu', filename)
                    # fast_save_future_anim(gifpath[:-4] + '_f_gt.gif', gt_im, zoom=5, correction=True)
                    # fast_save_future_anim(gifpath[:-4] + '_f_pre.gif', img, zoom=5, correction=True)

                    # Store all predictions
                    chkp_preds[sf_to_i[(s_ind, f_ind)]] = img
                    if chkp_i == 0:
                        all_gts[sf_to_i[(s_ind, f_ind)]] = gt_im

                    if np.all([chkp_pred is not None for chkp_pred in chkp_preds]):
                        break

                if np.all([chkp_pred is not None for chkp_pred in chkp_preds]):
                    break

            # Store all predictions
            chkp_preds = np.stack(chkp_preds, axis=0)
            all_preds.append(chkp_preds)

        # All predictions shape: [chkp_n, frames_n, T, H, W, 3]
        all_preds = np.stack(all_preds, axis=0)

        # All gts shape: [frames_n, T, H, W, 3]
        all_gts = np.stack(all_gts, axis=0)

        # Save each preds
        for ind_i, ind in enumerate(wanted_inds):
            wanted_ind_file = join(visu_path, 'preds_{:08d}.pkl'.format(ind))
            with open(wanted_ind_file, 'wb') as wfile:
                pickle.dump((all_preds[:, ind_i], all_gts[ind_i]), wfile)

    ################
    # Visualizations
    ################

    # First idea: future for different chkp
    idea1 = True
    if idea1:

        for frame_i, _ in enumerate(wanted_inds):

            # Colorize and zoom both preds and gts
            showed_preds = colorize_collisions(all_preds[:, frame_i])
            showed_preds = zoom_collisions(showed_preds, 5)
            showed_gts = colorize_collisions(all_gts[frame_i])
            showed_gts = zoom_collisions(showed_gts, 5)
            
            # Repeat gt for all checkpoints and merge with preds
            showed_gts = np.expand_dims(showed_gts, 0)
            showed_gts = np.tile(showed_gts, (showed_preds.shape[0], 1, 1, 1, 1))
            merged_imgs = superpose_gt(showed_preds, showed_gts)

            c_showed = [0, 5, 10, -1]
            n_showed = len(c_showed)

            fig, axes = plt.subplots(1, n_showed)
            images = []
            for ax_i, chkp_i in enumerate(c_showed):
                images.append(axes[ax_i].imshow(merged_imgs[chkp_i, 0]))

            def animate(i):
                for ax_i, chkp_i in enumerate(c_showed):
                    images[ax_i].set_array(merged_imgs[chkp_i, i])
                return images

            anim = FuncAnimation(fig, animate,
                                 frames=np.arange(merged_imgs.shape[1]),
                                 interval=50,
                                 blit=True)

            plt.show()

            # SAME BUT COMPARE MULTIPLE LOGS AT THE END OF THEIR CONFERGENCE

    # Second idea: evolution of prediction for different timestamps
    idea2 = False
    if idea2:

        for frame_i, _ in enumerate(wanted_inds):

            # Colorize and zoom both preds and gts
            showed_preds = colorize_collisions(all_preds[:, frame_i])
            showed_preds = zoom_collisions(showed_preds, 5)
            showed_gts = colorize_collisions(all_gts[frame_i])
            showed_gts = zoom_collisions(showed_gts, 5)

            # Repeat gt for all checkpoints and merge with preds
            showed_gts = np.expand_dims(showed_gts, 0)
            showed_gts = np.tile(showed_gts, (showed_preds.shape[0], 1, 1, 1, 1))
            merged_imgs = superpose_gt(showed_preds, showed_gts)

            t_showed = [2, 10, 18, 26]
            n_showed = len(t_showed)

            fig, axes = plt.subplots(1, n_showed)
            images = []
            for t, ax in zip(t_showed, axes):
                images.append(ax.imshow(merged_imgs[0, t]))

            # Add progress rectangles
            xy = (0.2 * merged_imgs.shape[-3], 0.015 * merged_imgs.shape[-2])
            dx = 0.6 * merged_imgs.shape[-3]
            dy = 0.025 * merged_imgs.shape[-2]
            rect1 = patches.Rectangle(xy, dx, dy, linewidth=1, edgecolor='white', facecolor='white')
            rect2 = patches.Rectangle(xy, dx * 0.01, dy, linewidth=1, edgecolor='white', facecolor='green')
            axes[0].add_patch(rect1)
            axes[0].add_patch(rect2)
            images.append(rect1)
            images.append(rect2)

            def animate(i):
                for t_i, t in enumerate(t_showed):
                    images[t_i].set_array(merged_imgs[i, t])
                progress = float(i + 1) / merged_imgs.shape[0]
                images[-1].set_width(dx * progress)
                return images

            n_gif = merged_imgs.shape[0]
            animation_frames = np.arange(n_gif)
            animation_frames = np.pad(animation_frames, 10, mode='edge')
            anim = FuncAnimation(fig, animate,
                                 frames=animation_frames,
                                 interval=100,
                                 blit=True)

            plt.show()

    # # Create superposition of gt and preds
    # r = preds[:, :, :, 0]
    # g = preds[:, :, :, 1]
    # b = preds[:, :, :, 2]
    # r[gt_mask] += 0
    # g[gt_mask] += 0
    # b[gt_mask] += 255

    # # Compute precision recall curves
    # figPR = show_PR(p, gt)

    # #fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # #anim0 = anim_multi_PR(p, gt, axis=axes[0])
    # #anim = show_future_anim(preds, axis=axes[1])
    # fig, double_anim = anim_PR_gif(preds, p, gt)

    # plt.show()

    a = 1 / 0

    return


def comparison_gifs(list_of_paths):

    ############
    # Parameters
    ############

    wanted_inds = [100, 150, 700, 800, 900]
    #wanted_inds = [100, 900]
    comparison_gts = []
    comparison_ingts = []
    comparison_preds = []

    for chosen_log in list_of_paths:

        ############
        # Parameters
        ############

        # Load parameters
        config = Config()
        config.load(chosen_log)

        # Find all checkpoints in the chosen training folder
        chkp_path = join(chosen_log, 'checkpoints')
        chkps = np.sort([join(chkp_path, f) for f in listdir(chkp_path) if f[:4] == 'chkp'])

        # Get training and validation days
        val_path = join(chosen_log, 'val_preds')
        val_days = np.unique([f.split('_')[0] for f in listdir(val_path) if f.endswith('pots.ply')])

        # Util ops
        softmax = torch.nn.Softmax(1)
        sigmoid_2D = torch.nn.Sigmoid()
        fake_loss = FakeColliderLoss(config)

        # Result folder
        visu_path = join(config.saving_path, 'test_visu')
        if not exists(visu_path):
            makedirs(visu_path)

        ####################################
        # Preload to avoid long computations
        ####################################

        # List all precomputed preds:
        saved_preds = np.sort([f for f in listdir(visu_path) if f.endswith('.pkl')])
        saved_pred_inds = [int(f[:-4].split('_')[-1]) for f in saved_preds]

        # Load if available
        if np.all([ind in saved_pred_inds for ind in wanted_inds]):

            print('\nFound previous predictions, loading them')

            all_preds = []
            all_gts = []
            all_ingts = []
            for ind in wanted_inds:
                wanted_ind_file = join(visu_path, 'preds_{:08d}.pkl'.format(ind))
                with open(wanted_ind_file, 'rb') as wfile:
                    ind_preds, ind_gts, ind_ingts = pickle.load(wfile)
                all_preds.append(np.copy(ind_preds))
                all_gts.append(np.copy(ind_gts))
                all_ingts.append(np.copy(ind_ingts))

            #print([ppp.shape for ppp in all_preds])
            all_preds = np.stack(all_preds, axis=1)
            all_gts = np.stack(all_gts, axis=0)
            all_ingts = np.stack(all_ingts, axis=0)

        ########
        # Or ...
        ########

        else:
            ############
            # Choose GPU
            ############

            # Set which gpu is going to be used (auto for automatic choice)
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
                a = 1 / 0

            else:
                print('\nUsing GPU:', GPU_ID, '\n')

            # Set GPU visible device
            os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
            chosen_gpu = int(GPU_ID)

            ##################################
            # Change model parameters for test
            ##################################

            # Change parameters for the test here. For example, you can stop augmenting the input data.
            config.augment_noise = 0
            config.augment_scale_min = 1.0
            config.augment_scale_max = 1.0
            config.augment_symmetries = [False, False, False]
            config.augment_rotation = 'none'
            config.validation_size = 100

            ##########################################
            # Choice of the image we want to visualize
            ##########################################

            # Dataset
            test_dataset = MyhalCollisionDataset(config, val_days, chosen_set='validation', balance_classes=False)

            wanted_s_inds = [test_dataset.all_inds[ind][0] for ind in wanted_inds]
            wanted_f_inds = [test_dataset.all_inds[ind][1] for ind in wanted_inds]
            sf_to_i = {tuple(test_dataset.all_inds[ind]): i for i, ind in enumerate(wanted_inds)}

            ###########################
            # Initialize model and data
            ###########################

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

            # Choose to train on CPU or GPU
            if torch.cuda.is_available():
                device = torch.device("cuda:{:d}".format(chosen_gpu))
                net.to(device)
            else:
                device = torch.device("cpu")

            ######################################
            # Start predictions with ckpts weights
            ######################################

            all_preds = []
            all_gts = [None for _ in wanted_inds]
            all_ingts = [None for _ in wanted_inds]

            for chkp_i, chkp in enumerate(chkps):

                # Load new checkpoint weights
                if torch.cuda.is_available():
                    checkpoint = torch.load(chkp, map_location=device)
                else:
                    checkpoint = torch.load(chkp, map_location=torch.device('cpu'))
                net.load_state_dict(checkpoint['model_state_dict'])
                epoch_i = checkpoint['epoch'] + 1
                net.eval()
                print("\nModel and training state restored from " + chkp)

                chkp_preds = [None for _ in wanted_inds]

                # Predict wanted inds with this chkp
                for i, batch in enumerate(test_loader):

                    if 'cuda' in device.type:
                        batch.to(device)

                    # Forward pass
                    outputs, preds_init_2D, preds_2D = net(batch, config)

                    # Get probs and labels
                    f_inds = batch.frame_inds.cpu().numpy()
                    lengths = batch.lengths[0].cpu().numpy()
                    stck_init_preds = sigmoid_2D(preds_init_2D).cpu().detach().numpy()
                    stck_future_logits = preds_2D.cpu().detach().numpy()
                    stck_future_preds = sigmoid_2D(preds_2D).cpu().detach().numpy()
                    stck_future_gts = batch.future_2D.cpu().detach().numpy()
                    torch.cuda.synchronize(device)

                    # Loop on batch
                    i0 = 0
                    for b_i, length in enumerate(lengths):

                        # Get the 2D predictions and gt (init_2D)
                        img0 = stck_init_preds[b_i, 0, :, :, :]
                        gt_im0 = np.copy(stck_future_gts[b_i, config.n_frames - 1, :, :, :])
                        gt_im1 = np.copy(stck_future_gts[b_i, config.n_frames - 1, :, :, :])
                        gt_im1[:, :, 2] = np.max(stck_future_gts[b_i, :, :, :, 2], axis=0)
                        img1 = stck_init_preds[b_i, 1, :, :, :]
                        s_ind = f_inds[b_i, 0]
                        f_ind = f_inds[b_i, 1]

                        # Get the 2D predictions and gt (prop_2D)
                        img = stck_future_preds[b_i, :, :, :, :]
                        gt_im = stck_future_gts[b_i, config.n_frames:, :, :, :]
                        
                        # Get the input frames gt
                        ingt_im = stck_future_gts[b_i, :config.n_frames, :, :, :]

                        # # Future errors defined the same as the loss
                        future_errors_bce = fake_loss.apply(gt_im, stck_future_logits[b_i, :, :, :, :], error='bce')
                        # future_errors = fake_loss.apply(gt_im, stck_future_logits[b_i, :, :, :, :], error='linear')
                        # future_errors = np.concatenate((future_errors_bce, future_errors), axis=0)

                        # # Save prediction too in gif format
                        # s_ind = f_inds[b_i, 0]
                        # f_ind = f_inds[b_i, 1]
                        # filename = '{:s}_{:07d}_e{:04d}.npy'.format(test_dataset.sequences[s_ind], f_ind, epoch_i)
                        # gifpath = join(config.saving_path, 'test_visu', filename)
                        # fast_save_future_anim(gifpath[:-4] + '_f_gt.gif', gt_im, zoom=5, correction=True)
                        # fast_save_future_anim(gifpath[:-4] + '_f_pre.gif', img, zoom=5, correction=True)

                        # Store all predictions
                        chkp_preds[sf_to_i[(s_ind, f_ind)]] = img
                        if chkp_i == 0:
                            all_gts[sf_to_i[(s_ind, f_ind)]] = gt_im
                            all_ingts[sf_to_i[(s_ind, f_ind)]] = ingt_im

                        if np.all([chkp_pred is not None for chkp_pred in chkp_preds]):
                            break

                    if np.all([chkp_pred is not None for chkp_pred in chkp_preds]):
                        break

                # Store all predictions
                chkp_preds = np.stack(chkp_preds, axis=0)
                all_preds.append(chkp_preds)

            # All predictions shape: [chkp_n, frames_n, T, H, W, 3]
            all_preds = np.stack(all_preds, axis=0)

            # All gts shape: [frames_n, T, H, W, 3]
            all_gts = np.stack(all_gts, axis=0)
            all_ingts = np.stack(all_ingts, axis=0)

            # Save each preds
            for ind_i, ind in enumerate(wanted_inds):
                wanted_ind_file = join(visu_path, 'preds_{:08d}.pkl'.format(ind))
                with open(wanted_ind_file, 'wb') as wfile:
                    pickle.dump((np.copy(all_preds[:, ind_i]),
                                 np.copy(all_gts[ind_i]),
                                 np.copy(all_ingts[ind_i])), wfile)


        comparison_preds.append(all_preds)
        comparison_gts.append(all_gts)
        comparison_ingts.append(all_ingts)

    # All predictions shape: [log_n, chkp_n, frames_n, T, H, W, 3]
    comparison_preds = np.stack([cp[-1] for cp in comparison_preds], axis=0)


    # All gts shape: [frames_n, T, H, W, 3]
    comparison_gts = comparison_gts[0]
    comparison_ingts = comparison_ingts[0]


    ################
    # Visualizations
    ################

    for frame_i, w_i in enumerate(wanted_inds):

        # Colorize and zoom both preds and gts
        showed_preds = zoom_collisions(comparison_preds[:, frame_i], 5)
        showed_gts = zoom_collisions(comparison_gts[frame_i], 5)
        showed_ingts = zoom_collisions(comparison_ingts[frame_i], 5)

        # Repeat gt for all checkpoints and merge with preds
        showed_gts = np.expand_dims(showed_gts, 0)
        showed_gts = np.tile(showed_gts, (showed_preds.shape[0], 1, 1, 1, 1))
        showed_ingts = np.expand_dims(showed_ingts, 0)
        showed_ingts = np.tile(showed_ingts, (showed_preds.shape[0], 1, 1, 1, 1))

        # Merge colors
        merged_imgs = superpose_gt(showed_preds, showed_gts, showed_ingts)

        c_showed = np.arange(showed_preds.shape[0])
        n_showed = len(c_showed)

        fig, axes = plt.subplots(1, n_showed)
        images = []
        for ax_i, log_i in enumerate(c_showed):

            # Init plt
            images.append(axes[ax_i].imshow(merged_imgs[log_i, 0]))

            # Save
            #imageio.mimsave('results/test_{:05d}_{:d}.gif'.format(w_i, ax_i+1), merged_imgs[log_i], fps=20)

        def animate(i):
            for ax_i, log_i in enumerate(c_showed):
                images[ax_i].set_array(merged_imgs[log_i, i])
            return images

        anim = FuncAnimation(fig, animate,
                                frames=np.arange(merged_imgs.shape[1]),
                                interval=50,
                                blit=True)



        plt.show()


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
    end = 'Log_2021-04-21_15-05-54'

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
                  'Bouncer-Loss=1/2-NEW_METRIC',
                  'test'
                  ]

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def collider_tests_2(old_result_limit):
    """
    Nouveau loss, et nouvelles valeur de validation pour ces tests
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-04-21_15-05-54'
    end = 'Log_2021-04-30_11-07-40'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['Bouncer-Loss=1/2-v2',
                  'Bouncer-Loss=1/2-v0',
                  'Bouncer-Loss=1/2-v1',
                  'Bouncer-5frames-failed',
                  'Bouncer-5frames-Loss=1/8-v2',
                  ]

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def collider_tests_Bouncers(old_result_limit):
    """
    Nouveau loss, et nouvelles valeur de validation pour ces tests
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-04-30_11-07-41'
    end = 'Log_2021-05-02_10-19-22'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['Bouncer-5frames',
                  'Bouncer-3frames']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def collider_tests_Wanderers(old_result_limit):
    """
    Nouveau loss, et nouvelles valeur de validation pour ces tests
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2021-05-03_17-57-57'
    end = 'Log_2021-05-07_10-19-22'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, log) for log in listdir(res_path) if start <= log <= end])
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['Wand-indep',
                  'Wand-shared',
                  'Wand-indep-5frames',
                  'Wand-indep-d120']

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
    logs, logs_names = collider_tests_Wanderers(old_res_lim)
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
    # ##############

    gifs = True

    if gifs == True:

        # # Evolution of predictions from checkpoints to checkpoints
        # evolution_gifs(logs[1])

        # Comparison of last checkpoints of each logs
        comparison_gifs(logs[:1])


    else:

        # Plot the training loss and accuracy
        compare_trainings(logs, logs_names, smooth_epochs=3.0)

        # Plot the validation
        if config.dataset_task == 'collision_prediction':
            if config.dataset.startswith('MyhalCollision'):
                compare_convergences_collision2D(logs, logs_names, smooth_n=20)
        else:
            raise ValueError('Unsupported dataset : ' + plot_dataset)
