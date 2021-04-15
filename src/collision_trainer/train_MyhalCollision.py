#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on MyhalCollision dataset
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
import sys
import time
import signal
import os
import numpy as np
import torch


# Dataset
from slam.PointMapSLAM import pointmap_slam, detect_short_term_movables, annotation_process
from slam.dev_slam import bundle_slam, pointmap_for_AMCL
from torch.utils.data import DataLoader
from datasets.MyhalCollision import MyhalCollisionDataset, MyhalCollisionSlam, MyhalCollisionSampler, \
    MyhalCollisionCollate

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPCollider

from os.path import exists, join
from os import makedirs


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class MyhalCollisionConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'MyhalCollision'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 16

    #########################
    # Architecture definition
    #########################

    # Define layers (only concerning the 3D architecture)
    architecture = ['simple',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    ######################
    # Collision parameters
    ######################

    # Number of propagating layer
    n_2D_layers = 30

    # Total time propagated
    T_2D = 3.0

    # Size of 2D convolution grid
    dl_2D = 0.12

    # Power of the loss for the 2d predictions (use smaller prop loss when shared weights)
    power_2D_init_loss = 0.5
    power_2D_prop_loss = 4.0
    neg_pos_ratio = 4.0

    # Specification of the 2D networks composition
    init_2D_levels = 3
    init_2D_resnets = 3
    prop_2D_resnets = 2

    # Path to a pretrained 3D network. if empty, ignore, if 'todo', then only train 3D part of the network.
    #pretrained_3D = 'Log_2021-01-27_18-53-05'
    pretrained_3D = ''

    # Detach the 2D network from the 3D network when backpropagating gradient
    detach_2D = False

    # Share weights for 2D network TODO: see if not sharing makes a difference
    shared_2D = False

    # Trainable backend 3D network
    apply_3D_loss = True
    #frozen_layers = ['encoder_blocks', 'decoder_blocks', 'head_mlp', 'head_softmax']

    # Use visibility mask for training
    use_visibility = False

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 8.0
    val_radius = 8.0
    n_frames = 3
    max_in_points = -1
    max_val_points = -1

    # Number of batch
    batch_num = 5
    val_batch_num = 1

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.06

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 100
    in_features_dim = 3

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 500

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 30

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = False
    augment_symmetries = [False, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_color = 1.0

    # Do we nee to save convergence
    saving = True
    saving_path = None


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    # NOT_NOW_TODO: Optimize online predictions
    #           > Try to parallelise the batch preprocessing for a single input frame.
    #           > Use OMP for neighbors processing
    #           > Use the polar coordinates to get neighbors???? (avoiding tree building time)
    #           > cpp extension for conversion into a 2D lidar_range_scan
    #

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###################
    # Training sessions
    ###################

    # Day used as map
    map_day = '2020-10-02-13-39-05'

    # Fisrt dataset: successful tours without filtering Initiate dataset. Remember last day is used as validation for
    # the training
    train_days_0 = ['2020-10-12-22-06-54',
                    '2020-10-12-22-14-48',
                    '2020-10-12-22-28-15']

    # Second dataset
    train_days_1 = ['2020-10-12-22-06-54',
                    '2020-10-12-22-14-48',
                    '2020-10-12-22-28-15',
                    '2020-10-16-12-29-11',
                    '2020-10-16-12-37-53',
                    '2020-10-16-12-50-41',
                    '2020-10-16-13-06-53',
                    '2020-10-16-13-20-04',
                    '2020-10-16-13-38-50',
                    '2020-10-16-14-01-49',
                    '2020-10-16-14-36-12',
                    '2020-10-16-14-56-40']

    train_days_RandWand = ['2021-03-19-22-58-27',
                           '2021-03-19-23-07-18',
                           '2021-03-19-23-26-42',
                           '2021-03-19-23-47-49',
                           '2021-03-19-23-54-55',
                           '2021-03-20-00-12-29',
                           '2021-03-20-00-25-03',
                           '2021-03-20-00-38-13',
                           '2021-03-20-00-50-36',
                           '2021-03-20-01-12-54',
                           '2021-03-20-01-21-43',
                           '2021-03-20-01-33-11',
                           '2021-03-20-02-01-40',
                           '2021-03-20-02-09-17',
                           '2021-03-20-02-26-40']

    # train_days_RandFlow_old = ['2021-03-23-18-25-53',
    #                            '2021-03-23-18-46-44',
    #                            '2021-03-23-19-00-19',
    #                            '2021-03-23-19-10-40',
    #                            '2021-03-23-19-24-04',
    #                            '2021-03-23-19-30-09',
    #                            '2021-03-23-19-41-16',
    #                            '2021-03-23-19-52-42',
    #                            '2021-03-23-20-05-42',
    #                            '2021-03-23-20-12-16',
    #                            '2021-03-23-20-24-52',
    #                            '2021-03-23-20-35-45',
    #                            '2021-03-23-20-44-11',
    #                            '2021-03-23-20-57-59']

    # train_days_RandFlow = ['2021-03-25-19-32-51',
    #                        '2021-03-25-19-51-21',
    #                        '2021-03-25-20-16-26',
    #                        '2021-03-25-21-02-40',
    #                        '2021-03-25-21-17-42',
    #                        '2021-03-25-21-39-19',
    #                        '2021-03-25-22-08-50',
    #                        '2021-03-25-22-20-56',
    #                        '2021-03-25-22-51-18',
    #                        '2021-03-25-23-31-49',
    #                        '2021-03-25-23-50-48',
    #                        '2021-03-26-00-17-18',
    #                        '2021-03-26-00-49-57',
    #                        '2021-03-26-01-09-11',
    #                        '2021-03-26-01-41-16']

    # train_days_flow = ['2021-03-16-17-44-40',
    #                    '2021-03-16-17-49-39',
    #                    '2021-03-16-18-05-16',
    #                    '2021-03-16-18-10-53',
    #                    '2021-03-16-18-28-46',
    #                    '2021-03-16-19-01-26',
    #                    '2021-03-16-19-12-29',
    #                    '2021-03-16-19-26-10',
    #                    '2021-03-16-19-46-05',
    #                    '2021-03-16-20-06-30']

                       
    train_days_RandFlow = ['2021-03-26-16-53-44',
                           '2021-03-26-17-22-31',
                           '2021-03-26-18-02-18',
                           '2021-03-26-18-58-03',
                           '2021-03-26-19-22-03',
                           '2021-03-26-20-09-03',
                           '2021-03-26-21-07-18',
                           '2021-03-26-21-34-51',
                           '2021-03-26-22-20-38']

    train_days_RandBounce = ['2021-04-12-15-10-19',
                             '2021-04-12-15-34-52',
                             '2021-04-12-16-22-40',
                             '2021-04-12-17-20-23',
                             '2021-04-12-17-42-45',
                             '2021-04-12-18-25-47',
                             '2021-04-12-19-08-58',
                             '2021-04-12-19-32-40',
                             '2021-04-12-20-10-47',
                             '2021-04-12-21-13-04',
                             '2021-04-12-21-39-16',
                             '2021-04-12-22-13-09',
                             '2021-04-12-22-59-57',
                             '2021-04-12-23-28-00']

    ######################
    # Automatic Annotation
    ######################

    # Choose the dataset
    train_days = np.array(train_days_RandBounce)
    val_inds = [0, 1, 2]
    train_inds = [i for i in range(len(train_days)) if i not in val_inds]

    # Check if we need to redo annotation (only if there is no video)
    redo_annot = False
    for day in train_days:
        annot_path = join('../../../Myhal_Simulation/collisions', day)
        if not exists(annot_path):
            redo_annot = True
            break

    # train_days = ['2020-10-20-16-30-49']
    #redo_annot = True
    if redo_annot:

        # Initiate dataset
        slam_dataset = MyhalCollisionSlam(day_list=train_days, map_day=map_day)
        #slam_dataset = MyhalCollisionDataset(first_day='2020-06-24-14-36-49', last_day='2020-06-24-14-40-33')

        # Create a refined map from the map_day
        # slam_dataset.refine_map()

        # Groundtruth mapping
        #slam_dataset.debug_angular_velocity()
        #slam_dataset.gt_mapping() # can you add all frames at onec in this function?

        # Groundtruth annotation
        #annotation_process(slam_dataset, on_gt=True)

        # SLAM mapping
        #slam_dataset.pointmap_slam()

        # Groundtruth annotation
        annotation_process(slam_dataset, on_gt=False)

        # TODO: Loop closure for aligning days together when not simulation

        slam_dataset.collision_annotation()

        a = 1 / 0

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = MyhalCollisionConfig()
  
    # Override with configuration from previous 3D network if given
    if config.pretrained_3D and config.pretrained_3D != 'todo':

        # Check if path exists
        previous_path = os.path.join('results', config.pretrained_3D)
        if not exists(previous_path):
            raise ValueError('Given path for previous 3D network does not exist')
        
        # Load config
        prev_config = MyhalCollisionConfig()
        prev_config.load(previous_path)

        # List of params we should not overwrite:
        kept_params = ['n_2D_layers',
                       'T_2D',
                       'dl_2D',
                       'power_2D_init_loss',
                       'power_2D_prop_loss',
                       'neg_pos_ratio',
                       'init_2D_levels',
                       'init_2D_resnets',
                       'prop_2D_resnets',
                       'pretrained_3D',
                       'detach_2D',
                       'shared_2D',
                       'apply_3D_loss',
                       'frozen_layers',
                       'max_epoch',
                       'learning_rate',
                       'momentum',
                       'lr_decays',
                       'grad_clip_norm',
                       'epoch_steps',
                       'validation_size',
                       'checkpoint_gap',
                       'saving',
                       'saving_path',
                       'input_threads']
        
        for attr_name, attr_value in vars(config).items():
            if attr_name not in kept_params:
                setattr(config, attr_name, getattr(prev_config, attr_name))


    # Get path from argument if given
    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]

    ###############
    # Previous chkp
    ###############
    # Choose here if you want to start training from a previous snapshot (None for new training)

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None

    chosen_chkp = None
    if config.pretrained_3D and config.pretrained_3D != 'todo':

        # Check if path exists
        chkp_path = os.path.join('results', config.pretrained_3D, 'checkpoints')
        if not exists(chkp_path):
            raise ValueError('Given path for previous 3D network does contain any checkpoints')

        # Find all snapshot in the chosen training folder
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', config.pretrained_3D, 'checkpoints', chosen_chkp)

    #####################
    # Init input pipeline
    #####################

    # Initialize datasets (dummy validation)
    training_dataset = MyhalCollisionDataset(config, train_days[train_inds], chosen_set='training', balance_classes=True)
    test_dataset = MyhalCollisionDataset(config, train_days[val_inds], chosen_set='validation', balance_classes=False)

    # Initialize samplers
    training_sampler = MyhalCollisionSampler(training_dataset)
    test_sampler = MyhalCollisionSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=MyhalCollisionCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=MyhalCollisionCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate max_in_point value
    if config.max_in_points < 0:
        config.max_in_points = 1e9
        training_loader.dataset.max_in_p = 1e9
        training_sampler.calib_max_in(config, training_loader, untouched_ratio=0.9, verbose=True)
    if config.max_val_points < 0:
        config.max_val_points = 1e9
        test_loader.dataset.max_in_p = 1e9
        test_sampler.calib_max_in(config, test_loader, untouched_ratio=0.95, verbose=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_class_w(training_dataset, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPCollider(config, training_dataset.label_values, training_dataset.ignored_labels)

    debug = False
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    # Freeze layers if necessary
    if config.frozen_layers:
        for name, child in net.named_children():
            if name in config.frozen_layers:
                for param in child.parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                child.eval()


    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    trainer.train(net, training_loader, test_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)
