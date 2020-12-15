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
import signal
import os
import numpy as np
import sys
import torch
import time

# Dataset
from slam.PointMapSLAM import pointmap_slam, detect_short_term_movables, annotation_process
from slam.dev_slam import bundle_slam, pointmap_for_AMCL
from torch.utils.data import DataLoader
from datasets.MyhalSim import MyhalSimDataset, MyhalSimSlam, MyhalSimSampler, MyhalSimCollate

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN

from os.path import exists, join
from os import makedirs


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###################
    # Training sessions
    ###################

    # Day used as map
    map_day = '2020-10-02-13-39-05'

    # Third dataset
    train_days_0 = ['2020-11-05-05-17-48',
                    '2020-11-05-17-24-43',
                    '2020-11-05-17-57-45',
                    '2020-11-05-18-34-15',
                    '2020-11-05-20-28-03',
                    '2020-11-05-17-10-47',
                    '2020-11-05-17-45-12',
                    '2020-11-05-18-17-53',
                    '2020-11-05-19-08-29',
                    '2020-11-05-22-09-11']

    train_days_0 = ['2020-11-05-19-08-29',
                    '2020-11-05-22-09-11']

    ######################
    # Automatic Annotation
    ######################

    # Choose the dataset
    train_days = train_days_0

    # Check if we need to redo annotation (only if there is no video)
    redo_annot = False
    for day in train_days:
        annot_path = join('../../Myhal_Simulation/annotated_frames', day)
        if not exists(annot_path):
            redo_annot = True
            break

    redo_annot = True
    if redo_annot:

        # Initiate dataset
        slam_dataset = MyhalSimSlam(day_list=train_days, map_day=map_day)
        #slam_dataset = MyhalSimDataset(first_day='2020-06-24-14-36-49', last_day='2020-06-24-14-40-33')

        # Create a refined map from the map_day
        slam_dataset.refine_map()

        # Groundtruth mapping
        # slam_dataset.debug_angular_velocity()
        # slam_dataset.gt_mapping() #TODO: can you add all frames at onec in this function?

        # Groundtruth annotation
        #annotation_process(slam_dataset, on_gt=True)

        # SLAM mapping
        # slam_dataset.pointmap_slam()

        # Groundtruth annotation
        annotation_process(slam_dataset, on_gt=False)

        # TODO: Loop closure for aligning days together when niot simulation
        # TODO: Verify mapping does not fail anymore
        #       Better annotation: think of adding more and more days, and the effect it will have on annot
        #           > 1. Use only 1 or 2 reference days to annotate the longterm movables
        #           > 2. Better still detection. For now some wall are uncertain and then classified into longterm,
        #                maybe because we use too many days combinerd to annotated walls????

        a = 1/0
