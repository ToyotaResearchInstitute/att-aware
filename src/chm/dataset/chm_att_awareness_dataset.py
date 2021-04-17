# Copyright 2020 Toyota Research Institute.  All rights reserved.
import os
import cv2
import tqdm
import numpy as np
import os
import csv
import collections
import hashlib
import pickle
import importlib
import subprocess
import shutil
import copy
import json
import collections
import bisect
import itertools

from torch.utils.data import Dataset
from PIL import Image
from collections import OrderedDict
from utils.chm_consts import *
from chm_base_dataset import CognitiveHeatMapBaseDataset


class CognitiveHeatMapAttAWarenessDataset(CognitiveHeatMapBaseDataset):
    def __init__(self, data_dir=None, precache_dir=None, dataset_type=None, params_dict=None):
        """
        CognitiveHeatMapAttAWarenessDataset dataset class

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the DREYEVE VIDEOS and gaze data
        precache_dir : str
            Path to the directory containing image (video frames, segmentations, optic flow) caches
        dataset_type : str {'train, 'test', 'vis'}
            String indicating the type of dataset
        params_dict : dict
            Dictionary containing the args passed from the training script
        """
        super().__init__(
            data_dir=data_dir, precache_dir=precache_dir, dataset_type=dataset_type, params_dict=params_dict
        )
        self.att_awareness_labels_csv_path = self.params_dict.get("att_awareness_labels", None)
        assert (
            self.att_awareness_labels_csv_path is not None
        ), "Please provide the full path to the awareness labels csv file"

    def _create_metadata_tuple_list(self):
        """
        Generates the metadata list for this class. The function is called at the very end of the CognitiveHeatmapBaseDataset init function

        Parameters
        ----------
        None

        Returns
        -------
        None. Results in populating the self.metadata_list
        """
        pass

    def __getitem__(self, idx):
        """
        Required getitem() for PyTorch dataset.

        Parameters
        ----------
        idx: Index of the data item in self.metadata_list

        Returns
        -------
        data_dict: Ordered dictionary containing the various data items needed for training. Each item in the dict is a tensor or numpy.array

        auxiliary_info_list: List of auxiliary information needed for other purposes. Only returned if auxiliary info flag is set to be True.
        pass
