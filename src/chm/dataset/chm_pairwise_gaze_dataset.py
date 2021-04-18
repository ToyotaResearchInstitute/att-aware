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


class CognitiveHeatMapPairwiseGazeDataset(CognitiveHeatMapBaseDataset):
    def __init__(self, data_dir=None, precache_dir=None, dataset_type=None, params_dict=None):
        """
        CognitiveHeatMapGazeDataset dataset class

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

    def _setup_resources(self):
        """
        Sets up any resources (such loading csv files etc) needed for this derived Dataset.

        Parameters
        ----------
        None

        Returns
        -------
        None.
        """
        pass

    def _create_metadata_tuple_list(self):
        """
        Initializes the metadata_len and metadata_list if needed. The function is called at the very end of the CognitiveHeatmapBaseDataset init function

        Parameters
        ----------
        None

        Returns
        -------
        None. Only Results in populating the self.metadata_list
        """

        metadata_list_all_comb = list(
            itertools.product(self.sequence_ids, self.subject_ids, self.task_ids)
        )  # create all combinations of video, subject, task tuples for the specified video, subject, task args
        metadata_list_all_comb = [
            d for d in metadata_list_all_comb if d in self.all_videos_subject_task_list
        ]  # filter out those combinations that are not present in the available combinations

        self.metadata_list = [
            (a, b, b + 1) for a in metadata_list_all_comb for b in self.query_frame_idxs_list[:-1]
        ]  # append the query_frame at t and t+1 to each video, subject, task tuple

        self.metadata_len = len(self.metadata_list)  # Total number of available snippets

    def __getitem__(self, idx):
        """
        Required getitem() for PyTorch gaze dataset.

        Parameters
        ----------
        idx: Index of the data item in self.metadata_list

        Returns
        -------
        data_dict: dict
        Dictionary containing the data_dict, auxiliary_info for sequence at t and at t+1
        """
        data_dict = {}
        (video_id, subject, task), query_frame_t, query_frame_tp1 = self.metadata_list[idx]
        data_dict_t, auxiliary_info_list_t = self._get_sequence(
            video_id, subject, task, query_frame_t
        )  # get gaze data for sequence at t
        data_dict_tp1, auxiliary_info_list_tp1 = self._get_sequence(
            video_id, subject, task, query_frame_tp1
        )  # get gaze data for sequence at t+1

        data_dict["data_t"] = (data_dict_t, auxiliary_info_list_t)
        data_dict["data_tp1"] = (data_dict_tp1, auxiliary_info_list_tp1)
        return data_dict