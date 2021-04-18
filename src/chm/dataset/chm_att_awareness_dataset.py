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
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
from collections import OrderedDict
from utils.chm_consts import *
from chm_base_dataset import CognitiveHeatMapBaseDataset


class CognitiveHeatMapAttAwarenessDataset(CognitiveHeatMapBaseDataset):
    def __init__(self, data_dir=None, precache_dir=None, dataset_type=None, params_dict=None):
        """
        CognitiveHeatMapAttAwarenessDataset dataset class

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
        self.att_awareness_labels_csv_path = self.params_dict.get("att_awareness_labels", None)
        assert (
            self.att_awareness_labels_csv_path is not None
        ), "Please provide the full path to the awareness labels csv file"
        df = pd.read_csv(
            self.att_awareness_labels_csv_path, delimiter=","
        )  # read in the att awareness as a pandas dataframe

        self.att_awareness_labels_unfiltered = copy.deepcopy(df)

        # filter dataframe according to what video_ids, subject and task were requested for the dataset
        df_filtered = df[df["video_id"].isin(self.sequence_ids)]
        df_filtered = df_filtered[df_filtered["cognitive_modifier"].isin(self.task_ids)]
        df_filtered = df_filtered[df_filtered["subject"].isin(self.subject_ids)]

        self.att_awareness_labels = copy.deepcopy(df_filtered)

    def _create_metadata_tuple_list(self):
        """
        Initializes the metadata_len and metadata_list if needed. The function is called at the very end of the CognitiveHeatmapBaseDataset init function

        Parameters
        ----------
        None

        Returns
        -------
        None. Results in populating the self.metadata_list
        """
        self.metadata_len = self.att_awareness_labels.shape[0]  # number of rows in the filtered data frame

    def __getitem__(self, idx):
        """
        Required getitem() for PyTorch dataset.

        Parameters
        ----------
        idx: Index of the data item in self.att_awareness_labels

        Returns
        -------
        data_dict: Ordered dictionary containing the various data items needed for training.

        auxiliary_info_list: List of auxiliary information needed for other purposes. Only returned if auxiliary info flag is set to be True.
        """

        att_label_item = self.att_awareness_labels.iloc[idx]

        video_id = att_label_item["video_id"]
        subject = att_label_item["subject"]
        task = att_label_item["cognitive_modifier"]
        query_frame = att_label_item["query_frame"]

        data_dict, auxiliary_info_dict = self._get_sequence(video_id, subject, task, query_frame)  # get gaze info

        # append annotation info to the data_dict dictionary
        annotation_dict = att_label_item.to_dict()
        data_dict["att_annotation"] = annotation_dict
        return data_dict, auxiliary_info_dict
