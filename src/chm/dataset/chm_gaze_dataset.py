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


class CognitiveHeatMapGazeDataset(CognitiveHeatMapBaseDataset):
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

    def _create_metadata_tuple_list(self):
        """
        Generates the metadata list for this class. The function is called at the very end of the CognitiveHeatmapBaseDataset init function

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
            (a, b) for a in metadata_list_all_comb for b in self.query_frame_idxs_list
        ]  # append the frame query list to each tuple

        self.metadata_len = len(self.metadata_list)  # Total number of available snippets

    def __getitem__(self, idx):
        """
        Required getitem() for PyTorch gaze dataset.

        Parameters
        ----------
        idx: Index of the data item in self.metadata_list

        Returns
        -------
        data_dict: Ordered dictionary containing the various data items needed for training. Each item in the dict is a tensor or numpy.array

        auxiliary_info_list: List of auxiliary information needed for other purposes. Only returned if auxiliary info flag is set to be True.
        """
        (video_id, subject, task), query_frame = self.metadata_list[idx]
        data_item_query_framelist = [
            self.temporal_downsample_factor * (i) + query_frame - self.first_query_frame
            for i in range(self.sequence_length)
        ]  # list of frame idxs to be used for the snippet

        data_item_list = []
        auxiliary_info_list = []

        for query_frame_id_t in data_item_query_framelist:
            metadata = (video_id, subject, task, query_frame_id_t)
            data_item_t, auxiliary_info_t = self._get_single_item(metadata)
            # append the data at each frame at time t to the list
            data_item_list.append(data_item_t)
            auxiliary_info_list.append(auxiliary_info_t)

        # convert the data list into a dictionary for cleaner parsing by the dataloader later in the training loop
        data_dict = collections.OrderedDict()
        try:
            import IPython

            IPython.embed(banner1="check")
            for key in data_item_list[0]:
                data_dict[key] = []
                for data_item in data_item_list:  # iterate over the list containing the sequence of data_items
                    data_item[key].append(np.expand_dims(data_item[key], axis=0))

                data_dict[key] = np.concatenate(data_dict[key], axis=0)
        except Exception as e:
            import IPython

            IPython.embed(
                header="getitem invalid: " + str(e)
            )  # Embed to catch the exception if something goes wrong in the conversion of the data_item_list to the data dict

        import IPython

        IPython.embed(banner1="check getitem")
