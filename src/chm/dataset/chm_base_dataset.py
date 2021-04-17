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
import json
import collections
import bisect
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
from collections import OrderedDict
from utils.chm_consts import *


class CognitiveHeatMapBaseDataset(Dataset):
    def __init__(self, data_dir=None, precache_dir=None, dataset_type=None, params_dict=None):
        """
        CognitiveHeatMapBaseDataset dataset class. Base class for all the other Dataset classes used for CHM.
        Implements all the getters for caches and loads up the pandas dataframe with all gaze information

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the DREYEVE VIDEOS and gaze data
        precache_dir : str
            Path to the directory containing image (video frames, segmentations, optic flow) caches
        params_dict : dict
            Dictionary containing the args passed from the training script

        """
        # Asserts
        assert (
            params_dict is not None
        ), "Params Dict should be passed. Contains all args from parse_arguments in args_file"
        assert (
            data_dir is not None
        ), "Data dir has to be a valid directory path to the directory containing the DREYEVE_VIDEOS and gaze data"
        assert (
            precache_dir is not None
        ), "Precache dir has to be a valid directory path to the directory containing the image (video frames, segmentations, optic_flow) caches"

        assert dataset_type is not None and dataset_type in [
            "train",
            "test",
            "vis",
        ], "Dataset type has to be a string and has to either [train, test, vis]"

        self.data_dir = data_dir
        self.precache_dir = precache_dir
        self.dataset_type = dataset_type
        self.params_dict = params_dict
        self.metadata_list = None
        self.metadata_len = None

        self.aspect_ratio_reduction_factor = self.params_dict.get(
            "aspect_ratio_reduction_factor", 8.0
        )  # Factor by which the full sized image needs to be rescaled before used by network for training
        self.temporal_downsample_factor = self.params_dict.get(
            "temporal_downsample_factor", 6
        )  # Hopsize for downsampled sequences

        # Depending on the dataset type grab the corresponding
        # sequence length, sequence id, subject id and task id from the params dict
        self.sequence_length = self.params_dict.get("{}_sequence_length".format(self.dataset_type), 20)
        self.sequence_ids = self.params_dict.get("{}_sequence_ids".format(self.dataset_type))
        self.subject_ids = self.params_dict.get("{}_subject_ids".format(self.dataset_type))
        self.task_ids = self.params_dict.get(
            "{}_task_ids".format(self.dataset_type)
        )  # cognitive-task-modifiers (referred to as task in the code)

        self.first_query_frame = (self.sequence_length - 1) * self.temporal_downsample_factor
        self.query_frame_idxs_list = list(
            range(self.first_query_frame, MAX_NUM_VIDEO_FRAMES)
        )  # list of query frames used for each video. Each query frame idx corresponds to the last frame of the snippet used.

        self.ORIG_ROAD_IMG_DIMS = self.params_dict.get("orig_road_img_dims")
        self.ORIG_ROAD_IMAGE_HEIGHT = self.ORIG_ROAD_IMG_DIMS[1]
        self.ORIG_ROAD_IMAGE_WIDTH = self.ORIG_ROAD_IMG_DIMS[2]

        self.return_reduced_size = (
            False if self.aspect_ratio_reduction_factor == 1.0 else True
        )  # Flag which ensure that the scaled version of the cached images are returned from the fetch functions when aspect_ratio is smaller.

        self.new_image_width = int(round(self.ORIG_ROAD_IMAGE_WIDTH / self.aspect_ratio_reduction_factor))
        self.new_image_height = int(round(self.ORIG_ROAD_IMAGE_HEIGHT / self.aspect_ratio_reduction_factor))

        assert os.path.exists(
            os.path.join(self.data_dir, "DREYEVE_DATA")
        ), "Dreyeve video data not present at the data directory."

        self.all_videos_subject_task_list = []
        self.all_videos_subjects_tasks_gaze_data_dict_path = os.path.join(
            self.precache_dir, "all_videos_subjects_tasks_gaze_data_dict.pkl"
        )
        if not os.path.exists(self.all_videos_subjects_tasks_gaze_data_dict_path):
            self.all_videos_subjects_tasks_gaze_data_dict = collections.OrderedDict()

            for video_id in ALL_DREYEVE_VIDEO_IDS:
                self.all_videos_subjects_tasks_gaze_data_dict[video_id] = collections.OrderedDict()
                gaze_data_folder = os.path.join(
                    self.data_dir, "DREYEVE_DATA", "{0:02d}".format(video_id), "eyelink_data"
                )
                for subject_task_gaze_file in os.listdir(gaze_data_folder):
                    subject = int(
                        subject_task_gaze_file.split("_")[0][1:]
                    )  # for example extract int(017) from 's017_blurred.txt'
                    task = subject_task_gaze_file.split("_")[1][
                        :-4
                    ]  # for example extract 'blurred' from 's017_blurred.txt'
                    print("Extracting ", video_id, subject, task)
                    if subject not in self.all_videos_subjects_tasks_gaze_data_dict[video_id]:
                        self.all_videos_subjects_tasks_gaze_data_dict[video_id][subject] = collections.OrderedDict()
                    if task not in self.all_videos_subjects_tasks_gaze_data_dict[video_id][subject]:
                        self.all_videos_subjects_tasks_gaze_data_dict[video_id][subject][
                            task
                        ] = collections.OrderedDict()
                        self.all_videos_subject_task_list.append((video_id, subject, task))

                    self.all_videos_subjects_tasks_gaze_data_dict[video_id][subject][task] = pd.read_csv(
                        os.path.join(gaze_data_folder, subject_task_gaze_file), sep=" "
                    )  # extract the gaze data from the txt file as a pandas data frame

            with open(self.all_videos_subjects_tasks_gaze_data_dict_path, "wb") as fp:
                pickle.dump(
                    (self.all_videos_subjects_tasks_gaze_data_dict, sorted(self.all_videos_subject_task_list)), fp
                )
        else:
            print("Loading from cached pkl file")
            with open(self.all_videos_subjects_tasks_gaze_data_dict_path, "rb") as fp:
                self.all_videos_subjects_tasks_gaze_data_dict, self.all_videos_subject_task_list = pickle.load(fp)
                self.all_videos_subject_task_list = sorted(self.all_videos_subject_task_list)

        self._create_metadata_tuple_list()  # implementation is the respective derived classes.
        assert (
            self.metadata_len is not None or self.metadata_len <= 0 or self.metadata_list is not None
        ), "Metadata list not properly initialized or is empty"

    def _create_metadata_tuple_list(self):
        raise NotImplementedError

    def fetch_image_from_id(self, video_id, frame_idx, return_reduced_size=True):
        """
        Get the cached video frame image from video_id and frame index

        Parameters
        ----------
        video_id : int
            DREYEVE VIDEO ID - [6,7,10,11,26,35,53,60]
        frame_idx: int
            Query frame number in video_id - [0, 7500]
        return_reduced_size: bool
            Flag indicating whether the returned video frame image should be rescaled to the correct resolution or not. If False, full size will be returned.
            If True and if cache doesn't exists, full sized frame image is resized and cached back in the same directory for later reuse

        Returns
        -------
        video_frame: numpy.array (H, W, 3) or (h, w, 3), where H,W refers to full size image dimensions and h, w refers to resized dimensions
        """

        video_directory_cache_path = os.path.join(self.precache_dir, "frame_image_cached", "{0:02d}".format(video_id))
        #  precache id for the fullsized image.
        full_size_precache_id = "frame_{}".format(frame_idx)
        #  path for full size image
        full_size_precache_filename = os.path.join(video_directory_cache_path, full_size_precache_id) + ".jpg"
        assert os.path.exists(
            full_size_precache_filename
        ), "Full resolution video frames need to be extracted and cached beforehand"
        # precache id for the reduced size for the fullsized image (different aspect ratios will have different images)
        reduced_size_precache_id = full_size_precache_id + "_ar_{}".format(self.aspect_ratio_reduction_factor)
        # full path to reduced_size image (in the same folder as the full size image)
        reduced_size_precache_filename = os.path.join(video_directory_cache_path, reduced_size_precache_id) + ".jpg"

        if return_reduced_size:  # if reduced size image to be returned
            if os.path.exists(reduced_size_precache_filename):  # if already cached read and return
                video_frame = cv2.imread(reduced_size_precache_filename)  # in BGR format
            # if not, read in the full size image (assumes it exists), and then resize it and cache it and return it
            else:
                full_frame = cv2.imread(full_size_precache_filename)
                # resize full frame and cache the result in the same folder for later use.
                full_frame = np.float32(full_frame)
                video_frame = cv2.resize(full_frame, (self.new_image_width, self.new_image_height))
                cv2.imwrite(reduced_size_precache_filename, video_frame)
            return reduced_frame
        else:  # return full sized frame
            video_frame = cv2.imread(full_size_precache_filename)

        return video_frame

    def fetch_optic_flow_from_id(self, video_id, frame_idx, return_reduced_size=True):
        """
        Get the cached optic flow from video_id and frame index

        Parameters
        ----------
        video_id : int
            DREYEVE VIDEO ID - [6,7,10,11,26,35,53,60]
        frame_idx: int
            Query frame number in video_id - [0, 7500]
        return_reduced_size: bool
            Flag indicating whether the returned optic flow should be rescaled to the correct resolution or not. If False, full size will be returned.
            If True and if cache doesn't exists, full sized flow is resized and cached back in the same directory for later reuse

        Returns
        -------
        optical_frame: numpy.array (H, W, 2) or (h, w, 2), where H,W refers to full size image dimensions and h, w refers to resized dimensions
        """
        pass

    def fetch_segmentation_mask_from_id(self, video_id, frame_idx, return_reduced_size=True):
        """
        Get the cached segmentation mask image from video_id and frame index

        Parameters
        ----------
        video_id : int
            DREYEVE VIDEO ID - [6,7,10,11,26,35,53,60]
        frame_idx: int
            Query frame number in video_id - [0, 7500]
        return_reduced_size: bool
            Flag indicating whether the returned segmentation mask image should be rescaled to the correct resolution or not. If False, full size will be returned.
            If True and if cache doesn't exists, full sized segmentation mask image is resized and cached back in the same directory for later reuse

        Returns
        -------
        segmentation_frame: numpy.array (H, W, 3) or (h, w, 3), where H,W refers to full size image dimensions and h, w refers to resized dimensions
        """
        pass

    def fetch_gaze_points_from_id(self, video_id, frame_idx, subject, task):
        """
        Get the cached segmentation mask image from video_id and frame index

        Parameters
        ----------
        video_id : int
            DREYEVE VIDEO ID - [6,7,10,11,26,35,53,60]
        frame_idx: int
            Query frame number in video_id - [0, 7500]
        subject: int
            Subject ID
        task: str
            Cognitive task modifier experienced by subject during gazing.

        Returns
        -------
        gaze_points_array: numpy.array (L, 2) , where L is the number of gaze points requested. L is fixed for a particular training run and is determined by args.fixed_gaze_list_length
        """
        pass

    def __len__(self):
        return self.metadata_len

    def _get_single_item(self, metadata):
        """
        Get the cached segmentation mask image from video_id and frame index

        Parameters
        ----------
        metadata: tuple
            Tuple containing (video_id, subject, task, frame_id), where
            video_id : int
                DREYEVE VIDEO ID - [6,7,10,11,26,35,53,60]
            frame_id: int
                Query frame number in video_id - [0, 7500]
            subject: int
                Subject ID
            task: str
                Cognitive task modifier experienced by subject during gazing.

        Returns
        -------
        data_item: dict, with following keys
            ROAD_IMAGE_0: numpy.array, (C, h, w), resized video_frame at frame_idx
            SHOULD_TRAIN_INPUT_GAZE_0: bool, (L, 2) Flag indicating whether the gaze points are valid for training
            ....

        auxiliary_info: OrderedDict, with following keys
            ....
        """

        (video_id, subject, task, frame_idx) = metadata

        # fetch road video frame image at frame idx
        road_img = self.fetch_image_from_id(video_id, frame_idx, self.return_reduced_size)
        import IPython

        IPython.embed(banner1="check image")

        return data_item, auxiliary_info
        pass
