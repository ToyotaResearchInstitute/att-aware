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

        self.fixed_gaze_list_length = self.params_dict.get("fixed_gaze_list_length", 3)
        self.request_auxiliary_info = self.params_dict.get("request_auxiliary_info", True)
        assert (
            self.fixed_gaze_list_length <= 10
        ), "The number of gaze points used per frame should in the range [1, 10]. "

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
        road_frame: numpy.array (H, W, 3) or (h, w, 3), where H,W refers to full size image dimensions and h, w refers to resized dimensions
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
                road_frame = cv2.imread(reduced_size_precache_filename)  # in BGR format
            # if not, read in the full size image (assumes it exists), and then resize it and cache it and return it
            else:
                full_frame = cv2.imread(full_size_precache_filename)
                # resize full frame and cache the result in the same folder for later use.
                full_frame = np.float32(full_frame)
                road_frame = cv2.resize(full_frame, (self.new_image_width, self.new_image_height))
                cv2.imwrite(reduced_size_precache_filename, video_frame)
        else:  # return full sized frame
            road_frame = cv2.imread(full_size_precache_filename)

        return road_frame

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
        optic_flow_frame: numpy.array (H, W, 2) or (h, w, 2), where H,W refers to full size image dimensions and h, w refers to resized dimensions
        """
        optic_flow_directory_cache_path = os.path.join(self.precache_dir, "optic_flow", "{0:02d}".format(video_id))
        #  precache id for the fullsized image.
        cached_size_precache_id = "frame_{}".format(frame_idx)
        cached_size_precache_filename = os.path.join(optic_flow_directory_cache_path, cached_size_precache_id) + ".npy"
        assert os.path.exists(
            cached_size_precache_filename
        ), "Optic flow frames need to be extracted and cached beforehand"

        # precache id for the reduced size for the fullsized image (different aspect ratios will have different images)
        reduced_size_precache_id = cached_size_precache_id + "_ar_{}".format(self.aspect_ratio_reduction_factor)
        # full path to reduced_size image (in the same folder as the cached size image)
        reduced_size_precache_filename = (
            os.path.join(optic_flow_directory_cache_path, reduced_size_precache_id) + ".npy"
        )

        full_size_precache_id = cached_size_precache_id + "_full_"
        # full path to full_size image (in the same folder as the cached size image)
        full_size_precache_filename = os.path.join(optic_flow_directory_cache_path, full_size_precache_id) + ".npy"

        if return_reduced_size:
            if os.path.exists(reduced_size_precache_filename):
                optic_flow_frame = np.load(reduced_size_precache_filename)  # (h, w, 2)
            else:
                cached_size_frame = np.load(cached_size_precache_filename)  # (h'+hpad, w'+wpad, 2)
                if OPTIC_FLOW_H_PAD > 0:
                    cached_size_frame = cached_size_frame[OPTIC_FLOW_H_PAD:-OPTIC_FLOW_H_PAD, :, :]  # (h', w'+wpad, 2)
                if OPTIC_FLOW_W_PAD > 0:
                    cached_size_frame = cached_size_frame[:, OPTIC_FLOW_W_PAD:-OPTIC_FLOW_W_PAD, :]  # (h', w', 2)
                optic_flow_frame = cv2.resize(
                    cached_size_frame / (self.aspect_ratio_reduction_factor / OPTIC_FLOW_SCALE_FACTOR),
                    (self.new_image_width, self.new_image_height),
                )  # the array has to be scaled, because the ux and uy values change according to resolution. OPTIC_FLOW_SCALE_FACTOR allows for optic flow to be cached at a lower resolution than full scale.
                np.save(reduced_size_precache_filename, optic_flow_frame)
        else:
            if os.path.exists(full_size_precache_filename):
                optic_flow_frame = np.load(full_size_precache_filename)
            else:
                cached_size_frame = np.load(cached_size_precache_filename)  # (h'+hpad, w'+wpad, 2)
                if OPTIC_FLOW_H_PAD > 0:
                    cached_size_frame = cached_size_frame[OPTIC_FLOW_H_PAD:-OPTIC_FLOW_H_PAD, :, :]  # (h', w'+wpad, 2)
                if OPTIC_FLOW_W_PAD > 0:
                    cached_size_frame = cached_size_frame[:, OPTIC_FLOW_W_PAD:-OPTIC_FLOW_W_PAD, :]  # (h', w', 2)

                optic_flow_frame = cv2.resize(
                    cached_size_frame * OPTIC_FLOW_SCALE_FACTOR, (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                )  # scale the optic flow ux uy values to the full resolution before resizing and saving
                np.save(full_size_precache_filename, optic_flow_frame)

        return optic_flow_frame  # (h, w, 2) or (H, W, 2)

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
        segmentations_directory_cache_path = os.path.join(
            self.precache_dir, "segmentations_from_video", "{0:02d}".format(video_id), "segmentation_frames"
        )
        full_size_precache_id = "frame_{}".format(frame_idx)
        #  path for full size image
        full_size_precache_filename = (
            os.path.join(segmentations_directory_cache_path, full_size_precache_id) + ".png"
        )  # assumed to be .png because lossless format is needed
        assert os.path.exists(
            full_size_precache_filename
        ), "Full resolution video frames need to be extracted and cached beforehand"
        # precache id for the reduced size for the fullsized image (different aspect ratios will have different images)
        reduced_size_precache_id = full_size_precache_id + "_ar_{}".format(self.aspect_ratio_reduction_factor)
        # full path to reduced_size image (in the same folder as the full size image)
        reduced_size_precache_filename = (
            os.path.join(segmentations_directory_cache_path, reduced_size_precache_id) + ".png"
        )
        if return_reduced_size:
            if os.path.exists(reduced_size_precache_filename):
                segmentation_frame = np.asarray(Image.open(reduced_size_precache_filename))  # RGB
            else:
                full_frame = np.asarray(Image.open(full_size_precache_filename))
                full_frame = np.float64(full_frame)
                segmentation_frame = cv2.resize(full_frame, (self.new_image_width, self.new_image_height))
                cv2.imwrite(reduced_size_precache_filename, segmentation_frame)
        else:
            # get full size segmentation frame.
            segmentation_frame = np.asarray(Image.open(full_size_precache_filename))
            segmentation_frame = np.float64(segmentation_frame)

        return segmentation_frame

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
        full_size_gaze_points_array: numpy.array (L, 2)
            Gaze points in full screen resolution where L is the number of gaze points requested. L is fixed for a particular training run and is determined by args.fixed_gaze_list_length
        resized_gaze_points_array: numpy.array (L, 2)
            Gaze points in network input dimensions
        normalized_gaze_points_array: numpy.array (L, 2)
            Gaze_points normalized to [0, 1]
        should_train_array: numpy.array (L, 1)
            Bits indicating whether the gaze points can be used for training
        """
        video_subject_task_gaze_df = self.all_videos_subjects_tasks_gaze_data_dict[video_id][subject][
            task
        ]  # cached pandas dataframe for the specified video_id, subject and task
        all_gaze_df_at_frame_idx = video_subject_task_gaze_df[
            video_subject_task_gaze_df["frame_gar"] == frame_idx
        ]  # data frame slice at specified frame idx containing 10 gaze points.

        gaze_df_at_frame_idx = all_gaze_df_at_frame_idx.sample(
            n=self.fixed_gaze_list_length, random_state=1
        )  # sample self.fixed_gaze_list_length number of gaze points at frame_idx using a fixed seed

        # initialized gaze point array
        resized_gaze_points_array = np.zeros(
            (self.fixed_gaze_list_length, 2), dtype=np.float32
        )  # (L, 2) #gaze points in the resolution needed for network training
        should_train_array = np.zeros(
            (self.fixed_gaze_list_length, 1), dtype=np.float32
        )  # (L, 2). Boolean array indicating whether the gaze points are valid on not.
        full_size_gaze_points_array = np.zeros(
            (self.fixed_gaze_list_length, 2), dtype=np.float32
        )  # (L, 2) #raw data in display dimensions (DISPLAY WIDTH, DISPLAY HEIGHT)
        normalized_gaze_points_array = np.zeros(
            (self.fixed_gaze_list_length, 2), dtype=np.float32
        )  # (L, 2) #normalized to [0,1]

        gaze_points_x = gaze_df_at_frame_idx["X"].values  # np.array (L, 1) #in full size dimension
        gaze_points_y = gaze_df_at_frame_idx["Y"].values  # np.array (L, 1)
        event_type_list = gaze_df_at_frame_idx["event_type"].values

        gaze_points = np.concatenate(
            (
                gaze_points_x.reshape(self.fixed_gaze_list_length, 1),
                gaze_points_y.reshape(self.fixed_gaze_list_length, 1),
            ),
            axis=1,
        )  # (L, 2)
        for i in range(self.fixed_gaze_list_length):
            should_train_array[i, :] = 1.0  # set should train flag to be 1 (True)
            if (
                np.isnan(gaze_points[i, :]).sum() or event_type_list[i] != "Fixation"
            ):  # if there are nans in the gaze points or if the type is not Fixation
                gaze_points[i, :] = np.array(
                    [-10 * DISPLAY_WIDTH, -10 * DISPLAY_HEIGHT]
                )  # if gaze points are NAN, modify the gaze to be outside the screen dimensions and set the train bit to be False
                should_train_array[i, :] = 0.0  # set should train flag to be 0.0 (False)

            # full resolution gaze points
            full_size_gp = gaze_points[i, :]  # (2,) #in display dimensions
            full_size_gp = np.expand_dims(full_size_gp, axis=0)  # (1,2)
            full_size_gaze_points_array[i, :] = full_size_gp  # in display dim (1080, 1920)  (H, W)

            # Resize gaze. Resize it to the size of the network input
            resized_gaze_points_array[i, 0] = (full_size_gp[0, 0] / DISPLAY_WIDTH) * self.new_image_width
            resized_gaze_points_array[i, 1] = (full_size_gp[0, 1] / DISPLAY_HEIGHT) * self.new_image_height

            # normalized transformed gaze. [0,1]
            normalized_gaze_points_array[i, 0] = full_size_gp[0, 0] / DISPLAY_WIDTH
            normalized_gaze_points_array[i, 1] = full_size_gp[0, 1] / DISPLAY_HEIGHT

        return full_size_gaze_points_array, resized_gaze_points_array, normalized_gaze_points_array, should_train_array

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
        road_frame = self.fetch_image_from_id(video_id, frame_idx, self.return_reduced_size)
        road_frame = np.float32(road_frame)  # (h, w, 3) if self.return_reduced_size is True else (H, W, 3)
        road_frame = road_frame.transpose([2, 0, 1])  # (3, h, w) or (3, H, W)

        # fetch segmentation masks at frame idx
        segmentation_frame = self.fetch_segmentation_mask_from_id(video_id, frame_idx, self.return_reduced_size)
        segmentation_frame = np.float32(
            segmentation_frame
        )  # (h, w, 3) if self.return_reduced_size is True else (H, W, 3)
        segmentation_frame = segmentation_frame.transpose([2, 0, 1])  # (3, h, w) or (3, H, W)

        # fetch optic flow at frame idx
        optic_flow_frame = self.fetch_optic_flow_from_id(
            video_id, frame_idx, self.return_reduced_size
        )  # (h, w ,C=2) or (H, W, C=2) depending on return_reduced_size flags
        optic_flow_frame = np.float32(optic_flow_frame)
        optic_flow_frame = optic_flow_frame.transpose([2, 0, 1])  # (C=2, h, w) or (C=2, H, W)

        # extract gaze points from the pandas dataframe
        (
            full_size_gaze_points_array,
            resized_gaze_points_array,
            normalized_gaze_points_array,
            should_train_array,
        ) = self.fetch_gaze_points_from_id(video_id, frame_idx, subject, task)

        data_item = {
            ROAD_IMAGE_0: road_frame,  # (C, h, w)
            SHOULD_TRAIN_INPUT_GAZE_0: should_train_array,  # (L, 1)
            # in reshaped pixel coordinates (L, 2) #network input dimension
            TRANSFORMED_INPUT_GAZE_0: resized_gaze_points_array,
            NORMALIZED_TRANSFORMED_INPUT_GAZE_0: normalized_gaze_points_array,  # (L, 2), [0, 1]
            GROUND_TRUTH_GAZE_0: resized_gaze_points_array,  # (L, 2)
            SEGMENTATION_MASK_0: segmentation_frame,  # (C, h, w)
            OPTIC_FLOW_IMAGE_0: optic_flow_frame,  # (2, h, w)
        }  # (L, 2) #network input dimensions. Used for network training

        if not self.request_auxiliary_info:
            return data_item, None
        else:
            auxiliary_info = OrderedDict()
            auxiliary_info[AUXILIARY_INFO_VIDEO_ID] = video_id
            auxiliary_info[AUXILIARY_INFO_SUBJECT_ID] = subject
            auxiliary_info[AUXILIARY_INFO_FULL_SIZE_GAZE_0] = full_size_gaze_points_array

            return data_item, auxiliary_info
