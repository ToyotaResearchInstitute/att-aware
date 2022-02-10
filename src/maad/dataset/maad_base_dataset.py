# Copyright 2020 Toyota Research Institute.  All rights reserved.
import os
import cv2
import numpy as np
import collections
import pickle
import collections

from torch.utils.data import Dataset
from PIL import Image
from utils.maad_consts import *


class MAADBaseDataset(Dataset):
    def __init__(self, dataset_type=None, params_dict=None, **kwargs):
        """
        MAADBaseDataset dataset class. Base class for all the other Dataset classes used for MAAD.
        Implements all the getters for caches and loads up the pandas dataframe with all gaze information

        Parameters
        ----------
        dataset_type : str {'train, 'test', 'vis'}
            String indicating the type of dataset
        params_dict : dict
            Dictionary containing the args passed from the training script

        """
        # Asserts

        assert dataset_type is not None and dataset_type in [
            "train",
            "test",
            "vis",
        ], "Dataset type has to be a string and has to either [train, test, vis]"

        assert (
            params_dict is not None
        ), "Params dict should be passed. Contains all args from parse_arguments in args_file"

        self.dataset_type = dataset_type
        self.params_dict = params_dict

        self.precache_dir = self.params_dict.get("precache_dir", None)
        assert self.precache_dir is not None and os.path.exists(
            self.precache_dir
        ), "Valid path to directory containing video_frame, segmentation and optic flow cache is necessary"

        # factor by which the full sized image needs to be rescaled before used by network for training
        self.aspect_ratio_reduction_factor = self.params_dict.get("aspect_ratio_reduction_factor", 8.0)
        # hopsize for downsampled sequences
        self.temporal_downsample_factor = self.params_dict.get("temporal_downsample_factor", 6)

        self.all_videos_subjects_tasks_gaze_data_dict_path = self.params_dict.get("all_gaze_data_dict", None)
        assert (
            self.all_videos_subjects_tasks_gaze_data_dict_path is not None
        ), "Please provide the full path to the gaze data pkl file"

        self.fixed_gaze_list_length = self.params_dict.get("fixed_gaze_list_length", 3)
        assert (
            self.fixed_gaze_list_length <= 10
        ), "The number of gaze points used per frame should in the range [1, 10]. "

        self.request_auxiliary_info = self.params_dict.get("request_auxiliary_info", True)

        # depending on the dataset type grab the corresponding
        # sequence length, sequence id, subject id and task id from the params dict
        self.sequence_length = self.params_dict.get("{}_sequence_length".format(self.dataset_type), 20)
        self.sequence_ids = self.params_dict.get("{}_sequence_ids".format(self.dataset_type))
        self.subject_ids = self.params_dict.get("{}_subject_ids".format(self.dataset_type))
        # cognitive-task-modifiers (referred to as task in the code)
        self.task_ids = self.params_dict.get("{}_task_ids".format(self.dataset_type))

        self.first_query_frame = (self.sequence_length - 1) * self.temporal_downsample_factor
        # list of query frames used for each video. Each query frame idx corresponds to the last frame of the snippet used.
        self.query_frame_idxs_list = list(range(self.first_query_frame, MAX_NUM_VIDEO_FRAMES))

        # full size road dimensions.
        self.ORIG_ROAD_IMG_DIMS = self.params_dict.get("orig_road_image_dims")
        self.ORIG_ROAD_IMAGE_HEIGHT = self.ORIG_ROAD_IMG_DIMS[1]
        self.ORIG_ROAD_IMAGE_WIDTH = self.ORIG_ROAD_IMG_DIMS[2]

        # flag which ensures that the scaled version of the cached images are returned from the
        # fetch functions when aspect_ratio is smaller.
        self.return_reduced_size = False if self.aspect_ratio_reduction_factor == 1.0 else True

        self.new_image_width = int(round(self.ORIG_ROAD_IMAGE_WIDTH / self.aspect_ratio_reduction_factor))
        self.new_image_height = int(round(self.ORIG_ROAD_IMAGE_HEIGHT / self.aspect_ratio_reduction_factor))

        self.all_videos_subject_task_list = []
        with open(self.all_videos_subjects_tasks_gaze_data_dict_path, "rb") as fp:
            self.all_videos_subjects_tasks_gaze_data_dict, self.all_videos_subject_task_list = pickle.load(fp)
            self.all_videos_subject_task_list = sorted(self.all_videos_subject_task_list)

        # setup metadata list.
        self.metadata_len = None
        self._setup_resources(**kwargs)  # set up any resources needed for creation of metadata tuple list
        self._create_metadata_tuple_list()  # implementation is in the respective derived classes.
        assert (
            self.metadata_len is not None and self.metadata_len > 0
        ), "Metadata list not properly initialized or is empty"

    def _setup_resources(self):
        raise NotImplementedError

    def _create_metadata_tuple_list(self):
        raise NotImplementedError

    def fetch_image_from_id(self, video_id, frame_idx, return_reduced_size=True):
        """
        Get the cached video frame image from video_id and frame index.
        Assumes that the images as stored .jpg at ~/(self.precache_dir)/frame_image_cached/"{0:02d}".format(video_id)/frame_{frame_idx}.jpg
        Performs appropriate resizing if necessary and stores the resized video frame image in the same directory

        Parameters:
        ----------
        video_id : int
            DREYEVE VIDEO ID - [6,7,10,11,26,35,53,60]
        frame_idx: int
            Query frame number in video_id - [0, 7500]
        return_reduced_size: bool
            Flag indicating whether the returned video frame image should be rescaled to the correct resolution or not. If False, full size will be returned.
            If True and if cache doesn't exists, full sized frame image is resized and cached back in the same directory for later reuse

        Returns:
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
                cv2.imwrite(reduced_size_precache_filename, road_frame)
        else:  # return full sized frame
            road_frame = cv2.imread(full_size_precache_filename)

        return road_frame

    def fetch_optic_flow_from_id(self, video_id, frame_idx, return_reduced_size=True):
        """
        Get the cached optic flow from video_id and frame index. Assumes optic flow is cached as a .npy file with dimensions
        (DISPLAY_HEIGHT/OPTIC_FLOW_SCALE_FACTOR + 2*OPTIC_FLOW_H_PAD, DISPLAY_WIDTH/OPTIC_FLOW_SCALE_FACTOR + 2*OPTIC_FLOW_W_PAD, 2)
        Assumes that the optic flow is cached at ~/(self.precache_dir)/optic_flow/"{0:02d}".format(video_id)/frame_{frame_idx}.npy.
        Performs appropriate resizing if necessary and stores the resized optic flow image in the same directory

        Parameters:
        ----------
        video_id : int
            DREYEVE VIDEO ID - [6,7,10,11,26,35,53,60]
        frame_idx: int
            Query frame number in video_id - [0, 7500]
        return_reduced_size: bool
            Flag indicating whether the returned optic flow should be rescaled to the correct resolution or not. If False, full size will be returned.
            If True and if cache doesn't exists, full sized flow is resized and cached back in the same directory for later reuse

        Returns:
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

                # the array has to be scaled, because the ux and uy values change according to resolution.
                # OPTIC_FLOW_SCALE_FACTOR allows for optic flow to be cached at a lower resolution than full scale.
                optic_flow_frame = cv2.resize(
                    cached_size_frame / (self.aspect_ratio_reduction_factor / OPTIC_FLOW_SCALE_FACTOR),
                    (self.new_image_width, self.new_image_height),
                )
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

                # scale the optic flow ux uy values to the full resolution before resizing and saving
                optic_flow_frame = cv2.resize(
                    cached_size_frame * OPTIC_FLOW_SCALE_FACTOR, (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                )
                np.save(full_size_precache_filename, optic_flow_frame)

        return optic_flow_frame  # (h, w, 2) or (H, W, 2)

    def fetch_segmentation_mask_from_id(self, video_id, frame_idx, return_reduced_size=True):
        """
        Get the cached segmentation mask image from video_id and frame index.
        Assumes that the images as stored .png at ~/(self.precache_dir)/segmentations_from_video/"{0:02d}".format(video_id)/segmentation_frames/frame_{frame_idx}.jpg
        Performs appropriate resizing if necessary and stores the resized video frame image in the same directory

        Parameters:
        ----------
        video_id : int
            DREYEVE VIDEO ID - [6,7,10,11,26,35,53,60]
        frame_idx: int
            Query frame number in video_id - [0, 7500]
        return_reduced_size: bool
            Flag indicating whether the returned segmentation mask image should be rescaled to the correct resolution or not. If False, full size will be returned.
            If True and if cache doesn't exists, full sized segmentation mask image is resized and cached back in the same directory for later reuse

        Returns:
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
        Get the gaze points for video_id, subject, task at frame_idx from the all_videos_subjects_tasks_gaze_data_dict

        Parameters:
        ----------
        video_id : int
            DREYEVE VIDEO ID - [6,7,10,11,26,35,53,60]
        frame_idx: int
            Query frame number in video_id - [0, 7500]
        subject: int
            Subject ID  - [1,23]
        task: str
            Cognitive task modifier experienced by subject during gazing ['control', 'blurred', 'flipped', 'readingtext', 'roadonly'].

        Returns:
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
        # cached pandas dataframe for the specified video_id, subject and task
        video_subject_task_gaze_df = self.all_videos_subjects_tasks_gaze_data_dict[video_id][subject][task]
        # data frame slice at specified frame idx containing 10 gaze points.
        all_gaze_df_at_frame_idx = video_subject_task_gaze_df[video_subject_task_gaze_df["frame_gar"] == frame_idx]

        # sample self.fixed_gaze_list_length number of gaze points at frame_idx using a fixed seed
        gaze_df_at_frame_idx = all_gaze_df_at_frame_idx.sample(n=self.fixed_gaze_list_length, random_state=1)

        # initialize all gaze data arrays

        # gaze points in the resolution needed for network training
        resized_gaze_points_array = np.zeros((self.fixed_gaze_list_length, 2), dtype=np.float32)  # (L, 2)
        # Array indicating whether the gaze points are valid or not. Valid = 1.0, Invalid = 0.0
        should_train_array = np.zeros((self.fixed_gaze_list_length, 1), dtype=np.float32)  # (L, 2)
        # gaze data in full display dimensions (DISPLAY WIDTH, DISPLAY HEIGHT)
        full_size_gaze_points_array = np.zeros((self.fixed_gaze_list_length, 2), dtype=np.float32)  # (L, 2)
        # gaze normalized to [0,1]
        normalized_gaze_points_array = np.zeros((self.fixed_gaze_list_length, 2), dtype=np.float32)  # (L, 2)

        gaze_points_x = gaze_df_at_frame_idx["X"].values  # np.array (L, 1) #in full size dimension
        gaze_points_y = gaze_df_at_frame_idx["Y"].values  # np.array (L, 1)
        # type of gaze point. [Fixation, Saccade, Blink, NA] etc
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
                # modify the gaze to be outside the screen dimensions and set the train bit to be False
                gaze_points[i, :] = np.array([-10 * DISPLAY_WIDTH, -10 * DISPLAY_HEIGHT])
                should_train_array[i, :] = 0.0  # set should train flag to be 0.0 (False)

            # full resolution gaze points
            full_size_gp = gaze_points[i, :]  # (2,) #in display dimensions
            full_size_gp = np.expand_dims(full_size_gp, axis=0)  # (1,2)
            full_size_gaze_points_array[i, :] = full_size_gp  # in display dim (1080, 1920)  (H, W)

            # resize gaze. Resize it to the size of the network input
            resized_gaze_points_array[i, 0] = (full_size_gp[0, 0] / DISPLAY_WIDTH) * self.new_image_width
            resized_gaze_points_array[i, 1] = (full_size_gp[0, 1] / DISPLAY_HEIGHT) * self.new_image_height

            # normalized transformed gaze. [0,1]
            normalized_gaze_points_array[i, 0] = full_size_gp[0, 0] / DISPLAY_WIDTH
            normalized_gaze_points_array[i, 1] = full_size_gp[0, 1] / DISPLAY_HEIGHT

        return full_size_gaze_points_array, resized_gaze_points_array, normalized_gaze_points_array, should_train_array

    def __len__(self):
        return self.metadata_len

    def get_metadata_list(self):
        """
        Getter for self.metadata_list. To be implemented by the derived classes
        """
        raise NotImplementedError

    def _get_sequence(self, video_id, subject, task, query_frame):
        """
        Get the data_dict for a single sequence

        Parameters:
        ----------
        video_id : int
            DREYEVE VIDEO ID - [6,7,10,11,26,35,53,60]
        frame_id: int
            Query frame number in video_id - [0, 7500]
        subject: int
            Subject ID - [1, 23]
        task: str
            Cognitive task modifier experienced by subject during gazing - ['control', 'blurred', 'flipped', 'readingtext', 'roadonly']

        Returns:
        -------
        data_dict: dict, data_dict containing the gaze data for the entire sequence
            ROAD_IMAGE_0: numpy.array, (T, C, h, w)
                Resized video_frame at frame_idx
            SHOULD_TRAIN_INPUT_GAZE_0: numpy.array, (T, L, 2)
                Flag indicating whether the gaze points are valid for training
            RESIZED_INPUT_GAZE_0: numpy.array (T, L, 2)
                Gaze points resized to the network dimensions
            NORMALIZED_INPUT_GAZE_0: numpy.array (T, L, 2)
                Gaze points resized to [0, 1]
            GROUND_TRUTH_GAZE_0: numpy.array (T, L, 2)
                Gaze points resized to the network dimensions
            SEGMENTATION_MASK_0: numpy.array (T, C, h, w)
                Resized segmentation frame at frame_idx
            OPTIC_FLOW_IMAGE_0: numpy.array (T, 2, h, w)
                Resized optic flow frame at frame_idx

        auxiliary_info_list: list (of length T), with each element an OrderedDict
            AUXILIARY_INFO_VIDEO_ID: int
                video_id of the video corresponding to the data item
            AUXILIARY_INFO_SUBJECT_ID: int
                subject id of the subject corresponding to the data_item
            AUXILIARY_INFO_FULL_SIZE_GAZE_0: numpy.array (L, 2)
                Gaze points in full resolution
        """
        # list of frame idxs to be used for the snippet
        data_item_query_framelist = [
            self.temporal_downsample_factor * (i) + query_frame - self.first_query_frame
            for i in range(self.sequence_length)
        ]

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
            for key in data_item_list[0]:
                data_dict[key] = []
                for data_item in data_item_list:  # iterate over the list containing the sequence of data_items
                    data_dict[key].append(np.expand_dims(data_item[key], axis=0))

                data_dict[key] = np.concatenate(data_dict[key], axis=0)
        except Exception as e:
            import IPython

            # embed to catch the exception if something goes wrong in the conversion of the data_item_list to the data dict
            IPython.embed(header="getitem invalid: " + str(e))

        if not self.request_auxiliary_info:
            return data_dict, []
        else:
            return data_dict, auxiliary_info_list

    def _get_single_item(self, metadata):
        """
        Get the data_dict for a single frame in a sequence

        Parameters
        ----------
        metadata: tuple
            Tuple containing (video_id, subject, task, frame_id), where
            video_id : int
                DREYEVE VIDEO ID - [6,7,10,11,26,35,53,60]
            frame_id: int
                Query frame number in video_id - [0, 7500]
            subject: int
                Subject ID - [1,23]
            task: str
                Cognitive task modifier experienced by subject during gazing - ['control', 'blurred', 'flipped', 'readingtext', 'roadonly']

        Returns
        -------
        data_item: dict, with following keys
            ROAD_IMAGE_0: numpy.array, (C, h, w)
                Resized video_frame at frame_idx
            SHOULD_TRAIN_INPUT_GAZE_0: numpy.array, (L, 2)
                Flag indicating whether the gaze points are valid for training
            RESIZED_INPUT_GAZE_0: numpy.array (L, 2)
                Gaze points resized to the network dimensions
            NORMALIZED_INPUT_GAZE_0: numpy.array (L, 2)
                Gaze points resized to [0, 1]
            GROUND_TRUTH_GAZE_0: numpy.array (L, 2)
                Gaze points resized to the network dimensions
            SEGMENTATION_MASK_0: numpy.array (C, h, w)
                Resized segmentation frame at frame_idx
            OPTIC_FLOW_IMAGE_0: numpy.array (2, h, w)
                Resized optic flow frame at frame_idx

        auxiliary_info: OrderedDict, with following keys
            AUXILIARY_INFO_VIDEO_ID: int
                video_id of the video corresponding to the data item
            AUXILIARY_INFO_SUBJECT_ID: int
                subject id of the subject corresponding to the data_item
            AUXILIARY_INFO_FRAME_IDX: int
                frame index of the corresponding data item
            AUXILIARY_INFO_FULL_SIZE_GAZE_0: numpy.array (L, 2)
                Gaze points in full resolution

        """

        (video_id, subject, task, frame_idx) = metadata

        # fetch road video frame image at frame idx
        road_frame = self.fetch_image_from_id(video_id, frame_idx, self.return_reduced_size)
        road_frame = np.float32(road_frame)  # (h, w, 3) if self.return_reduced_size is True else (H, W, 3)
        road_frame = road_frame.transpose([2, 0, 1])  # (3, h, w) or (3, H, W)

        # fetch segmentation masks at frame idx
        segmentation_frame = self.fetch_segmentation_mask_from_id(video_id, frame_idx, self.return_reduced_size)
        # (h, w, 3) if self.return_reduced_size is True else (H, W, 3)
        segmentation_frame = np.float32(segmentation_frame)
        segmentation_frame = segmentation_frame.transpose([2, 0, 1])  # (3, h, w) or (3, H, W)

        # fetch optic flow at frame idx
        # (h, w ,C=2) or (H, W, C=2) depending on return_reduced_size flags
        optic_flow_frame = self.fetch_optic_flow_from_id(video_id, frame_idx, self.return_reduced_size)
        optic_flow_frame = np.float32(optic_flow_frame)
        optic_flow_frame = optic_flow_frame.transpose([2, 0, 1])  # (C=2, h, w) or (C=2, H, W)

        # extract gaze points from the pandas dataframe
        (
            full_size_gaze_points_array,
            resized_gaze_points_array,
            normalized_gaze_points_array,
            should_train_array,
        ) = self.fetch_gaze_points_from_id(video_id, frame_idx, subject, task)

        # create data item dictionary
        data_item = {
            ROAD_IMAGE_0: road_frame,  # (C, h, w)
            SHOULD_TRAIN_INPUT_GAZE_0: should_train_array,  # (L, 1)
            RESIZED_INPUT_GAZE_0: resized_gaze_points_array,  # (L, 2) in #network input dimension
            NORMALIZED_INPUT_GAZE_0: normalized_gaze_points_array,  # (L, 2), [0, 1]
            GROUND_TRUTH_GAZE_0: resized_gaze_points_array,  # (L, 2)
            SEGMENTATION_MASK_0: segmentation_frame,  # (C, h, w)
            OPTIC_FLOW_IMAGE_0: optic_flow_frame,  # (2, h, w)
        }

        if not self.request_auxiliary_info:
            return data_item, None
        else:
            auxiliary_info = collections.OrderedDict()
            auxiliary_info[AUXILIARY_INFO_VIDEO_ID] = video_id
            auxiliary_info[AUXILIARY_INFO_FRAME_IDX] = frame_idx
            auxiliary_info[AUXILIARY_INFO_SUBJECT_ID] = subject
            auxiliary_info[AUXILIARY_INFO_FULL_SIZE_GAZE_0] = full_size_gaze_points_array

            return data_item, auxiliary_info
