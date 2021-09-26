# Copyright 2020 Toyota Research Institute.  All rights reserved.
import itertools

from maad.dataset.maad_base_dataset import MAADBaseDataset


class MAADPairwiseGazeDataset(MAADBaseDataset):
    def __init__(self, dataset_type=None, params_dict=None, **kwargs):
        """
        MAADPairwiseGazeDataset dataset class.
        Returns a pair of sequences that is used to compute the consistency cost term.

        Parameters:
        ----------
        dataset_type : str {'train, 'test', 'vis'}
            String indicating the type of dataset
        params_dict : dict
            Dictionary containing the args passed from the training script

        """
        super().__init__(dataset_type=dataset_type, params_dict=params_dict, **kwargs)

    def _setup_resources(self, **kwargs):
        """
        Sets up any resources (such loading csv files etc) needed for this derived Dataset.

        Parameters:
        ----------
        kwargs: dict
            dictionary of named arguments.

        Returns:
        -------
        None
        """
        pass

    def _create_metadata_tuple_list(self):
        """
        Initializes the metadata_len and metadata_list if needed. The function is called at the very end of the MAADBaseDataset init function

        Parameters:
        ----------
        None

        Returns:
        -------
        None. Only Results in populating the self.metadata_list
        """
        # create all combinations of video, subject, task tuples for the specified video, subject, task args
        metadata_list_all_comb = list(itertools.product(self.sequence_ids, self.subject_ids, self.task_ids))
        # filter out those combinations that are not present in the available combinations
        metadata_list_all_comb = [d for d in metadata_list_all_comb if d in self.all_videos_subject_task_list]
        # append the query_frame at t and t+self.temporal_downsample_factor to each video, subject, task tuple
        self.metadata_list = [
            (a, b, b + self.temporal_downsample_factor)
            for a in metadata_list_all_comb
            for b in self.query_frame_idxs_list[: -self.temporal_downsample_factor]
        ]

        self.metadata_len = len(self.metadata_list)  # Total number of available snippets

    def get_metadata_list(self):
        return self.metadata_list

    def __getitem__(self, idx):
        """
        Required getitem() for PyTorch gaze dataset.

        Parameters:
        ----------
        idx: int
            Index of the data item in self.metadata_list

        Returns:
        -------
        data_dict: dict
            Dictionary containing the data_dict, auxiliary_info for sequence at frame t and at t+self.temporal_downsample_factor
        """
        data_dict = {}
        (video_id, subject, task), query_frame_t, query_frame_tp1 = self.metadata_list[idx]
        # get gaze data for sequence at t
        data_dict_t, auxiliary_info_list_t = self._get_sequence(video_id, subject, task, query_frame_t)
        # get gaze data for sequence at t+self.temporal_downsample_factor
        data_dict_tp1, auxiliary_info_list_tp1 = self._get_sequence(video_id, subject, task, query_frame_tp1)

        data_dict["data_t"] = (data_dict_t, auxiliary_info_list_t)
        data_dict["data_tp1"] = (data_dict_tp1, auxiliary_info_list_tp1)
        return data_dict
