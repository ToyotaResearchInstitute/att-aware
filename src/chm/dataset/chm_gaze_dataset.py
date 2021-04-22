# Copyright 2020 Toyota Research Institute.  All rights reserved.
import itertools

from chm.dataset.chm_base_dataset import CHMBaseDataset


class CHMGazeDataset(CHMBaseDataset):
    def __init__(self, dataset_type=None, params_dict=None, **kwargs):
        """
        CHMGazeDataset dataset class/
        Dataset class for returning gaze and image data for a single sequence

        Parameters
        ----------
        dataset_type : str {'train, 'test', 'vis'}
            String indicating the type of dataset
        params_dict : dict
            Dictionary containing the args passed from the training script
        skip_list: list
            List containing ((video_id, subject, task), query_frame) tuples that needed to be excluded from the gaze dataset.

        """
        super().__init__(dataset_type=dataset_type, params_dict=params_dict, **kwargs)

    def _setup_resources(self, **kwargs):
        """
        Sets up any resources (such loading csv files etc) needed for this derived Dataset.

        Parameters
        ----------
        None

        Returns
        -------
        None.
        """
        self.skip_list = None
        if "skip_list" in kwargs:
            self.skip_list = kwargs["skip_list"]

    def _create_metadata_tuple_list(self):
        """
        Initializes the metadata_len and metadata_list if needed. The function is called at the very end of the CHMBaseDataset init function

        Parameters
        ----------
        None

        Returns
        -------
        None. Only Results in populating the self.metadata_list
        """

        # create all combinations of video, subject, task tuples for the specified video, subject, task args
        metadata_list_all_comb = list(itertools.product(self.sequence_ids, self.subject_ids, self.task_ids))
        # filter out those combinations that are not present in the available combinations
        metadata_list_all_comb = [d for d in metadata_list_all_comb if d in self.all_videos_subject_task_list]
        # append the query frame list to each tuple
        self.metadata_list = [(a, b) for a in metadata_list_all_comb for b in self.query_frame_idxs_list]

        # if a skip_list is provided, filter the metadata list accordingly
        if self.skip_list is not None:
            self.metadata_list = [m for m in self.metadata_list if m not in self.skip_list]

        self.metadata_len = len(self.metadata_list)  # Total number of available snippets

    def get_metadata_list(self):
        """
        Returns the metadata list (of tuples) for this dataset

        Parameters
        ----------
        None

        Returns
        -------
        metadata_list: list
            List of tuples containing metadata information (video_id, subject, task, query_frame) for each data item
        """
        return self.metadata_list

    def __getitem__(self, idx):
        """
        Required getitem() for PyTorch gaze dataset.

        Parameters
        ----------
        idx: int
            Index of the data item in self.metadata_list

        Returns
        -------
        data_dict: dict
            Ordered dictionary containing the various data items needed for training. Each item in the dict is a tensor or numpy.array

        auxiliary_info_list: list
            List of auxiliary information needed for other purposes. If auxiliary info flag is set to be False, auxiliary_info_list = [].
        """
        (video_id, subject, task), query_frame = self.metadata_list[idx]
        data_dict, auxiliary_info_list = self._get_sequence(video_id, subject, task, query_frame)
        return data_dict, auxiliary_info_list