# Copyright 2020 Toyota Research Institute.  All rights reserved.
import copy
import pandas as pd

from chm.dataset.chm_base_dataset import CognitiveHeatMapBaseDataset


class CognitiveHeatMapAttAwarenessDataset(CognitiveHeatMapBaseDataset):
    def __init__(self, dataset_type=None, params_dict=None):
        """
        CognitiveHeatMapAttAwarenessDataset dataset class

        Parameters
        ----------
        dataset_type : str {'train, 'test', 'vis'}
            String indicating the type of dataset
        params_dict : dict
            Dictionary containing the args passed from the training script
        """
        super().__init__(dataset_type=dataset_type, params_dict=params_dict)

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

        self.metadata_list = []
        for idx in range(len(self.att_awareness_labels)):
            att_label_item = self.att_awareness_labels.iloc[idx]

            video_id = att_label_item["video_id"]
            subject = att_label_item["subject"]
            task = att_label_item["cognitive_modifier"]
            query_frame = att_label_item["query_frame"]
            self.metadata_list.append(((video_id, subject, task), query_frame))

        import IPython

        IPython.embed(banner1="checl")

    def get_metadata_list(self):
        """
        Returns the metadata list (of tuples) for this dataset

        Parameters
        ----------
        None

        Returns
        -------
        metadata_lisr: list
            List of tuples containing metadata information (video_id, subject, task), query_frame for each data item
        """
        return self.metadata_list

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
        self.metadata_len = len(self.metadata_list)  # number of rows in the filtered data frame

    def __getitem__(self, idx):
        """
        Required getitem() for PyTorch dataset.

        Parameters
        ----------
        idx: int
            Index of the data item in self.att_awareness_labels

        Returns
        -------
        data_dict:  dict
            Same keys as the data_dict in chm_base_dataset.
            Additional key: att_annotation

        auxiliary_info_list: list
            Same as chm_base_dataset

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
