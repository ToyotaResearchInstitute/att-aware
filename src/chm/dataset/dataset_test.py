#!/usr/bin/env python3 -B
from configs.args_file import parse_arguments
from chm_gaze_dataset import CHMGazeDataset
from chm_att_awareness_dataset import CHMAttAwarenessDataset
from chm_pairwise_gaze_dataset import CHMPairwiseGazeDataset

import uuid

if __name__ == "__main__":
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash)
    params_dict = vars(args)
    dataset_type = params_dict.get("dataset_type")

    awareness_dataset = CHMAttAwarenessDataset("train", params_dict)

    gaze_dataset = CHMGazeDataset("train", params_dict, skip_list=awareness_dataset.get_metadata_list())
    gaze_data_dict, gaze_auxiliary_info_list = gaze_dataset[0]

    awareness_data_dict, awareness_auxiliary_info_list = awareness_dataset[0]

    pairwise_dataset = CHMPairwiseGazeDataset("train", params_dict)
    pairwise_data_dict = pairwise_dataset[0]  # contains data_t, data_tp1
    import IPython

    IPython.embed(banner1="check data_dict")
