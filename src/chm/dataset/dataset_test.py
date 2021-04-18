#!/usr/bin/env python3 -B
from configs.args_file import parse_arguments
from chm_gaze_dataset import CognitiveHeatMapGazeDataset
from chm_att_awareness_dataset import CognitiveHeatMapAttAwarenessDataset
from chm_pairwise_gaze_dataset import CognitiveHeatMapPairwiseGazeDataset

import uuid


if __name__ == "__main__":
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash)
    params_dict = vars(args)
    data_dir = params_dict.get("data_dir")
    precache_dir = params_dict.get("precache_dir")
    dataset_type = params_dict.get("dataset_type")

    gaze_dataset = CognitiveHeatMapGazeDataset(data_dir, precache_dir, "train", params_dict)
    gaze_data_dict, gaze_auxiliary_info_list = gaze_dataset[0]

    awareness_dataset = CognitiveHeatMapAttAwarenessDataset(data_dir, precache_dir, "train", params_dict)
    awareness_data_dict, awareness_auxiliary_info_list = awareness_dataset[0]

    pairwise_dataset = CognitiveHeatMapPairwiseGazeDataset(data_dir, precache_dir, "train", params_dict)
    pairwise_data_dict = pairwise_dataset[0]  # contains data_t, data_tp1
    import IPython

    IPython.embed(banner1="check data_dict")
