#!/usr/bin/env python3 -B
from configs.args_file import parse_arguments
from chm_gaze_dataset import CognitiveHeatMapGazeDataset
from chm_att_awareness_dataset import CognitiveHeatMapAttAwarenessDataset

import uuid


if __name__ == "__main__":
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash)
    params_dict = vars(args)
    data_dir = params_dict.get("data_dir")
    precache_dir = params_dict.get("precache_dir")
    dataset_type = params_dict.get("dataset_type")

    # train_dataset = CognitiveHeatMapGazeDataset(data_dir, precache_dir, "train", params_dict)
    # data_dict, auxiliary_info_list = train_dataset[126]

    awareness_dataset = CognitiveHeatMapAttAwarenessDataset(data_dir, precache_dir, "train", params_dict)
    data_dict, auxiliary_info_list = awareness_dataset[12]
    import IPython

    IPython.embed(banner1="check data_dict")
