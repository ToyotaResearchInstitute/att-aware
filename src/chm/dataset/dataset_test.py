#!/usr/bin/env python3 -B
from configs.args_file import parse_arguments
from chm_gaze_dataset import CognitiveHeatMapGazeDataset

import uuid


if __name__ == "__main__":
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash)
    params_dict = vars(args)
    data_dir = params_dict.get("data_dir")
    precache_dir = params_dict.get("precache_dir")
    dataset_type = params_dict.get("dataset_type")

    train_dataset = CognitiveHeatMapGazeDataset(data_dir, precache_dir, "train", params_dict)
    print(train_dataset[312])
