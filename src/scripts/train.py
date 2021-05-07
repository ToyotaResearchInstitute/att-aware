# Copyright 2020 Toyota Research Institute.  All rights reserved.
import os
import torch
import uuid
import random
import numpy as np

from chm.configs.args_file import parse_arguments
from chm.model.model_wrapper import ModelWrapper
from chm.trainers.cognitive_heatmap_trainer import CHMTrainer


def train(args, session_hash):
    params_dict = vars(args)
    # select training device
    if params_dict["no_cuda"] or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # set random seed
    # if random seed is provided use it if not create a random seed
    random_seed = params_dict["random_seed"] or random.randint(1, 10000)
    print("Random Seed used is ", random_seed)
    params_dict["random_seed"] = random_seed  # update the params dict
    np.random.seed(params_dict["random_seed"])
    torch.manual_seed(params_dict["random_seed"])

    # create model wrapper instance
    model_wrapper = ModelWrapper(params_dict, session_hash)

    # instantiate trainer class
    trainer = CHMTrainer(params_dict)

    # start training
    trainer.fit(model_wrapper)


if __name__ == "__main__":
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash)
    train(args, session_hash)