import os
import torch
import uuid

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

    model_wrapper = ModelWrapper(params_dict, session_hash)

    trainer = CHMTrainer(params_dict)

    trainer.fit(model_wrapper)


if __name__ == "__main__":
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash)
    train(args, session_hash)