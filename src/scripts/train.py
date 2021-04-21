import os
import torch
import uuid
import tensorboardX

from chm.configs.args_file import parse_arguments
from chm.model.model_wrapper import ModelWrapper
from chm.trainers.cognitive_heatmap_trainer import CognitiveHeatmapTrainer


def train(args, session_hash):
    params_dict = vars(args)
    # select training device
    if params_dict["no_cuda"] or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    log_dir = os.path.join(params_dict["log_dir"], session_hash)
    if not log_dir is None:
        if params_dict["training_hash"] is None:
            params_dict["training_hash"] = self.log_dir
        logger = tensorboardX.SummaryWriter(
            log_dir=log_dir + params_dict["training_hash"], comment=params_dict["training_hash"]
        )

    model_wrapper = ModelWrapper(params_dict, logger)

    trainer = CognitiveHeatmapTrainer(params_dict)

    trainer.fit(model_wrapper)


if __name__ == "__main__":
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash)
    train(args, session_hash)