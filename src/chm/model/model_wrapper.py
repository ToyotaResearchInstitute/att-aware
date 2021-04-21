import torch

from chm.utils.trainer_utils import create_model_and_loss_fn, load_datasets


class ModelWrapper(torch.nn.Module):
    """
    Top-level torch.nn.Module wrapper around a CognitiveHeatMap modesl.
    Designed to use models with high-level Trainer classes (cf. trainers/).

    Parameters
    ----------
    config : dict
        params dict containing the args from args_file.py
    logger : TensorboardX instance
        tensorboardx logger used for logging loss curves and visualizations
    """

    def __init__(self, params_dict, logger=None):
        super().__init__()
        self.params_dict = params_dict
        self.logger = logger
        if self.params_dict["no_cuda"] or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        self.model, self.loss_fn = create_model_and_loss_fn(
            self.params_dict
        )  # create model and loss function and put it on the correct device.
        self.loss_fn.to(device)
        self.model.to(device)

        gaze_datasets, awareness_datasets, pairwise_gaze_datasets = load_datasets(
            self.params_dict
        )  # load gaze, awareness and pairwise-gaze datasets
