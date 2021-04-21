import torch

from chm.utils.trainer_utils import create_model_and_loss_fn


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

        self.model, self.loss_fn = create_model_and_loss_fn(params_dict)