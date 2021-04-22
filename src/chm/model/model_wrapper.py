import torch

from chm.utils.trainer_utils import create_model_and_loss_fn, load_datasets


class ModelWrapper(torch.nn.Module):
    """
    Top-level torch.nn.Module wrapper around a CHM modesl.
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

        # create model and loss function and put it on the correct device.
        self.model, self.loss_fn = create_model_and_loss_fn(self.params_dict)
        self.loss_fn.to(device)
        self.model.to(device)

        # load gaze, awareness and pairwise-gaze datasets
        gaze_datasets, awareness_datasets, pairwise_gaze_datasets = load_datasets(self.params_dict)
        gaze_dataloaders, awareness_dataloaders, pairwise_gaze_dataloaders = create_dataloaders(
            gaze_datasets, awareness_datasets, pairwise_gaze_datasets
        )

    def configure_optimizers(self):

        # ensure all the correct parameters have requires_grad
        # initialize Adam and the scheduler
        pass

    def training_step(self):
        # parse all the batches properly.
        # pass it through model.
        # compute loss functions.
        pass

    def testing_step(self):
        pass

    def visualization_step(self):
        pass