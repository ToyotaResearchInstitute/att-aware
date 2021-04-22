import os
import torch
import collections
import pickle
import tensorboardX

from chm.utils.trainer_utils import create_model_and_loss_fn, load_datasets, create_dataloaders


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

    def __init__(self, params_dict, session_hash=None):
        super().__init__()
        self.params_dict = params_dict
        self.log_dir = os.path.join(params_dict["log_dir"], session_hash)
        if not self.log_dir is None:
            if params_dict["training_hash"] is None:
                params_dict["training_hash"] = self.log_dir
            logger = tensorboardX.SummaryWriter(
                log_dir=self.log_dir + params_dict["training_hash"], comment=params_dict["training_hash"]
            )
        self.logger = logger

        if self.params_dict["no_cuda"] or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # create model and loss function and put it on the correct device.
        self.model, self.loss_fn = create_model_and_loss_fn(self.params_dict)
        # self.loss_fn.to(device)
        # self.model.to(device)

        self.gaze_datasets = self.awareness_datasets = self.pairwise_gaze_datasets = None
        self.gaze_dataset_indices = self.awareness_dataset_indices = self.pairwise_gaze_dataset_indices = None
        # create gaze, awareness and pairwise-gaze datasets
        (self.gaze_datasets, self.awareness_datasets, self.pairwise_gaze_datasets), (
            gaze_dataset_indices,
            awareness_dataset_indices,
            pairwise_gaze_dataset_indices,
        ) = load_datasets(self.params_dict)

        # if use_std_train_test_split save the indices
        self.save_train_test_indices()

        # create gaze, awareness and pairwise-gaze dataloaders
        self.gaze_dataloaders, self.awareness_dataloaders, self.pairwise_gaze_dataloaders = create_dataloaders(
            gaze_datasets, awareness_datasets, pairwise_gaze_datasets, params_dict
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

    def save_train_test_indices(self):
        if self.params_dict["use_std_train_test_split"]:
            indices_dict = collections.OrderedDict()
            indices_dict["gaze_train_idx"] = self.gaze_dataset_indices["train"]
            indices_dict["gaze_test_idx"] = self.gaze_dataset_indices["test"]
            indices_dict["awareness_train_idx"] = self.awareness_dataset_indices["train"]
            indices_dict["awareness_test_idx"] = self.awareness_dataset_indices["test"]
            indices_dict["pairwise_gaze_train_idx"] = self.pairwise_gaze_dataset_indices["train"]
            indices_dict["pairwise_gaze_test_idx"] = self.pairwise_gaze_dataset_indices["test"]

            indices_dict_folder = os.path.join(self.log_dir, "indices_dict_folder")
            os.makedirs(indices_dict_folder, exist_ok=True)
            indices_dict_filename = os.path.join(indices_dict_folder, "indices_dict.pkl")
            with open(indices_dict_filename, "wb") as fp:
                pickle.dump(indices_dict, fp)


def parse_data_item():
    pass