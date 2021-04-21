import collections
import torch

from chm.dataset.chm_gaze_dataset import CHMGazeDataset
from chm.dataset.chm_att_awareness_dataset import CHMAttAwarenessDataset
from chm.dataset.chm_pairwise_gaze_dataset import CHMPairwiseGazeDataset
from chm.model.cognitive_heatnet import CognitiveHeatNet
from chm.losses.cognitive_heatnet_loss import CognitiveHeatNetLoss


def load_datasets(params_dict):
    gaze_datasets = collections.OrderedDict()
    awareness_datasets = collections.OrderedDict()
    pairwise_gaze_datasets = collections.OrderedDict()

    # Create train datasets
    awareness_datasets["train"] = CHMAttAwarenessDataset("train", params_dict)
    # pass the awareness_dataset as the skip list so that it is not double counted during training.
    gaze_datasets["train"] = CHMGazeDataset("train", params_dict, awareness_datasets["train"].get_metadata_list())
    pairwise_gaze_datasets["train"] = CHMPairwiseGazeDataset("train", params_dict)

    if not params_dict["use_std_train_test_split"]:
        """
        splits are according dreyeve train/test video splits.
        """
        # Create test datasets
        awareness_datasets["test"] = CHMAttAwarenessDataset("test", params_dict)
        gaze_datasets["test"] = CHMGazeDataset("test", params_dict)

    else:
        """
        splits are according to query frame id.
        """

        # Create train and test split indices according to where the query frames are.
        awareness_train_idx, awareness_test_idx = generate_train_test_split_indices(
            awareness_datasets["train"], params_dict
        )
        gaze_train_idx, gaze_test_idx = generate_train_test_split_indices(gaze_datasets["train"], params_dict)

        awareness_datasets["test"] = Subset(awareness_datasets["train"], awareness_test_idx)
        awareness_datasets["train"] = Subset(awareness_datasets["train"], awareness_train_idx)

        gaze_datasets["test"] = Subset(gaze_datasets["train"], gaze_test_idx)
        gaze_datasets["train"] = Subset(gaze_datasets["train"], gaze_test_idx)

    import IPython

    IPython.embed(banner1="check par")


def generate_train_test_split_indices(dataset, params_dict):
    train_idx = None
    test_idx = None
    return train_idx, test_idx


def create_dataloaders(gaze_datasets, awareness_datasets, pairwise_datasets):
    gaze_dataloaders = collections.OrderedDict()
    awareness_dataloaders = collections.OrderedDict()
    pairwise_gaze_dataloaders = collections.OrderedDict()
    return gaze_dataloaders, awareness_dataloaders, pairwise_gaze_dataloaders


def create_model_and_loss_fn(params_dict):

    model = CognitiveHeatNet(params_dict)
    # _, _, gaze_transform_prior_loss = model.get_modules()
    loss_fn = CognitiveHeatNetLoss(params_dict)

    load_model_path = params_dict["load_model_path"]
    if load_model_path is not None:
        model.load_state_dict(torch.load(load_model_path))
        model.eval()

    return model, loss_fn


def parse_data_item():
    pass
