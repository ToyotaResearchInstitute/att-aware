import collections
import torch

from chm.dataset.chm_gaze_dataset import CognitiveHeatMapGazeDataset
from chm.dataset.chm_att_awareness_dataset import CognitiveHeatMapAttAwarenessDataset
from chm.dataset.chm_pairwise_gaze_dataset import CognitiveHeatMapPairwiseGazeDataset
from chm.model.cognitive_heatnet import CognitiveHeatNet
from chm.losses.cognitive_heatnet_loss import CognitiveHeatNetLoss


def load_datasets(params_dict):
    gaze_datasets = collections.OrderedDict()
    awareness_datasets = collections.OrderedDict()
    pairwise_datasets = collections.OrderedDict()

    if not params_dict["use_std_train_test_split"]:
        """
        splits are according dreyeve train/test video splits.
        """
        awareness_datasets["train"] = CognitiveHeatMapAttAwarenessDataset("train", params_dict)
        awareness_datasets["test"] = CognitiveHeatMapAttAwarenessDataset("test", params_dict)

        gaze_datasets["train"] = CognitiveHeatMapGazeDataset("train", params_dict)
        gaze_datasets["test"] = CognitiveHeatMapGazeDataset("test", params_dict)
        gaze_datasets["vis"] = CognitiveHeatMapGazeDataset("vis", params_dict)

        pairwise_datasets["train"] = CognitiveHeatMapPairwiseGazeDataset("train", params_dict)
    else:
        pass

    import IPython

    IPython.embed(banner1="check par")


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
