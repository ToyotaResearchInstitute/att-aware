# Copyright 2020 Toyota Research Institute.  All rights reserved.
import collections
import torch
import numpy as np
import pickle
import os

from chm.dataset.chm_gaze_dataset import CHMGazeDataset
from chm.dataset.chm_att_awareness_dataset import CHMAttAwarenessDataset
from chm.dataset.chm_pairwise_gaze_dataset import CHMPairwiseGazeDataset
from chm.model.chm_net import CHMNet
from chm.losses.chm_loss import CHMLoss
from torch.utils.data import DataLoader, Subset, sampler
from utils.chm_consts import *
from functools import partial


def load_datasets(params_dict):
    gaze_datasets = collections.OrderedDict()
    awareness_datasets = collections.OrderedDict()
    pairwise_gaze_datasets = collections.OrderedDict()

    gaze_dataset_indices = None
    awareness_dataset_indices = None
    pairwise_gaze_dataset_indices = None

    # Create train datasets
    print("Creating Awareness Train DS")
    awareness_datasets["train"] = CHMAttAwarenessDataset("train", params_dict)
    # pass the awareness_dataset as the skip list so that it is not double counted during training.
    print("Creating Gaze Train DS")
    gaze_datasets["train"] = CHMGazeDataset(
        "train", params_dict, skip_list=awareness_datasets["train"].get_metadata_list()
    )
    print("Creating Pairwise Train DS")
    pairwise_gaze_datasets["train"] = CHMPairwiseGazeDataset("train", params_dict)
    if not params_dict["use_std_train_test_split"]:
        """
        splits are according dreyeve train/test video splits.
        """
        # Create test datasets
        print("Creating Awareness Test DS")
        awareness_datasets["test"] = CHMAttAwarenessDataset("test", params_dict)
        print("Creating Gaze Test DS")
        gaze_datasets["test"] = CHMGazeDataset("test", params_dict)
    else:
        """
        splits are according to query frame id.
        """
        if params_dict["load_indices_dict_path"] is None:
            # Create train and test split indices according to where the query frames are for gaze, awareness and pairwise-gaze datasets
            awareness_train_idx, awareness_test_idx = generate_train_test_split_indices(
                awareness_datasets["train"], params_dict
            )
            gaze_train_idx, gaze_test_idx = generate_train_test_split_indices(gaze_datasets["train"], params_dict)
            pairwise_gaze_train_idx, pairwise_gaze_test_idx = generate_train_test_split_indices(
                pairwise_gaze_datasets["train"], params_dict
            )
        else:
            assert os.path.exists(
                params_dict["load_indices_dict_path"]
            ), "Valid path to train/test split indices required."
            print("Loading train test split indices from file")
            with open(params_dict["load_indices_dict_path"], "rb") as fp:
                indices_dict = pickle.load(fp)

            gaze_train_idx = indices_dict["gaze_train_idx"]
            gaze_test_idx = indices_dict["gaze_test_idx"]
            awareness_train_idx = indices_dict["awareness_train_idx"]
            awareness_test_idx = indices_dict["awareness_test_idx"]
            pairwise_gaze_train_idx = indices_dict["pairwise_gaze_train_idx"]
            pairwise_gaze_test_idx = indices_dict["pairwise_gaze_test_idx"]

        print("Creating Awareness Test DS")
        awareness_datasets["test"] = Subset(awareness_datasets["train"], awareness_test_idx)
        awareness_datasets["train"] = Subset(awareness_datasets["train"], awareness_train_idx)

        print("Creating Gaze Test DS")
        gaze_datasets["test"] = Subset(gaze_datasets["train"], gaze_test_idx)
        gaze_datasets["train"] = Subset(gaze_datasets["train"], gaze_train_idx)

        print("Creating Pairwise Test DS")
        pairwise_gaze_datasets["test"] = Subset(pairwise_gaze_datasets["train"], pairwise_gaze_test_idx)
        pairwise_gaze_datasets["train"] = Subset(pairwise_gaze_datasets["train"], pairwise_gaze_train_idx)

        # Store the train and test indices in a dict. Will be cached for a particular training run.
        # To be later used for experiments, if needed
        gaze_dataset_indices = collections.OrderedDict()
        gaze_dataset_indices["train"] = gaze_train_idx
        gaze_dataset_indices["test"] = gaze_test_idx

        awareness_dataset_indices = collections.OrderedDict()
        awareness_dataset_indices["train"] = awareness_train_idx
        awareness_dataset_indices["test"] = awareness_test_idx

        pairwise_gaze_dataset_indices = collections.OrderedDict()
        pairwise_gaze_dataset_indices["train"] = pairwise_gaze_train_idx
        pairwise_gaze_dataset_indices["test"] = pairwise_gaze_test_idx

    return (gaze_datasets, awareness_datasets, pairwise_gaze_datasets), (
        gaze_dataset_indices,
        awareness_dataset_indices,
        pairwise_gaze_dataset_indices,
    )


def generate_train_test_split_indices(dataset, params_dict):
    """
    Creates the train/test split according to frame_id
    Parameters
    ----------
    dataset: Pytorch Dataset (CHMGazeDataset, CHMAttAwarenessDataset, CHMPairwiseGazeDataset)
        Gaze, Awareness or PairwiseGaze dataset with a metadata_list.
    params_dict: dict
        Dictionary containing the args passed from the training script.
    Returns
    -------
    train_idx: list
        Indices of the dataset that correspond to training data items
    test_idx: list
        Indices of the dataset that correspond to testing data items
    """

    def get_list_indices(a):
        d = collections.defaultdict(list)
        for i, j in enumerate(a):
            d[j].append(i)
        return d

    video_chunk_size = params_dict.get("video_chunk_size", 30.0)
    train_test_split_factor = params_dict.get("train_test_split_factor", 0.2)
    fps = params_dict.get("video_frame_rate", 25.0)
    chunk_size_in_frames = fps * video_chunk_size
    num_frames = 7501  # number of frames in each video
    # split the video into chunks of size chunk_size_in_frames
    video_chunks = list(divide_chunks(list(range(num_frames)), int(chunk_size_in_frames)))
    train_frames = []
    test_frames = []

    # extract training and test frame ids from each chunk and collate them according to train/test split factor
    for vc in video_chunks:
        train_l = round(len(vc) * (1.0 - train_test_split_factor))
        test_l = round(len(vc) * (train_test_split_factor))
        train_frames.extend(vc[:train_l])
        test_frames.extend(vc[-test_l:])

    # list of query frames for all data items in the dataset
    frame_list = [d[1] for d in dataset.get_metadata_list()]
    # dict containing the indices of each unique frame id
    list_indices_for_each_frame_id = get_list_indices(frame_list)

    train_idx = []
    test_idx = []
    # parse the dict and assign indices to train and test according whether the unique_frame_id is in train_frames or test_frames
    for unique_frame_id, indices_for_unique_frame_id in list_indices_for_each_frame_id.items():
        if unique_frame_id in train_frames:
            train_idx.extend(indices_for_unique_frame_id)
        elif unique_frame_id in test_frames:
            test_idx.extend(indices_for_unique_frame_id)

    return train_idx, test_idx


def create_dataloaders(gaze_datasets, awareness_datasets, pairwise_gaze_datasets, params_dict):
    num_test_samples = params_dict.get("num_test_samples", 1000)
    batch_size = params_dict.get("batch_size", 8)
    awareness_batch_size = params_dict.get("awareness_batch_size", 8)
    num_dl_workers = params_dict.get("num_workers", 0)

    def individual_data_loaders(datasets, num_test_samples, batch_size, num_dl_workers):
        dataloaders = collections.OrderedDict()
        for key in datasets:
            assert datasets[key] is not None, "Dataset provided is invalid"
            if key == "train":
                sampler = None
            elif key == "test":
                test_inds = np.random.choice(
                    range(len(datasets[key])), min(len(datasets[key]), num_test_samples), replace=False
                )
                # fix the indices for the testing set.
                sampler = SubsetSampler(test_inds)
            else:
                assert "Unspecified dataset type. Has to be either train or test"

            dataloaders[key] = DataLoader(
                datasets[key],
                batch_size=batch_size,
                shuffle=(sampler is None),
                sampler=sampler,
                drop_last=True,
                num_workers=num_dl_workers,
            )
        return dataloaders

    # train and test data loaders for each gaze, awareness and pairwise-gaze dataset
    print("Creating gaze ds dataloaders")
    gaze_dataloaders = individual_data_loaders(gaze_datasets, num_test_samples, batch_size, num_dl_workers)
    print("Creating awareness ds dataloaders")
    awareness_dataloaders = individual_data_loaders(
        awareness_datasets, num_test_samples, awareness_batch_size, num_dl_workers
    )
    if pairwise_gaze_datasets is not None:
        print("Creating pairwise-gaze ds dataloaders")
        pairwise_gaze_dataloaders = individual_data_loaders(
            pairwise_gaze_datasets, num_test_samples, batch_size, num_dl_workers
        )
    else:
        pairwise_gaze_dataloaders = None

    return gaze_dataloaders, awareness_dataloaders, pairwise_gaze_dataloaders


def create_model_and_loss_fn(params_dict):

    # create model
    model = CHMNet(params_dict)
    _, _, gaze_transform_prior_loss = model.get_modules()  # handle to functor to compute gaze transform loss

    # create loss function handle
    loss_fn = CHMLoss(params_dict, gt_prior_loss=partial(gaze_transform_prior_loss))

    load_model_path = params_dict["load_model_path"]
    # if model path provided, load stored model
    if load_model_path is not None:
        model.load_state_dict(torch.load(load_model_path))
        model.eval()

    return model, loss_fn


def save_model(state_dict, save_path):
    torch.save(state_dict, save_path)


def parse_data_item(data_dict, aux_info_list, gaze_corruption=None, gaze_correction=None, input_process_dict=None):
    batch_input = {}

    batch_input["road_image"] = data_dict[ROAD_IMAGE_0]  # garmin image
    batch_input["segmentation_mask_image"] = data_dict[SEGMENTATION_MASK_0]  # mask image
    batch_input["optic_flow_image"] = data_dict[OPTIC_FLOW_IMAGE_0]
    batch_input["should_train_input_gaze"] = data_dict[SHOULD_TRAIN_INPUT_GAZE_0]  # (B, T, L, 1)
    batch_input["input_gaze"] = data_dict[RESIZED_INPUT_GAZE_0]  # (B, T, L, 2) in network dimensions
    batch_input["normalized_input_gaze"] = data_dict[NORMALIZED_INPUT_GAZE_0]  # normalized (0,1) gaze #(B, T, L, 2)

    if gaze_corruption is not None:
        # apply gaze corruption to the input gaze, if available
        batch_input["normalized_input_gaze"] = gaze_corruption.corrupt_gaze(batch_input["normalized_input_gaze"])

    # This is a flag which will be used during inference to determine whether a batch needs to be ignored or not. For training, this will always be true
    should_use_batch = True
    # input process dict is typically used to provide a function handle to modify the input or determine whether the batch should be used or not
    # after corruption. Used in experiments
    if input_process_dict is not None and input_process_dict["functor"] is not None:
        input_process_functor = input_process_dict["functor"]
        batch_input, aux_info_list, should_use_batch = input_process_functor(
            batch_input, aux_info_list, input_process_dict["params"]
        )

    if gaze_correction is not None:
        #  apply gaze correction to the input gaze, if available
        batch_input["normalized_input_gaze"] = gaze_correction(batch_input["normalized_input_gaze"])

    # training target
    batch_target = data_dict[GROUND_TRUTH_GAZE_0]
    return batch_input, batch_target, aux_info_list, should_use_batch


# Data batch parsing functions.
def process_and_extract_data_batch(
    data_batch, gaze_corruption, gaze_correction, input_process_dict, device, has_pairwise_item=True
):
    (
        gaze_data_batch,
        awareness_data_batch,
        pairwise_gaze_data_batch_t,
        pairwise_gaze_data_batch_tp1,
    ) = parse_data_batch(
        data_batch,
        gaze_corruption=gaze_corruption,
        gaze_correction=gaze_correction,
        input_process_dict=input_process_dict,
        has_pairwise_item=has_pairwise_item,
    )

    # move input and target to appropriate device.

    sample_to_device(
        (
            gaze_data_batch,
            awareness_data_batch,
            pairwise_gaze_data_batch_t,
            pairwise_gaze_data_batch_tp1,
        ),
        device,
    )

    # extract individual batch inputs
    individual_batch_inputs = extract_individual_batch_input(
        gaze_data_batch,
        awareness_data_batch,
        pairwise_gaze_data_batch_t,
        pairwise_gaze_data_batch_tp1,
        has_pairwise_item=has_pairwise_item,
    )

    # extract annotation info
    awareness_batch_annotation_data = {
        "query_x": awareness_data_batch["batch_annotation_query_x"],
        "query_y": awareness_data_batch["batch_annotation_query_y"],
        "annotation_target": awareness_data_batch["batch_annotation_target"],
    }
    # post process and separate the individual batch inputs
    individual_batch_inputs = post_process_individual_batch_inputs(
        individual_batch_inputs, input_process_dict=input_process_dict, has_pairwise_item=has_pairwise_item
    )

    return individual_batch_inputs, awareness_batch_annotation_data


def parse_data_batch(data_batch, gaze_corruption, gaze_correction, input_process_dict, has_pairwise_item=True):
    if has_pairwise_item:
        (gaze_item, awareness_item, pairwise_gaze_item) = data_batch
    else:
        (gaze_item, awareness_item) = data_batch

    (gaze_data_dict, gaze_aux_info_list) = gaze_item
    (awareness_data_dict, awareness_aux_info_list) = awareness_item
    if has_pairwise_item:
        (pairwise_gaze_data_dict_t, pairwise_gaze_aux_info_list_t) = pairwise_gaze_item["data_t"]
        (pairwise_gaze_data_dict_tp1, pairwise_gaze_aux_info_list_tp1) = pairwise_gaze_item["data_tp1"]

    # parse gaze data items
    gaze_batch_input, gaze_batch_target, gaze_aux_info_list, gaze_should_use_batch = parse_data_item(
        gaze_data_dict,
        gaze_aux_info_list,
        gaze_corruption=gaze_corruption,
        gaze_correction=gaze_correction,
        input_process_dict=input_process_dict,
    )
    gaze_data_batch = {}
    gaze_data_batch["batch_input"] = gaze_batch_input
    gaze_data_batch["batch_target"] = gaze_batch_target
    gaze_data_batch["aux_info_list"] = gaze_aux_info_list
    gaze_data_batch["should_use_batch"] = gaze_should_use_batch

    # parse awareness data item
    (
        awareness_batch_input,
        awareness_batch_target,
        awareness_aux_info_list,
        awareness_should_use_batch,
    ) = parse_data_item(
        awareness_data_dict,
        awareness_aux_info_list,
        gaze_corruption=gaze_corruption,
        gaze_correction=gaze_correction,
        input_process_dict=input_process_dict,
    )

    awareness_data_batch = {}
    awareness_data_batch["batch_input"] = awareness_batch_input
    awareness_data_batch["batch_target"] = awareness_batch_target
    awareness_data_batch["aux_info_list"] = awareness_aux_info_list
    awareness_data_batch["should_use_batch"] = awareness_should_use_batch
    awareness_data_batch["batch_annotation_target"] = awareness_data_dict["att_annotation"]["anno_is_aware"]
    awareness_data_batch["batch_annotation_query_x"] = awareness_data_dict["att_annotation"]["query_x"]
    awareness_data_batch["batch_annotation_query_y"] = awareness_data_dict["att_annotation"]["query_y"]

    # parse pairwise data item at t
    if has_pairwise_item:
        (
            pairwise_gaze_batch_input_t,
            pairwise_gaze_batch_target_t,
            pairwise_gaze_aux_info_list_t,
            pairwise_gaze_should_use_batch_t,
        ) = parse_data_item(
            pairwise_gaze_data_dict_t,
            pairwise_gaze_aux_info_list_t,
            gaze_corruption=gaze_corruption,
            gaze_correction=gaze_correction,
            input_process_dict=input_process_dict,
        )
        pairwise_gaze_data_batch_t = {}
        pairwise_gaze_data_batch_t["batch_input"] = pairwise_gaze_batch_input_t
        pairwise_gaze_data_batch_t["batch_target"] = pairwise_gaze_batch_target_t
        pairwise_gaze_data_batch_t["aux_info_list"] = pairwise_gaze_aux_info_list_t
        pairwise_gaze_data_batch_t["should_use_batch"] = pairwise_gaze_should_use_batch_t
        # parse pairwise data item at tp1
        (
            pairwise_gaze_batch_input_tp1,
            pairwise_gaze_batch_target_tp1,
            pairwise_gaze_aux_info_list_tp1,
            pairwise_gaze_should_use_batch_tp1,
        ) = parse_data_item(
            pairwise_gaze_data_dict_tp1,
            pairwise_gaze_aux_info_list_tp1,
            gaze_corruption=gaze_corruption,
            gaze_correction=gaze_correction,
            input_process_dict=input_process_dict,
        )
        pairwise_gaze_data_batch_tp1 = {}
        pairwise_gaze_data_batch_tp1["batch_input"] = pairwise_gaze_batch_input_tp1
        pairwise_gaze_data_batch_tp1["batch_target"] = pairwise_gaze_batch_target_tp1
        pairwise_gaze_data_batch_tp1["aux_info_list"] = pairwise_gaze_aux_info_list_tp1
        pairwise_gaze_data_batch_tp1["should_use_batch"] = pairwise_gaze_should_use_batch_tp1
    else:
        pairwise_gaze_data_batch_t = None
        pairwise_gaze_data_batch_tp1 = None

    return gaze_data_batch, awareness_data_batch, pairwise_gaze_data_batch_t, pairwise_gaze_data_batch_tp1


def sample_to_device(data_batch_list, device):
    for data_batch in data_batch_list:
        if data_batch is not None:
            for key in data_batch["batch_input"].keys():
                data_batch["batch_input"][key] = data_batch["batch_input"][key].to(device)

            data_batch["batch_target"] = data_batch["batch_target"].to(device)
            if "batch_annotation_target" in data_batch:
                data_batch["batch_annotation_target"] = data_batch["batch_annotation_target"].to(device)


def extract_individual_batch_input(
    gaze_data_batch,
    awareness_data_batch,
    pairwise_gaze_data_batch_t,
    pairwise_gaze_data_batch_tp1,
    has_pairwise_item=True,
):
    # extract chm input from each batch
    gaze_batch_input = gaze_data_batch["batch_input"]
    awareness_batch_input = awareness_data_batch["batch_input"]

    # extract aux_info list from each batch
    gaze_aux_info_list = gaze_data_batch["aux_info_list"]
    awareness_aux_info_list = awareness_data_batch["aux_info_list"]

    # extract target from each batch
    gaze_batch_target = gaze_data_batch["batch_target"]
    awareness_batch_target = awareness_data_batch["batch_target"]

    # extract should_use_batch
    gaze_batch_should_use_batch = gaze_data_batch["should_use_batch"]
    awareness_batch_should_use_batch = awareness_data_batch["should_use_batch"]

    if has_pairwise_item:
        pairwise_gaze_batch_input_t = pairwise_gaze_data_batch_t["batch_input"]
        pairwise_gaze_batch_input_tp1 = pairwise_gaze_data_batch_tp1["batch_input"]
        pairwise_gaze_aux_info_list_t = pairwise_gaze_data_batch_t["aux_info_list"]
        pairwise_gaze_aux_info_list_tp1 = pairwise_gaze_data_batch_tp1["aux_info_list"]
        pairwise_gaze_batch_target_t = pairwise_gaze_data_batch_t["batch_target"]
        pairwise_gaze_batch_target_tp1 = pairwise_gaze_data_batch_tp1["batch_target"]
        pairwise_gaze_batch_should_use_batch_t = pairwise_gaze_data_batch_t["should_use_batch"]
        pairwise_gaze_batch_should_use_batch_tp1 = pairwise_gaze_data_batch_tp1["should_use_batch"]
    else:
        pairwise_gaze_batch_input_t = None
        pairwise_gaze_batch_input_tp1 = None
        pairwise_gaze_aux_info_list_t = None
        pairwise_gaze_aux_info_list_tp1 = None
        pairwise_gaze_batch_target_t = None
        pairwise_gaze_batch_target_tp1 = None
        pairwise_gaze_batch_should_use_batch_t = None
        pairwise_gaze_batch_should_use_batch_tp1 = None

    return (
        gaze_batch_input,
        awareness_batch_input,
        pairwise_gaze_batch_input_t,
        pairwise_gaze_batch_input_tp1,
        gaze_aux_info_list,
        awareness_aux_info_list,
        pairwise_gaze_aux_info_list_t,
        pairwise_gaze_aux_info_list_tp1,
        gaze_batch_target,
        awareness_batch_target,
        pairwise_gaze_batch_target_t,
        pairwise_gaze_batch_target_tp1,
        gaze_batch_should_use_batch,
        awareness_batch_should_use_batch,
        pairwise_gaze_batch_should_use_batch_t,
        pairwise_gaze_batch_should_use_batch_tp1,
    )


def post_process_individual_batch_inputs(individual_batch_inputs, input_process_dict, has_pairwise_item=True):
    # post process input before model forward (typically used in controlled experiments to modify input)
    (
        gaze_batch_input,
        awareness_batch_input,
        pairwise_gaze_batch_input_t,
        pairwise_gaze_batch_input_tp1,
        gaze_aux_info_list,
        awareness_aux_info_list,
        pairwise_gaze_aux_info_list_t,
        pairwise_gaze_aux_info_list_tp1,
        gaze_batch_target,
        awareness_batch_target,
        pairwise_gaze_batch_target_t,
        pairwise_gaze_batch_target_tp1,
        gaze_batch_should_use_batch,
        awareness_batch_should_use_batch,
        pairwise_gaze_batch_should_use_batch_t,
        pairwise_gaze_batch_should_use_batch_tp1,
    ) = individual_batch_inputs
    if input_process_dict is not None:
        if "post_parse_data_item" in input_process_dict:
            if "functor" in input_process_dict["post_parse_data_item"]:
                post_parse_data_item_functor = input_process_dict["post_parse_data_item"]["functor"]
                gaze_batch_input, gaze_aux_info_list, gaze_batch_should_use_batch = post_parse_data_item_functor(
                    gaze_batch_input,
                    gaze_aux_info_list,
                    input_process_dict["post_parse_data_item"]["params"],
                )

                (
                    awareness_batch_input,
                    awareness_aux_info_list,
                    awareness_batch_should_use_batch,
                ) = post_parse_data_item_functor(
                    awareness_batch_input,
                    awareness_aux_info_list,
                    input_process_dict["post_parse_data_item"]["params"],
                )

                if has_pairwise_item:
                    (
                        pairwise_gaze_batch_input_t,
                        pairwise_gaze_aux_info_list_t,
                        pairwise_gaze_batch_should_use_batch_t,
                    ) = post_parse_data_item_functor(
                        pairwise_gaze_batch_input_t,
                        pairwise_gaze_aux_info_list_t,
                        input_process_dict["post_parse_data_item"]["params"],
                    )

                    (
                        pairwise_gaze_batch_input_tp1,
                        pairwise_gaze_aux_info_list_tp1,
                        pairwise_gaze_batch_should_use_batch_tp1,
                    ) = post_parse_data_item_functor(
                        pairwise_gaze_batch_input_tp1,
                        pairwise_gaze_aux_info_list_tp1,
                        input_process_dict["post_parse_data_item"]["params"],
                    )

    return (
        gaze_batch_input,
        awareness_batch_input,
        pairwise_gaze_batch_input_t,
        pairwise_gaze_batch_input_tp1,
        gaze_aux_info_list,
        awareness_aux_info_list,
        pairwise_gaze_aux_info_list_t,
        pairwise_gaze_aux_info_list_tp1,
        gaze_batch_target,
        awareness_batch_target,
        pairwise_gaze_batch_target_t,
        pairwise_gaze_batch_target_tp1,
        gaze_batch_should_use_batch,
        awareness_batch_should_use_batch,
        pairwise_gaze_batch_should_use_batch_t,
        pairwise_gaze_batch_should_use_batch_tp1,
    )


# utility functions
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


class SubsetSampler(sampler.Sampler):
    """Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
