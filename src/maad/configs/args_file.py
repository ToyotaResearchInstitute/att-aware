# Copyright 2021 Toyota Research Institute.  All rights reserved.

"""
Default maad arguments for training, loss function, dataset creation, inference
"""

import argparse
import os
import json


def parse_arguments(session_hash, additional_argument_setters=[]):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler="resolve")

    ########################################### DATASET TYPES AND DIRECTORY PATHS ###########################################

    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        default=os.path.join(os.path.expanduser("~"), "data", "dreyeve"),
        help="Full path to directory containing the DREYEVE videos and the gaze data",
    )
    parser.add_argument(
        "--precache_dir",
        dest="precache_dir",
        default=os.path.join(os.path.expanduser("~"), "maad_cache"),
        help="Full path to the directory in which the cached images, segmentation masks and optic flow are stored",
    )
    parser.add_argument(
        "--att_awareness_labels",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "data", "MAAD_ATT_AWARENESS_LABELS.csv"),
        help="Path to CSV file containing the attended awareness annotations",
    )
    parser.add_argument(
        "--all_gaze_data_dict",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "data", "all_videos_subjects_tasks_gaze_data_dict.pkl"),
        help="Path to PKL file containing all the gaze data for all videos, subjects, tasks",
    )
    parser.add_argument(
        "--dataset_type",
        dest="dataset_type",
        default="train",
        help="Argument indicating the dataset type when creating dataset instances during training. [train, test]",
    )
    parser.add_argument(
        "--inference_ds_type",
        dest="inference_ds_type",
        default="test",
        help="Argument indicating the dataset type when creating dataset instances during inference. [train, test]",
    )
    parser.add_argument(
        "--load_model_path",
        dest="load_model_path",
        default=None,
        help="Full path to model to be loaded in for the experiments. ",
    )
    parser.add_argument(
        "--load_indices_dict_path",
        dest="load_indices_dict_path",
        default=None,
        help="Full path to folder (generated during training) containing the train/test split indices to be loaded in during inference.",
    )
    parser.add_argument(
        "--log_dir",
        dest="log_dir",
        default=os.path.join(os.path.expanduser("~"), "maad", "logs", session_hash),
        help="Full path to the the directory where the tensorboard logs will be written",
    )
    parser.add_argument(
        "--training_hash",
        dest="training_hash",
        default="maad_train",
        help="string describing meaningful descriptor string for training session",
    )
    parser.add_argument(
        "--orig_road_image_dims",
        type=int,
        default=[3, 1080, 1920],
        help="Dimensions of the full original road_image in pixels",
    )

    ########################################### TRAINING ARGS ###########################################
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1234,
        help="Random seed for numpy and pytorch. If None is provided, then automatically generated in the trainer.",
    )
    parser.add_argument(
        "--no_cuda",
        dest="no_cuda",
        action="store_true",
        help="When no_cuda flag is used, model training will happen on CPU",
    )
    parser.add_argument(
        "--use_std_train_test_split",
        dest="use_std_train_test_split",
        action="store_true",
        default=False,
        help="Flag for using standard train/test split during model training. If false, training proceeds using the dreyeve train/test split (at the video id level",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        default=4,
        help="Batch size used for the gaze dataset and the overlap dataset",
    )
    parser.add_argument(
        "--max_inference_num_batches",
        action="store",
        type=int,
        default=20,
        help="max number of batches during inference",
    )
    parser.add_argument(
        "--awareness_batch_size", action="store", type=int, default=8, help="Batch size used for the awareness dataset",
    )
    parser.add_argument(
        "--batch_aggregation_size",
        action="store",
        type=int,
        default=8,
        help="Number of batches for which the gradients are accumulated before performing backprop",
    )
    parser.add_argument(
        "--learning_rate",
        action="store",
        type=float,
        default=5e-3,
        help="Initial learning rate used for the main training",
    )
    parser.add_argument(
        "--dropout_ratio",
        action="store",
        type=json.loads,
        default='{"driver_facing":0.2, "optic_flow":0.0}',
        help="Dictionary specifying the the dropout ratio for the side channel inputs during training. Each value has to be in [0.0, 1.0]",
    )
    parser.add_argument(
        "--force_dropout_list",
        nargs="*",
        type=str,
        default=["driver_facing"],
        help="List containing the side channels to be forced to dropout during inference, visualization and testing.",
    )
    parser.add_argument(
        "--dropout_ratio_external_inputs",
        action="store",
        type=float,
        default=0.0,
        help="Dropout_ratio for ALL side channel input. When dropout is applied ALL side channel input is zeroed out. Only active raining",
    )
    parser.add_argument(
        "--lr_update_num",
        action="store",
        type=int,
        default=2000,
        help="Interval (num of batches) before the learning rate scheduler gets updates",
    )
    parser.add_argument(
        "--learning_rate_decay",
        action="store",
        type=float,
        default=0.97,
        help="Learning rate decay factor for the scheduler which is activated once every lr_update_num batches.",
    )
    parser.add_argument(
        "--lr_min_bound",
        action="store",
        type=float,
        default=1e-4,
        help="Minimum bound for learning rate. The LR scheduler will not decay the LR below this value",
    )
    parser.add_argument(
        "--visualize_frequency",
        action="store",
        type=int,
        default=200,
        help="Interval (in number of batches) at which tensorboard visualization of the output is performed. Has to be positive.",
    )
    parser.add_argument(
        "--save_interval",
        action="store",
        type=int,
        default=1000,
        help="Interval (in number of batches) at which the model is saved",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        action="store",
        type=int,
        default=1000,
        help="Interval (in number of batches) at which the testing phase is activated during training",
    )
    parser.add_argument(
        "--num_workers",
        action="store",
        type=int,
        default=0,
        help="Number of dataloader workers used by the train, test and vis dataloaders.",
    )
    parser.add_argument(
        "--max_epochs",
        action="store",
        type=int,
        default=100,
        help="Number of max epochs used during the main training experiment",
    )
    parser.add_argument(
        "--max_overall_batch_during_training",
        action="store",
        type=int,
        default=None,
        help="max number of batches during training experiments",
    )
    parser.add_argument(
        "--train_test_split_factor",
        action="store",
        type=float,
        default=0.2,
        help="When running training using the train/test split of a single dataset, this arg specifies the proportion used for the test set. (0.0, 1.0)",
    )
    ########################################### FLAGS USED DURING TRAINING ###########################################
    parser.add_argument(
        "--enable_amp",
        action="store_true",
        default=False,
        help="Flag for using automatic mixed precision when training",
    )
    parser.add_argument(
        "--add_optic_flow",
        action="store_true",
        default=False,
        help="Flag to add optic flow network as a side channel input",
    )
    parser.add_argument(
        "--nograd_encoder",
        action="store_true",
        default=False,
        help="Flag for not computing the gradients for the encoder layers from training. Typically used when --use_s3d_encoders is not used",
    )
    parser.add_argument(
        "--no_maps_visualize",
        dest="no_maps_visualize",
        action="store_true",
        default=False,
        help="When set, the gaze and awareness maps are not saved.",
    )
    parser.add_argument(
        "--no_save_model",
        dest="no_save_model",
        action="store_true",
        default=False,
        help="when set, model saving does not happen. ",
    )
    parser.add_argument(
        "--no_run_test",
        dest="no_run_test",
        action="store_true",
        default=False,
        help="When set, the testing phase is skipped during the training run.",
    )
    parser.add_argument(
        "--num_visualization_examples",
        dest="num_visualization_examples",
        default=8,
        type=int,
        help="Number of examples (gaze/awareness maps) visualized during training",
    )
    parser.add_argument(
        "--num_test_samples",
        dest="num_test_samples",
        default=1000,
        type=int,
        help="Number of samples (fixed set) used during the testing phase",
    )

    ########################################### DATASET ARGS ###########################################
    parser.add_argument(
        "--temporal_downsample_factor",
        action="store",
        type=int,
        default=6,
        help="Temporal downsampling used for the sequences. The number indicates the hopsize when generating the sequences",
    )
    parser.add_argument(
        "--aspect_ratio_reduction_factor",
        action="store",
        type=float,
        default=8.0,
        help="Factor by which original images size will be scaled for network input",
    )
    parser.add_argument(
        "--train_sequence_length",
        action="store",
        type=int,
        default=20,
        help="Length of the video snippet used for training in frames",
    )
    parser.add_argument(
        "--test_sequence_length",
        action="store",
        type=int,
        default=20,
        help="Length of the video snippet used for testing in frames",
    )
    parser.add_argument(
        "--vis_sequence_length",
        action="store",
        type=int,
        default=20,
        help="Length of the video snippet used for visualization in frames",
    )
    parser.add_argument(
        "--train_sequence_ids",
        nargs="*",
        type=int,
        default=[6, 7, 10, 11, 26, 35],
        help="List containing the dreyeve video ids used for training",
    )
    parser.add_argument(
        "--test_sequence_ids",
        nargs="*",
        type=int,
        default=[53, 60],
        help="List containing the dreyeve video ids used for testing",
    )
    parser.add_argument(
        "--vis_sequence_ids",
        nargs="*",
        type=int,
        default=[53, 60],
        help="List containing the dreyeve video ids used for visualization",
    )
    parser.add_argument(
        "--train_subject_ids",
        nargs="*",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        help="List containing the subject ids to be considered for the train dataset",
    )
    parser.add_argument(
        "--test_subject_ids",
        nargs="*",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        help="List containing the subject ids to be considered for the test dataset",
    )
    parser.add_argument(
        "--vis_subject_ids",
        nargs="*",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        help="List containing the subject ids to be considered for the test dataset",
    )
    parser.add_argument(
        "--train_task_ids",
        nargs="*",
        type=str,
        default=["roadonly", "control", "readingtext", "blurred", "flipped"],
        help="List containing the cognitive task modifiers to be considered for the train dataset",
    )
    parser.add_argument(
        "--test_task_ids",
        nargs="*",
        type=str,
        default=["roadonly", "control", "readingtext", "blurred", "flipped"],
        help="List containing the cognitive task modifiers to be considered for the test dataset",
    )
    parser.add_argument(
        "--vis_task_ids",
        nargs="*",
        type=str,
        default=["roadonly", "control", "readingtext", "blurred", "flipped"],
        help="List containing the cognitive task modifiers to be considered for the visualization dataset",
    )
    parser.add_argument(
        "--retrieve_panoptic_masks",
        action="store_true",
        default=True,
        help="Flag for retrieving panoptic masks in the getitem",
    )
    parser.add_argument(
        "--request_auxiliary_info", action="store_true", default=True, help="Flag for requesting auxiliary info"
    )
    parser.add_argument(
        "--retrieve_optic_flow",
        action="store_true",
        default=False,
        help="Flag for retrieving optic flow to be used a side channel input to the decoder",
    )

    ################################### NETWORK ARGS #########################################3

    parser.add_argument(
        "--use_s3d",
        action="store_true",
        default=False,
        help="Flag for opting for separable 3d convolutions in the encoder. When False, the encoder will consist of only Conv2D from ResNet",
    )
    parser.add_argument(
        "--num_latent_layers",
        action="store",
        type=int,
        default=6,
        help="Number of latent layers in the common predictor used for emitting the final awareness and gaze maps.",
    )
    parser.add_argument(
        "--reduced_middle_layer_size",
        action="store",
        type=int,
        default=512,
        help="Size of the maximum features of the Conv3D with k=1 after encoder layer.",
    )
    parser.add_argument(
        "--decoder_layer_features",
        nargs="*",
        type=int,
        default=[16, 32, 64, 128, 256],
        help="Number of features in each of the decoder units in the Decoder. When using s3D, the list should contain 5 elements, otherwise 4",
    )
    ########################################## LOSS FUNCTION ARGS #######################################

    parser.add_argument(
        "--gt_prior_loss_coeff", action="store", type=float, default=0.0, help="Coeff on gaze transform prior loss",
    )
    parser.add_argument(
        "--awareness_of_gaze_weight_factor_max",
        action="store",
        type=float,
        default=1.0,
        help="Max value for the awareness of gaze weight factor",
    )
    parser.add_argument(
        "--unnormalized_gaze_loss_coeff",
        action="store",
        type=float,
        default=1e-5,
        help="Coeff on unnormalized_gaze_loss",
    )
    parser.add_argument(
        "--consistency_coeff_gaze",
        action="store",
        type=float,
        default=1e7,
        help="Coeff for consistency smoothness coeff for gaze. ",
    )
    parser.add_argument(
        "--consistency_coeff_awareness",
        action="store",
        type=float,
        default=10,
        help="Coeff for consistency smoothness coeff for awareness. ",
    )
    parser.add_argument(
        "--common_predictor_map_loss_coeff",
        action="store",
        type=float,
        default=1e-5,
        help="Coeff on common_predictor_map_loss_coeff",
    )
    parser.add_argument(
        "--gaze_data_coeff",
        action="store",
        type=float,
        default=1.2,
        help="Coeff on main loss term gaze log-probability",
    )
    parser.add_argument(
        "--gaze_spatial_regularization_coeff",
        action="store",
        type=float,
        default=5e10,
        help="Coeff for road image based spatial regularization coefficient for gaze",
    )
    parser.add_argument(
        "--gaze_temporal_regularization_coeff",
        action="store",
        type=float,
        default=0.0,
        help="Coeff for road image based temporal regularization coefficient for gaze",
    )

    ######## AWARENESS LOSS TERMS ##############
    parser.add_argument(
        "--awareness_label_coeff",
        action="store",
        type=float,
        default=1.0,
        help="Coeff for loss computed on the attended awareness annotations",
    )
    parser.add_argument(
        "--awareness_loss_type",
        type=str,
        default="huber_loss",
        help="Type of awareness loss [huber_loss, squared_loss]",
    )
    parser.add_argument(
        "--awareness_label_loss_patch_half_size",
        dest="awareness_label_loss_patch_half_size",
        type=int,
        default=4,
        help="Half of patch size for label loss computation",
    )
    parser.add_argument(
        "--awareness_at_gaze_points_loss_coeff",
        action="store",
        type=float,
        default=1,
        help="Coeff for awareness at gaze points loss",
    )
    parser.add_argument(
        "--awareness_steady_state_coeff",
        action="store",
        type=float,
        default=0.01,
        help="Regularization on awareness steady state",
    )
    parser.add_argument(
        "--awareness_spatial_regularization_coeff",
        action="store",
        type=float,
        default=100.0,
        help="Spatial regularization coefficient for awarensss",
    )
    parser.add_argument(
        "--awareness_temporal_regularization_coeff",
        action="store",
        type=float,
        default=0.0,
        help="Temporal regularization coefficient for awarensss",
    )
    parser.add_argument(
        "--regularization_eps",
        action="store",
        type=float,
        default=1e-3,
        help="eps to be added for diffusivity computation",
    )
    parser.add_argument(
        "--negative_difference_coeff",
        action="store",
        type=float,
        default=20.0,
        help="Negative difference coeff for asymmetric nonlinearity used in the awareness temporal regularization",
    )
    parser.add_argument(
        "--positive_difference_coeff",
        action="store",
        type=float,
        default=1.0,
        help="Positive difference coeff for asymmetric nonlinearity used in the awareness temporal regularization",
    )
    parser.add_argument(
        "--awareness_decay_coeff",
        action="store",
        type=float,
        default=1500000,
        help="Coeff for the decay loss term used for awareness",
    )
    parser.add_argument(
        "--awareness_decay_alpha",
        action="store",
        type=float,
        default=0.2,
        help="The decay coefficient used in the decay loss term for awareness",
    )
    parser.add_argument(
        "--optic_flow_temporal_smoothness_coeff",
        action="store",
        type=float,
        default=600,
        help="Coeff for optic flow based smoothness",
    )
    parser.add_argument(
        "--optic_flow_temporal_smoothness_decay",
        action="store",
        type=float,
        default=0.5,
        help="The decay coefficient between steps",
    )
    parser.add_argument(
        "--gaussian_kernel_size",
        action="store",
        type=int,
        default=2,
        help="Denotes the kernel size used for the gaussian filter for awareness at gaze loss function",
    )
    parser.add_argument(
        "--gaze_bias_std",
        action="store",
        type=float,
        default=0.00000001,
        help="Amount of bias to be used for the gaze corruption",
    )
    # uses sqrt(1.5^2+2^2)/72 for 1.5 deg error of foveated/center, and 2 deg for the gaze tracker
    parser.add_argument(
        "--gaze_noise_std",
        action="store",
        type=float,
        default=0.0347222222,
        help="Std for the Gaussian noise added in the gaze corruption",
    )
    parser.add_argument(
        "--sig_scale_factor",
        action="store",
        type=int,
        default=1,
        help="Factor by which sigma for gaussian kernel will be multiplied",
    )
    parser.add_argument(
        "--video_chunk_size",
        action="store",
        type=float,
        default=30.0,
        help="Each video will be split into chunks of length video_chunk_size before splitting into train and test indices",
    )
    parser.add_argument(
        "--fixed_gaze_list_length",
        dest="fixed_gaze_list_length",
        type=int,
        default=3,
        help="Length of gaze list returned by getitem functor. Should be in [1, 10]",
    )

    for setter in additional_argument_setters:
        setter(parser)
    args = parser.parse_args()
    return args
