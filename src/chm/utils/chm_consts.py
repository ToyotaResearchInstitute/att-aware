# Copyright 2021 Toyota Research Institute.  All rights reserved.

"""
This file details the field names used in datasets' for CHM training as well other constants. 
"""

from enum import Enum

MAX_NUM_VIDEO_FRAMES = 7501
ALL_DREYEVE_VIDEO_IDS = [6, 7, 10, 11, 26, 35, 53, 60]
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

# optic flow parameters for raft optic flow used for training pipeline
OPTIC_FLOW_SCALE_FACTOR = 2  # The factor by which is the resolution of the optic flow is scaled.
OPTIC_FLOW_H_PAD = 2  # Padding (in pixels) used along each dimension of the optic flow. During parsing of optic flow, the padding trimmed.
OPTIC_FLOW_W_PAD = 0


# Data item Info Field Template:
ROAD_IMAGE_TEMPLATE = "road_img_{:d}"
SHOULD_TRAIN_INPUT_GAZE_TEMPLATE = "should_train_input_gaze_{:d}"
RESIZED_INPUT_GAZE_TEMPLATE = "resized_input_gaze_{:d}"
NORMALIZED_INPUT_GAZE_TEMPLATE = "normalized_input_gaze_{:d}"
GROUND_TRUTH_GAZE_TEMPLATE = "ground_truth_gaze_{:d}"
SEGMENTATION_MASK_TEMPLATE = "segmentation_mask_img_{:d}"
OPTIC_FLOW_IMAGE_TEMPLATE = "optic_flow_img_{:d}"

# Auxiliary Info Field Template:
AUXILIARY_INFO_FULL_SIZE_GAZE_TEMPLATE = "full_size_gaze_{:d}"

# Gaze Dataset specific field names.
ROAD_IMAGE_0 = ROAD_IMAGE_TEMPLATE.format(0)
SHOULD_TRAIN_INPUT_GAZE_0 = SHOULD_TRAIN_INPUT_GAZE_TEMPLATE.format(0)
RESIZED_INPUT_GAZE_0 = RESIZED_INPUT_GAZE_TEMPLATE.format(0)
NORMALIZED_INPUT_GAZE_0 = NORMALIZED_INPUT_GAZE_TEMPLATE.format(0)
GROUND_TRUTH_GAZE_0 = GROUND_TRUTH_GAZE_TEMPLATE.format(0)
SEGMENTATION_MASK_0 = SEGMENTATION_MASK_TEMPLATE.format(0)
OPTIC_FLOW_IMAGE_0 = OPTIC_FLOW_IMAGE_TEMPLATE.format(0)

# Gaze Dataset specific field names.
AUXILIARY_INFO_VIDEO_ID = "video_id"
AUXILIARY_INFO_SUBJECT_ID = "subject_id"
AUXILIARY_INFO_FULL_SIZE_GAZE_0 = AUXILIARY_INFO_FULL_SIZE_GAZE_TEMPLATE.format(0)

# Enum for determining whether side channel gaze be used during inference.
class InferenceMode(Enum):
    WITH_GAZE = 0
    WITHOUT_GAZE = 1
    BOTH = 2