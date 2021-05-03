# Copyright 2021 Toyota Research Institute.  All rights reserved.

# This file details the field names and some notations used in datasets' getitems for CHM training
from enum import Enum
import numpy as np

TRACKER_IMAGE = "tracker_img"

ROAD_IMAGE_TEMPLATE = "road_img_{:d}"
SHOULD_TRAIN_INPUT_GAZE_TEMPLATE = "should_train_input_gaze_{:d}"
SEGMENTATION_MASK_TEMPLATE = "segmentation_mask_img_{:d}"
OPTIC_FLOW_IMAGE_TEMPLATE = "optic_flow_img_{:d}"
TRANSFORM_TEMPLATE = "transform_{:d}"

# (d_input+1 x S_max) where d_input is the input dim (2 for 2D gaze coordinate as in Tobii, may be different for DMS systems.) followed by a valid flag, and S_max is the maximum number of gaze inputs per sampled frame.
TRACKER_INPUT_GAZE = "tracker_input_gaze"
TRACKER_INPUT_GAZE_VALIDITY = "tracker_input_gaze_validity"
GAZE_TYPE_FLAG = "gaze_type_flag"
RESIZED_INPUT_GAZE_TEMPLATE = "resized_input_gaze_{:d}"
NORMALIZED_INPUT_GAZE_TEMPLATE = "normalized_input_gaze_{:d}"
TRANSFORMED_INPUT_GAZE_TEMPLATE = "transformed_input_gaze_{:d}"  # (3 x S_max) 3 is x,y,valid_flag
NORMALIZED_TRANSFORMED_INPUT_GAZE_TEMPLATE = "normalized_transformed_input_gaze_{:d}"
GROUND_TRUTH_GAZE_TEMPLATE = "ground_truth_gaze_{:d}"  # (3 x S_max)


FIXATION_TYPE = "fixation_type"  # np.array(one_hot[ dim_fixation_types])
TASK_TYPE = "task_type"
MAX_NUM_VIDEO_FRAMES = 7501
ALL_DREYEVE_VIDEO_IDS = [6, 7, 10, 11, 26, 35, 53, 60]
OPTIC_FLOW_SCALE_FACTOR = 2
OPTIC_FLOW_H_PAD = 2
OPTIC_FLOW_W_PAD = 0


class InferenceMode(Enum):
    WITH_GAZE = 0
    WITHOUT_GAZE = 1
    BOTH = 2


# Auxiliary Info Structure:
AUXILIARY_INFO_FULL_SIZE_GAZE_TEMPLATE = "full_size_gaze_{:d}"
AUXILIARY_INFO_SEGMENTATION_SEGMENTS_TEMPLATE = "segmentation_segments_{:d}"
AUXILIARY_INFO_IM_MATCHES = "im_matches"
AUXILIARY_INFO_SUBJECT_ID = "subject_id"
AUXILIARY_INFO_SEGMENTATION_FRAME_TEMPLATE = "segmentation_frame_{:d}"
AUXILIARY_INFO_TRACKER_FRAME_TEMPLATE = "tracker_frame_{:d}"
AUXILIARY_INFO_ROAD_FRAME_TEMPLATE = "road_frame_{:d}"
AUXILIARY_INFO_RAW_TRACKER_INPUT_GAZE = "raw_tracker_input_gaze"
AUXILIARY_INFO_RAW_TRANSFORMED_INPUT_GAZE_TEMPLATE = "raw_transformed_input_gaze_{:d}"
AUXILIARY_INFO_VIDEO_TRANSFORMED_INPUT_GAZE_TEMPLATE = "video_transformed_input_gaze_{:d}"

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

# Gaze Dataset specific field names.
ROAD_IMAGE_0 = ROAD_IMAGE_TEMPLATE.format(0)
SHOULD_TRAIN_INPUT_GAZE_0 = SHOULD_TRAIN_INPUT_GAZE_TEMPLATE.format(0)
TRANSFORMED_INPUT_GAZE_0 = TRANSFORMED_INPUT_GAZE_TEMPLATE.format(0)
RESIZED_INPUT_GAZE_0 = RESIZED_INPUT_GAZE_TEMPLATE.format(0)
NORMALIZED_TRANSFORMED_INPUT_GAZE_0 = NORMALIZED_TRANSFORMED_INPUT_GAZE_TEMPLATE.format(0)
NORMALIZED_INPUT_GAZE_0 = NORMALIZED_INPUT_GAZE_TEMPLATE.format(0)
GROUND_TRUTH_GAZE_0 = GROUND_TRUTH_GAZE_TEMPLATE.format(0)
SEGMENTATION_MASK_0 = SEGMENTATION_MASK_TEMPLATE.format(0)
OPTIC_FLOW_IMAGE_0 = OPTIC_FLOW_IMAGE_TEMPLATE.format(0)

AUXILIARY_INFO_VIDEO_ID = "video_id"
AUXILIARY_INFO_FULL_SIZE_GAZE_0 = AUXILIARY_INFO_FULL_SIZE_GAZE_TEMPLATE.format(0)
AUXILIARY_INFO_SEGMENTATION_SEGMENTS_0 = AUXILIARY_INFO_SEGMENTATION_SEGMENTS_TEMPLATE.format(0)
AUXILIARY_INFO_SEGMENTATION_FRAME_0 = AUXILIARY_INFO_SEGMENTATION_FRAME_TEMPLATE.format(0)
AUXILIARY_INFO_TRACKER_FRAME_0 = AUXILIARY_INFO_TRACKER_FRAME_TEMPLATE.format(0)
AUXILIARY_INFO_ROAD_FRAME_0 = AUXILIARY_INFO_ROAD_FRAME_TEMPLATE.format(0)
AUXILIARY_INFO_RAW_TRANSFORMED_INPUT_GAZE_0 = AUXILIARY_INFO_RAW_TRANSFORMED_INPUT_GAZE_TEMPLATE.format(0)
AUXILIARY_INFO_VIDEO_TRANSFORMED_INPUT_GAZE_0 = AUXILIARY_INFO_VIDEO_TRANSFORMED_INPUT_GAZE_TEMPLATE.format(0)
