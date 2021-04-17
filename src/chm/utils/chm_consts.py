# Copyright 2021 Toyota Research Institute.  All rights reserved.

# This file details the field names and some notations used in datasets' getitems for CHM training
from enum import Enum
import numpy as np

TRACKER_IMAGE = "tracker_img"

ROAD_IMAGE_TEMPLATE = "road_img_{:d}"
SHOULD_TRAIN_INPUT_GAZE_TEMPLATE = "should_train_input_gaze_{:d}"
PANOPTIC_MASK_TEMPLATE = "panoptic_mask_img_{:d}"
OPTIC_FLOW_IMAGE_TEMPLATE = "optic_flow_img_{:d}"
TRANSFORM_TEMPLATE = "transform_{:d}"

# (d_input+1 x S_max) where d_input is the input dim (2 for 2D gaze coordinate as in Tobii, may be different for DMS systems.) followed by a valid flag, and S_max is the maximum number of gaze inputs per sampled frame.
TRACKER_INPUT_GAZE = "tracker_input_gaze"
TRACKER_INPUT_GAZE_VALIDITY = "tracker_input_gaze_validity"
GAZE_TYPE_FLAG = "gaze_type_flag"
TRANSFORMED_INPUT_GAZE_TEMPLATE = "transformed_input_gaze_{:d}"  # (3 x S_max) 3 is x,y,valid_flag
NORMALIZED_TRANSFORMED_INPUT_GAZE_TEMPLATE = "normalized_transformed_input_gaze_{:d}"
TRANSFORM_VALIDITY_TEMPLATE = "transform_validity_{:d}"
GROUND_TRUTH_GAZE_TEMPLATE = "ground_truth_gaze_{:d}"  # (3 x S_max)
# confidence of the transform estimate, (1 x S_max)
TRANSFORM_CONFIDENCE_TEMPLATE = "transform_confidence_{:d}"

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


class InputValidityValue(Enum):
    VALID = 0
    INVALID = 1


class InputGazeType(Enum):
    FIXATION = 0
    BLINK = 1
    SACCADE = 2
    INVALID = 3
    UNKNOWN = 4


class TransformedInputValidityValue(Enum):
    VALID = 0
    POINT_OUTSIDE_IMAGE = 1
    HOMOGRAPH_ERROR = 2
    INVALID_GAZE_INPUT = 3
    LOW_CONFIDENCE = 4
    IGNORE = -1  # is this well defined?


class TaskType(Enum):
    CONTROL = 0
    ROADONLY = 1
    FLIPPED = 2
    READINGTEXT = 3
    BLURRED = 4


# Auxiliary Info Structure:
AUXILIARY_INFO_PANOPTIC_SEGMENTS_TEMPLATE = "panoptic_segments_{:d}"
AUXILIARY_INFO_IM_MATCHES = "im_matches"
AUXILIARY_INFO_SUBJECT_ID = "subject_id"
AUXILIARY_INFO_PANOPTIC_FRAME_TEMPLATE = "panoptic_frame_{:d}"
AUXILIARY_INFO_TRACKER_FRAME_TEMPLATE = "tracker_frame_{:d}"
AUXILIARY_INFO_ROAD_FRAME_TEMPLATE = "road_frame_{:d}"
AUXILIARY_INFO_RAW_TRACKER_INPUT_GAZE = "raw_tracker_input_gaze"
AUXILIARY_INFO_RAW_TRANSFORMED_INPUT_GAZE_TEMPLATE = "raw_transformed_input_gaze_{:d}"
AUXILIARY_INFO_VIDEO_TRANSFORMED_INPUT_GAZE_TEMPLATE = "video_transformed_input_gaze_{:d}"

# dreyeve gaze type enum lookup
DREYEVE_GAZE_TYPE_ENUMS = {}
DREYEVE_GAZE_TYPE_ENUMS["Fixation"] = np.int32(InputGazeType.FIXATION.value)
DREYEVE_GAZE_TYPE_ENUMS["Saccade"] = np.int32(InputGazeType.SACCADE.value)
DREYEVE_GAZE_TYPE_ENUMS["Blink"] = np.int32(InputGazeType.BLINK.value)
DREYEVE_GAZE_TYPE_ENUMS["-"] = np.int32(InputGazeType.INVALID.value)
DREYEVE_GAZE_TYPE_ENUMS["NA"] = np.int32(InputGazeType.UNKNOWN.value)


EYELINK_DREYEVE_TASK_TYPE_ENUMS = {}
EYELINK_DREYEVE_TASK_TYPE_ENUMS["control"] = np.int32(TaskType.CONTROL.value)
EYELINK_DREYEVE_TASK_TYPE_ENUMS["roadonly"] = np.int32(TaskType.ROADONLY.value)
EYELINK_DREYEVE_TASK_TYPE_ENUMS["flipped"] = np.int32(TaskType.FLIPPED.value)
EYELINK_DREYEVE_TASK_TYPE_ENUMS["readingtext"] = np.int32(TaskType.READINGTEXT.value)
EYELINK_DREYEVE_TASK_TYPE_ENUMS["blurred"] = np.int32(TaskType.BLURRED.value)

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

# Gaze Dataset specific field names.
ROAD_IMAGE_0 = ROAD_IMAGE_TEMPLATE.format(0)
SHOULD_TRAIN_INPUT_GAZE_0 = SHOULD_TRAIN_INPUT_GAZE_TEMPLATE.format(0)
TRANSFORM_0 = TRANSFORM_TEMPLATE.format(0)
TRANSFORM_VALIDITY_0 = TRANSFORM_VALIDITY_TEMPLATE.format(0)
TRANSFORMED_INPUT_GAZE_0 = TRANSFORMED_INPUT_GAZE_TEMPLATE.format(0)
NORMALIZED_TRANSFORMED_INPUT_GAZE_0 = NORMALIZED_TRANSFORMED_INPUT_GAZE_TEMPLATE.format(0)
GROUND_TRUTH_GAZE_0 = GROUND_TRUTH_GAZE_TEMPLATE.format(0)
TRANSFORM_CONFIDENCE_0 = TRANSFORM_CONFIDENCE_TEMPLATE.format(0)
PANOPTIC_MASK_0 = PANOPTIC_MASK_TEMPLATE.format(0)
OPTIC_FLOW_IMAGE_0 = OPTIC_FLOW_IMAGE_TEMPLATE.format(0)
AUXILIARY_INFO_PANOPTIC_SEGMENTS_0 = AUXILIARY_INFO_PANOPTIC_SEGMENTS_TEMPLATE.format(0)
AUXILIARY_INFO_PANOPTIC_FRAME_0 = AUXILIARY_INFO_PANOPTIC_FRAME_TEMPLATE.format(0)
AUXILIARY_INFO_TRACKER_FRAME_0 = AUXILIARY_INFO_TRACKER_FRAME_TEMPLATE.format(0)
AUXILIARY_INFO_ROAD_FRAME_0 = AUXILIARY_INFO_ROAD_FRAME_TEMPLATE.format(0)
AUXILIARY_INFO_RAW_TRANSFORMED_INPUT_GAZE_0 = AUXILIARY_INFO_RAW_TRANSFORMED_INPUT_GAZE_TEMPLATE.format(0)
AUXILIARY_INFO_VIDEO_TRANSFORMED_INPUT_GAZE_0 = AUXILIARY_INFO_VIDEO_TRANSFORMED_INPUT_GAZE_TEMPLATE.format(0)
