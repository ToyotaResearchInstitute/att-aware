# Copyright 2020 Toyota Research Institute.  All rights reserved.
import uuid
import os
import glob
import json
import numpy as np

from maad.configs.args_file import parse_arguments
from maad.utils.experiment_result_keys import *

"""
Script to extract error stats from gaze denoising experiments. If the file to be parsed is named
experiment_type_gaze_denoising_EXTRA_INFO.json, the script is to be used as follows

Usage:
python parse_denoising_data.py --results_json_prefix experiment_type_gaze_denoising_
"""


def arg_setter(parser):
    parser.add_argument(
        "--results_json_prefix",
        type=str,
        default="",
        help="Prefix for data json files from awareness estimation experiments",
    )


if __name__ == "__main__":
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash, [arg_setter])
    # extract all the data files with the specified prefix and sort them
    filenames = glob.glob(os.path.expanduser(args.results_json_prefix) + "*.json")
    filenames = sorted(filenames)
    for filename in filenames:
        print("     ")
        print(filename)
        with open(filename, "r") as fp:
            jsn = json.load(fp)

        # grab errors computed during experiment.
        error_noisy = jsn[GAZE_ERROR_NOISY_KEY]
        error_objectbased = jsn[GAZE_ERROR_OBJ_KEY]
        error_maad = jsn[GAZE_ERROR_MAAD_KEY_WITH_GAZE]
        error_saliency = jsn[GAZE_ERROR_MAAD_KEY_WITHOUT_GAZE]

        # grab the mean sqrt errors.
        sqrt_error_noisy = jsn["overall_" + GAZE_ERROR_NOISY_KEY + "_sqrt_mean"]
        sqrt_error_objectbased = jsn["overall_" + GAZE_ERROR_OBJ_KEY + "_sqrt_mean"]
        sqrt_error_maad = jsn["overall_" + GAZE_ERROR_MAAD_KEY_WITH_GAZE + "_sqrt_mean"]
        sqrt_error_saliency = jsn["overall_" + GAZE_ERROR_MAAD_KEY_WITHOUT_GAZE + "_sqrt_mean"]

        # compute mean and std of errors
        mean_noisy = np.average(error_noisy)
        mean_object = np.average(error_objectbased)
        mean_maad = np.average(error_maad)
        mean_saliency = np.average(error_saliency)
        std_noisy = np.std(error_noisy)
        std_object = np.std(error_objectbased)
        std_maad = np.std(error_maad)
        std_saliency = np.std(error_saliency)
        # print the mean and std errors for different denoising approaches. "Noisy" refers to without denoising
        print(
            "MAAD: {} +/- {}. Saliency: {} +/- {}. Object-based: {} +/- {}. Noisy: {} +/- {}".format(
                mean_maad, std_maad, mean_saliency, std_saliency, mean_object, std_object, mean_noisy, std_noisy
            )
        )
        print(
            "SQRT MEAN MAAD: {}. Saliency: {}. Object-based: {}. Noisy: {}".format(
                sqrt_error_maad, sqrt_error_saliency, sqrt_error_objectbased, sqrt_error_noisy
            )
        )
