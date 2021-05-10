import uuid
import glob
import os
import json
import numpy as np
import collections

from chm.configs.args_file import parse_arguments
from chm.utils.experiment_result_keys import *

"""
Script to extract error stats from the awareness estimation experiments. If the file to be parsed is named 
experiment_type_awareness_estimates_EXTRA_INFO.json, the script is to be used as follows

Usage:
python parse_awareness_estimation_data.py --results_json_prefix experiment_type_awareness_estimates_ (--display_balanced_results)
"""


def arg_setter(parser):
    parser.add_argument(
        "--results_json_prefix",
        type=str,
        default="",
        help="Prefix for data json files from awareness estimation experiments",
    )
    parser.add_argument(
        "--display_balanced_results",
        action="store_true",
        default=False,
        help="Flag for displaying target class balanced results",
    )


if __name__ == "__main__":
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash, [arg_setter])

    display_balanced_results = args.display_balanced_results
    # extract all the data files with the specified prefix and sort them
    filenames = glob.glob(os.path.expanduser(args.results_json_prefix) + "*.json")
    filenames = sorted(filenames)

    for filename in filenames:
        print("        ")
        print(filename)
        with open(filename, "r") as fp:
            jsn = json.load(fp)

        results_sorted_according_to_labels = collections.defaultdict(list)
        # gather all possible targets for target class balanced results
        targets = jsn[AWARENESS_TARGET_KEY]
        target_histogram = collections.Counter(targets)
        sample_weights = [1.0 / target_histogram[t] for t in targets]

        # extract errors and estimates from the jsons
        sq_error_chm = jsn[AWARENESS_ERROR_CHM_KEY]
        sq_error_of_spatiotemporal_gaussian = jsn[AWARENESS_ERROR_OF_SPATIOTEMPORAL_GAUSSIAN_KEY]

        abs_error_chm = jsn[AWARENESS_ABS_ERROR_CHM_KEY]
        abs_error_of_spatiotemporal_gaussian = jsn[AWARENESS_ABS_ERROR_OF_SPATIOTEMPORAL_GAUSSIAN_KEY]

        awareness_estimate_chm = jsn[AWARENESS_ESTIMATE_CHM_KEY]
        awareness_estimate_of_spatiotemporal_gaussian = jsn[AWARENESS_ESTIMATE_OF_SPATIOTEMPORAL_GAUSSIAN_KEY]

        # sanity check to make sure all metrics are logged properly
        assert (
            len(targets)
            == len(sq_error_chm)
            == len(sq_error_of_spatiotemporal_gaussian)
            == len(abs_error_chm)
            == len(abs_error_of_spatiotemporal_gaussian)
            == len(awareness_estimate_chm)
            == len(awareness_estimate_of_spatiotemporal_gaussian)
        )

        if display_balanced_results:
            # weighted abs and eq errors
            abs_error_chm = np.array(abs_error_chm) * np.array(sample_weights)
            abs_error_of_spatiotemporal_gaussian = np.array(abs_error_of_spatiotemporal_gaussian) * np.array(
                sample_weights
            )
            sq_error_chm = np.array(sq_error_chm) * np.array(sample_weights)
            sq_error_of_spatiotemporal_gaussian = np.array(sq_error_of_spatiotemporal_gaussian) * np.array(
                sample_weights
            )

            # absolute weighted error mean and std
            std_abs_chm = np.std(abs_error_chm)
            std_abs_of_spatiotemporal_gaussian = np.std(abs_error_of_spatiotemporal_gaussian)
            mean_abs_chm = np.sum(abs_error_chm) / np.sum(sample_weights)
            mean_abs_of_spatiotemporal_gaussian = np.sum(abs_error_of_spatiotemporal_gaussian) / np.sum(sample_weights)

            # squared error mean and std
            std_sq_chm = np.std(sq_error_chm)
            std_sq_of_spatiotemporal_gaussian = np.std(sq_error_of_spatiotemporal_gaussian)
            mean_sq_chm = np.sum(sq_error_chm) / np.sum(sample_weights)
            mean_sq_of_spatiotemporal_gaussian = np.sum(sq_error_of_spatiotemporal_gaussian) / np.sum(sample_weights)
        else:
            # absolute error mean and std
            std_abs_chm = np.std(abs_error_chm)
            std_abs_of_spatiotemporal_gaussian = np.std(abs_error_of_spatiotemporal_gaussian)
            mean_abs_chm = np.average(abs_error_chm)
            mean_abs_of_spatiotemporal_gaussian = np.average(abs_error_of_spatiotemporal_gaussian)
            # squared error mean and std
            std_sq_chm = np.std(sq_error_chm)
            std_sq_of_spatiotemporal_gaussian = np.std(sq_error_of_spatiotemporal_gaussian)
            mean_sq_chm = np.average(sq_error_chm)
            mean_sq_of_spatiotemporal_gaussian = np.average(sq_error_of_spatiotemporal_gaussian)

        # print the means and std deviations for chm and baseline estimate to screen
        print(
            "ABS ERROR CHM: {} +/- {}. SpatioTemporal OF Gaussian: {} +/- {}".format(
                mean_abs_chm,
                std_abs_chm,
                mean_abs_of_spatiotemporal_gaussian,
                std_abs_of_spatiotemporal_gaussian,
            )
        )

        print(
            "SQUARED ERROR CHM: {} +/- {}. SpatioTemporal OF Gaussian: {} +/- {}".format(
                mean_sq_chm,
                std_sq_chm,
                mean_sq_of_spatiotemporal_gaussian,
                std_sq_of_spatiotemporal_gaussian,
            )
        )

        # mean and std deviation of the awareness estimates
        std_awareness_estimate_chm = np.std(awareness_estimate_chm)
        std_awareness_estimate_of_spatiotemporal_gaussian = np.std(awareness_estimate_of_spatiotemporal_gaussian)
        mean_awareness_estimate_chm = np.average(awareness_estimate_chm)
        mean_awareness_estimate_of_spatiotemporal_gaussian = np.average(awareness_estimate_of_spatiotemporal_gaussian)

        # print the awareness estimate from CHM and baseline estimate to screen
        print(
            "AWARENESS ESTIMATE CHM: {} +/- {}. SpatioTemporal OF Gaussian: {} +/- {}".format(
                mean_awareness_estimate_chm,
                std_awareness_estimate_chm,
                mean_awareness_estimate_of_spatiotemporal_gaussian,
                std_awareness_estimate_of_spatiotemporal_gaussian,
            )
        )
