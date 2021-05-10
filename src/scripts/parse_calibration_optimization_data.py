import uuid
import json
import os
import numpy as np

from chm.configs.args_file import parse_arguments
from chm.utils.experiment_result_keys import *

"""
Script to extract error stats from calibration optimization experiments. 
The results of the experiments are expected to stored in files with the following filename convention

experiment_type_gaze_calibration_miscalibration_noise_level_NOISELEVEL_optimization_run_num_OPTIMIZATIONNUM_FILENAMEAPPEND.json,
where NOISELEVEL is in the miscalibration_noise_levels arg in experiment_chm_calibration_optimization
OPTIMIZATIONNUM goes from 0 to num_optimization_runs-1, 
and FILENAMEAPPEND is the 'filename_append' arg in the experiment. 

Usage:
python parse_denoising_data.py --folder_containing_results FOLDER_CONTAINING_JSONS --num_optimization_runs (same val as used in the experiment)
--miscalibration_noise_levels (same val as used in the experiment) --filename_append (same val as used in the experiment)
"""


def arg_setter(parser):
    parser.add_argument(
        "--folder_containing_results",
        type=str,
        default="",
        help="Path to folder containing calibration optimization results",
    )

    parser.add_argument(
        "--num_optimization_runs",
        type=int,
        default=15,
        help="Number of optimizations runs performed for a given noise level for the calibration experiment",
    )

    parser.add_argument(
        "--miscalibration_noise_levels",
        action="store",
        nargs="*",
        type=float,
        default=[0.1, 0.2, 0.3, 0.5],
        help="Noise levels used in the calibration experiments",
    )

    parser.add_argument(
        "--filename_append", type=str, default="", help="Additional descriptive string for filename string components"
    )


if __name__ == "__main__":
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash, [arg_setter])
    miscalibration_noise_levels = args.miscalibration_noise_levels
    num_optimization_runs = args.num_optimization_runs
    filename_append = args.filename_append
    file_prefix = "experiment_type_gaze_calibration_miscalibration_noise_level_"
    for noise_level in miscalibration_noise_levels:
        starting_mse_error_list = []
        end_mse_error_list = []
        for i in range(num_optimization_runs):
            jsnfile_name = (
                file_prefix + str(noise_level) + "_optimization_run_num_" + str(i) + filename_append + ".json"
            )
            print("Loading : ", jsnfile_name)
            jsn_full_path = os.path.join(args.folder_containing_results, jsnfile_name)
            with open(jsn_full_path, "r") as fp:
                res = json.load(fp)

            starting_mse_error_list.append(res["errors"][0]["error"])
            # assumes that the calibration optimization successfully converged
            end_mse_error_list.append(res["errors"][-1]["error"])

        print(
            "Mean and std for starting error for noise level {} is {} and {}".format(
                noise_level, np.mean(starting_mse_error_list), np.std(starting_mse_error_list)
            )
        )
        print(
            "Mean and std for ending error for noise level {} is {} and {}".format(
                noise_level, np.mean(end_mse_error_list), np.std(end_mse_error_list)
            )
        )
        print("              ")
