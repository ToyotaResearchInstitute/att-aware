# Copyright 2020 Toyota Research Institute.  All rights reserved.
import uuid
import copy
import functools
import numpy as np

from collections import OrderedDict

from chm.configs.args_file import parse_arguments
from chm.experiments.chm_experiments import ChmExperiment

from chm.utils.chm_consts import InferenceMode
from chm.utils.experiment_result_keys import *
from chm.utils.inference_utils import seek_mode

from chm.model.gaze_transform import GazeTransform
from chm.model.gaze_corruption import GazeCorruption


class CHMDenoisingExperiment(ChmExperiment):
    """
    Denoising Experiment

    Class to perform gaze denoising using predicted output (with and without gaze),as well as meanshift to nearest object using the segmentation masks
    """

    def __init__(self, args, session_hash):
        """
        Parameters:
        -----------
        args: argparse.Namespace
            Contains all args specified in the args_file and any additional arg_setter (specified in the derived classes)

        session_hash: str
            Unique string indicating the sessions id.
        """
        super().__init__(args, session_hash, training_experiment=False)
        self.args = args

        # init params for GazeTransform to be used by GazeCorruption
        self.gaze_linear_mtx = np.identity(2, dtype=np.float32)  # 2 by 2  identity matrix
        for i in range(2):
            # add some low level noise to the diagonal elements.
            self.gaze_linear_mtx[i, i] += np.random.normal(0, 0.01)

        self.gaze_translation_vec = np.zeros([2], dtype=np.float32)  # bias is now zero.
        self.gaze_corruption_transform = GazeTransform(
            linear_transform=self.gaze_linear_mtx, translation=self.gaze_translation_vec, pad_gaze_vector=False
        )

        # init gaze corruption modules
        self.gaze_corruption = GazeCorruption(
            bias_std=self.args.gaze_bias_std,
            noise_std=self.args.gaze_noise_std,
            transforms=[self.gaze_corruption_transform],
            x_weight_factor=self.args.weight_factor[0],
            y_weight_factor=self.args.weight_factor[1],
            is_spatially_varying=self.args.is_spatially_varying,
        )

        # list of dict keys used for logging results
        self.result_keys = [
            GAZE_ERROR_CHM_KEY_WITH_GAZE,
            GAZE_ERROR_CHM_KEY_WITHOUT_GAZE,
            GAZE_ERROR_NOISY_KEY,
            GAZE_ERROR_OBJ_KEY,
            GAZE_ERROR_MEANSHIFT_SEQUENCE_OBJECTBASED_KEY,
            GAZE_ERROR_MEANSHIFT_SEQUENCE_WITH_GAZE_KEY,
            GAZE_ERROR_MEANSHIFT_SEQUENCE_WITHOUT_GAZE_KEY,
            MEANSHIFT_SIGMA,
            GAZE_INFO,
        ]

    def initialize_functors(self):

        # initialize input process dict
        def chm_denoising_input_functor(batch_input, aux_info_list, input_process_params):
            """
            Functor to corrupt side-channel gaze according to different noise levels
            """
            should_use_batch = True
            gaze_key = "normalized_input_gaze"
            gaze_key_before_corruption = gaze_key + "_before_corruption"
            # save the gaze point before corruption.
            batch_input[gaze_key_before_corruption] = batch_input[gaze_key]
            gaze = batch_input[gaze_key]
            gaze = self.gaze_corruption.corrupt_gaze(gaze)  # corrupt the gaze
            batch_input[gaze_key] = gaze  # replace 'normalized_input_gaze' with the noisy version.
            return batch_input, aux_info_list, should_use_batch

        self.model_wrapper.input_process_dict = {}  # init empty input process dict
        self.model_wrapper.input_process_dict["functor"] = functools.partial(chm_denoising_input_functor)
        self.model_wrapper.input_process_dict["params"] = {}
        self.model_wrapper.input_process_dict["inference_mode"] = InferenceMode.BOTH
        self.model_wrapper.input_process_dict["max_batch_num"] = self.args.max_inference_num_batches

        # initialize output process dict
        def chm_denoising_output_functor(
            inference_output_dict, output_process_params, global_step, model=None, experiment_results_aggregator=None
        ):
            gaze_map_with_gaze = inference_output_dict["predicted_gaze_with_gaze"]["gaze_density_map"]
            gaze_map_without_gaze = inference_output_dict["predicted_gaze_without_gaze"]["gaze_density_map"]
            gaze_batch_input = inference_output_dict["gaze_batch_input"]

            batch_mask_image = gaze_batch_input["segmentation_mask_image"]  # (B, T, 3, H, W)
            # corrupted gaze coordinate
            gaze_coordinates = gaze_batch_input["normalized_input_gaze"]
            # ground truth gaze before corruption introduced in the input process functor
            groundtruth_gaze_coordinates = gaze_batch_input["normalized_input_gaze_before_corruption"]
            gaze_aux_info_list = inference_output_dict["gaze_aux_info_list"]

            # init experiment results aggregator
            if experiment_results_aggregator is None:
                experiment_results_aggregator = OrderedDict()  # create a dict to aggregate results

            for RESULT_KEY in self.result_keys:
                if RESULT_KEY not in experiment_results_aggregator:
                    experiment_results_aggregator[RESULT_KEY] = []

            # init metric variables
            # metric accumulator for the entire batch
            coord_err_L2_noisy = 0.0  # error between the ground truth and noisy gaze
            # error between ground truth and after meanshift on gaze map with gaze
            coord_err_L2_chm_meanshift_with_gaze = 0.0
            # error between ground truth and after meanshift on gaze map without gaze
            coord_err_L2_chm_meanshift_without_gaze = 0.0
            # error between ground truth and after meanshift on object map
            coord_err_L2_obj_meanshift = 0.0

            gaze_info_list = []
            meanshift_sigma_list = []
            seek_stats_without_gaze_list = []
            seek_stats_with_gaze_list = []
            seek_stats_object_based_list = []

            bt_counter = 1
            for b in range(gaze_coordinates.shape[0]):  # batch size
                gaze_info_list_b = []
                meanshift_sigma_list_b = []
                seek_stats_without_gaze_list_b = []
                seek_stats_with_gaze_list_b = []
                seek_stats_object_based_list_b = []
                t_counter = 0
                for t in range(gaze_coordinates.shape[1]):  # sequence length
                    gaze_info_dict = {}
                    gaze = gaze_coordinates[b, t, 0, :]
                    if int(gaze_batch_input["should_train_input_gaze"][b, t, 0, 0].cpu().float()) == 0:  # invalid gaze
                        continue
                    # get the corresponding ground truth gaze (which the network has not seen)
                    groundtruth_gaze = groundtruth_gaze_coordinates[b, t, 0, :]
                    # scale the noisy gaze x properly to the size of the gaze map width (240)
                    x = gaze[0].cpu().item() * gaze_map_with_gaze.shape[-1]
                    # scale the noisy gaze y value to the size of the gaze map height( 135)
                    y = gaze[1].cpu().item() * gaze_map_with_gaze.shape[-2]
                    # for invalid gaze points
                    if x < 0 or y < 0 or x >= gaze_map_with_gaze.shape[-1] or y >= gaze_map_with_gaze.shape[-2]:
                        continue
                    # scale to gaze map width
                    x_gt = groundtruth_gaze[0].cpu().item() * gaze_map_with_gaze.shape[-1]
                    # scale to gaze map height
                    y_gt = groundtruth_gaze[1].cpu().item() * gaze_map_with_gaze.shape[-2]
                    if (
                        x_gt < 0
                        or y_gt < 0
                        or x_gt >= gaze_map_with_gaze.shape[-1]
                        or y_gt >= gaze_map_with_gaze.shape[-2]
                    ):
                        continue
                    bt_counter += 1
                    t_counter += 1
                    img_with_gaze = gaze_map_with_gaze[b, t, 0, :, :]  # (H, W)
                    img_without_gaze = gaze_map_without_gaze[b, t, 0, :, :]  # (H, W)
                    # (H, W), Square because only concerned about pixel intensities
                    mask_image = (batch_mask_image[b, t, :, :, :] ** 2).sum(axis=0)
                    # sigma for the gaussian filter for meanshift algorithm
                    sigma = max(
                        5,
                        int(
                            self.args.meanshift_mult_factor
                            * self.args.gaze_noise_std
                            * np.sqrt(gaze_map_with_gaze.shape[-1] ** 2 + gaze_map_with_gaze.shape[-2] ** 2)
                        ),
                    )
                    noisy_coord = (x, y)
                    groundtruth_coord = (x_gt, y_gt)

                    # perform meanshift
                    # starting from the noisy coordinate, seek the nearest mode on three different images
                    # (heatmap with gaze, heatmap without gaze, mask image)
                    coord_after_seek_with_gaze, seek_stats_with_gaze = seek_mode(
                        img_with_gaze.cpu().numpy(), x, y, sigma=sigma
                    )
                    coord_after_seek_without_gaze, seek_stats_without_gaze = seek_mode(
                        img_without_gaze.cpu().numpy(), x, y, sigma=sigma
                    )
                    coord_after_seek_obj, seek_stats_obj = seek_mode(mask_image.cpu().numpy(), x, y, sigma=sigma)

                    seek_stats_obj["ground_truth"] = groundtruth_coord
                    seek_stats_without_gaze["ground_truth"] = groundtruth_coord
                    seek_stats_with_gaze["ground_truth"] = groundtruth_coord

                    gaze_info_dict["groundtruth_coord"] = groundtruth_coord
                    gaze_info_dict["noisy_coord"] = noisy_coord
                    gaze_info_dict["groundtruth_gaze"] = groundtruth_gaze.detach().cpu().numpy().tolist()
                    gaze_info_dict["noisy_gaze"] = gaze.detach().cpu().numpy().tolist()

                    meanshift_sigma_list_b.append(sigma)
                    gaze_info_list_b.append(gaze_info_dict)
                    seek_stats_without_gaze_list_b.append(seek_stats_without_gaze)
                    seek_stats_with_gaze_list_b.append(seek_stats_with_gaze)
                    seek_stats_object_based_list_b.append(seek_stats_obj)

                    # compute MSE error for each of the cases
                    noisy_error = np.linalg.norm(np.array(groundtruth_coord) - np.array(noisy_coord)) ** 2
                    with_gaze_error = (
                        np.linalg.norm(np.array(groundtruth_coord) - np.array(coord_after_seek_with_gaze)) ** 2
                    )
                    without_gaze_error = (
                        np.linalg.norm(np.array(groundtruth_coord) - np.array(coord_after_seek_without_gaze)) ** 2
                    )
                    obj_error = np.linalg.norm(np.array(groundtruth_coord) - np.array(coord_after_seek_obj)) ** 2

                    # log the errors in stats dict object
                    seek_stats_obj["ms_error"] = obj_error
                    seek_stats_without_gaze["ms_error"] = without_gaze_error
                    seek_stats_with_gaze["ms_error"] = with_gaze_error

                    # accumulate error
                    coord_err_L2_noisy += noisy_error
                    coord_err_L2_chm_meanshift_with_gaze += with_gaze_error
                    coord_err_L2_chm_meanshift_without_gaze += without_gaze_error
                    coord_err_L2_obj_meanshift += obj_error

                if t_counter == 0:
                    # if all t gaze points were invalid, continue to next batch item.
                    continue

                gaze_info_list.append(gaze_info_list_b)
                meanshift_sigma_list.append(meanshift_sigma_list_b)
                seek_stats_with_gaze_list.append(seek_stats_with_gaze_list_b)
                seek_stats_without_gaze_list.append(seek_stats_without_gaze_list_b)
                seek_stats_object_based_list.append(seek_stats_object_based_list_b)

            # add results into the results dict
            experiment_results_aggregator[MEANSHIFT_SIGMA].append(meanshift_sigma_list)
            experiment_results_aggregator[GAZE_INFO].append(gaze_info_list)
            experiment_results_aggregator[GAZE_ERROR_MEANSHIFT_SEQUENCE_WITHOUT_GAZE_KEY].append(
                seek_stats_without_gaze_list
            )
            experiment_results_aggregator[GAZE_ERROR_MEANSHIFT_SEQUENCE_WITH_GAZE_KEY].append(seek_stats_with_gaze_list)
            experiment_results_aggregator[GAZE_ERROR_MEANSHIFT_SEQUENCE_OBJECTBASED_KEY].append(
                seek_stats_object_based_list
            )

            experiment_results_aggregator[GAZE_ERROR_NOISY_KEY].append(coord_err_L2_noisy / bt_counter)
            experiment_results_aggregator[GAZE_ERROR_CHM_KEY_WITH_GAZE].append(
                coord_err_L2_chm_meanshift_with_gaze / bt_counter
            )
            experiment_results_aggregator[GAZE_ERROR_CHM_KEY_WITHOUT_GAZE].append(
                coord_err_L2_chm_meanshift_without_gaze / bt_counter
            )
            experiment_results_aggregator[GAZE_ERROR_OBJ_KEY].append(coord_err_L2_obj_meanshift / bt_counter)

            # compute mean, std for each of the MSE error.
            for key in [
                GAZE_ERROR_NOISY_KEY,
                GAZE_ERROR_CHM_KEY_WITH_GAZE,
                GAZE_ERROR_CHM_KEY_WITHOUT_GAZE,
                GAZE_ERROR_OBJ_KEY,
            ]:
                experiment_results_aggregator["overall_" + key + "_mean"] = np.mean(experiment_results_aggregator[key])
                experiment_results_aggregator["overall_" + key + "_std"] = np.std(experiment_results_aggregator[key])
                experiment_results_aggregator["overall_" + key + "_sqrt_mean"] = np.sqrt(
                    np.mean(experiment_results_aggregator[key])
                )
                experiment_results_aggregator["overall_" + key + "_median"] = np.median(
                    experiment_results_aggregator[key]
                )
                experiment_results_aggregator["overall_size"] = len(experiment_results_aggregator[key])

            experiment_results_aggregator["corruption_bias_std"] = self.gaze_corruption.bias_std
            experiment_results_aggregator["corruption_noise_std"] = self.gaze_corruption.noise_std

            return experiment_results_aggregator

        self.model_wrapper.output_process_dict = {}
        self.model_wrapper.output_process_dict["functor"] = functools.partial(chm_denoising_output_functor)
        self.model_wrapper.output_process_dict["params"] = {}

    def perform_experiment(self):
        self._perform_experiment()


def arg_setter(parser):
    parser.add_argument(
        "--experiment_noise_levels",
        action="store",
        nargs="*",
        type=float,
        default=[1.0e-3, 1.0e-2, 1.0e-1],
        help="Noise levels for side channel gaze for denoising experiment",
    )

    parser.add_argument(
        "--weight_factor",
        action="store",
        nargs="*",
        type=float,
        default=[1.0, 1.3],
        help="Direction dependent weight factor for x and y dimensions for spatially varying noise",
    )

    parser.add_argument(
        "--meanshift_mult_factor",
        action="store",
        type=float,
        default=5.0,
        help="Multiplication factor for sigma (std dev) for the meanshift algorithm",
    )

    parser.add_argument(
        "--is_spatially_varying",
        action="store_true",
        default=False,
        help="Option for spatially varying noise for gaze corruption. Flag gets passed on to the gaze corruption module. ",
    )

    parser.add_argument(
        "--filename_append", type=str, default="", help="Additional descriptive string for filename string components"
    )


def main():
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash, [arg_setter])
    for noise_level in args.experiment_noise_levels:
        print("Noise level: {}".format(noise_level))
        args_copy = copy.deepcopy(args)
        args_copy.gaze_bias_std = 1e-8  # zero mean white noise
        args_copy.gaze_noise_std = noise_level
        args_copy.experiment_noise_levels = []

        # instantiate the experiment
        denoising_experiment = CHMDenoisingExperiment(args_copy, session_hash)
        denoising_experiment.initialize_functors()
        denoising_experiment.perform_experiment()

        # file name components
        filename_variables = OrderedDict()
        filename_variables["type"] = "gaze_denoising"
        filename_variables["spatially_varying"] = str(args_copy.is_spatially_varying)
        filename_variables["filename_append"] = args_copy.filename_append
        filename_variables["noise_level"] = noise_level

        # save json after experiment
        denoising_experiment.save_experiment(name_values=filename_variables)


if __name__ == "__main__":
    main()