import uuid
import copy
import torch
import functools
import numpy as np

from collections import OrderedDict
from chm.configs.args_file import parse_arguments
from chm.experiments.chm_experiments import ChmExperiment

from chm.utils.chm_consts import InferenceMode
from chm.utils.experiment_result_keys import GAZE_INFO

from chm.model.gaze_transform import GazeTransform
from chm.model.gaze_corruption import GazeCorruption
from chm.model.gaze_transform import compute_inverted_affine_transform


class ChmCalibrationOptimizationExperiment(ChmExperiment):
    def __init__(self, args, session_hash):
        super().__init__(args, session_hash, training_experiment=True)
        self.args = args

        # corruption and correction variables
        self.gaze_corruption_transform_linear_matrix = None
        self.gaze_corruption_transform_translation_vec = None
        self.gaze_correction_transform_linear_mtx = None
        self.gaze_correction_translation_vec = None
        self.gaze_corruption_transform = None

        self.gaze_transform = None
        self.gaze_correction_transform = None

        # Miscalibrated transform that will be corrected
        self._generate_miscalibration_transform(self.args.miscalibration_noise)
        self.gaze_correction_transform_linear_mtx = np.identity(2, dtype=np.float32)
        # small amount of rotation
        for i in range(2):
            self.gaze_correction_transform_linear_mtx[i, i] += np.random.normal(0, 0.01)
        self.gaze_correction_translation_vec = np.zeros([2], dtype=np.float32)  # no shifting.

        # initialize the gaze correction transform with the "noisy" transform.
        self.gaze_correction_transform = GazeTransform(
            linear_transform=self.gaze_correction_transform_linear_mtx,
            translation=self.gaze_correction_translation_vec,
            pad_gaze_vector=False,
        )
        self.gaze_transform = GazeCorruption(
            bias_std=self.args.gaze_bias_std,
            noise_std=self.args.gaze_noise_std,
            transforms=[self.gaze_corruption_transform],
            x_weight_factor=self.args.weight_factor[0],
            y_weight_factor=self.args.weight_factor[1],
            is_spatially_varying=self.args.is_spatially_varying,
        )

    def _generate_miscalibration_transform(self, miscalibration_noise):
        print("MISCALIBRATION NOISE LEVEL ", miscalibration_noise)
        self.gaze_corruption_transform_linear_matrix = np.identity(2, dtype=np.float32)
        for i in range(2):
            for j in range(2):
                self.gaze_corruption_transform_linear_matrix[i, j] += np.random.normal(0, miscalibration_noise)

        self.gaze_corruption_transform_translation_vec = np.random.normal(0, miscalibration_noise, (2,))
        self.gaze_corruption_transform = GazeTransform(
            linear_transform=self.gaze_corruption_transform_linear_matrix,
            translation=self.gaze_corruption_transform_translation_vec,
            pad_gaze_vector=False,
        )

    def initialize_functors(self):
        # initialize input process dict

        def chm_calibration_input_functor(batch_input, aux_info_list, input_process_params):
            # batch_input is created in parse_data_item and has all necessary fields for performing model inference.
            # use cognitive_heatmap/datasets/chm_data_names.py
            should_use_batch = True
            gaze_key = "normalized_input_gaze"
            gaze_key_before_corruption = gaze_key + "_before_corruption"
            batch_input[gaze_key_before_corruption] = batch_input[gaze_key]
            gaze = batch_input[gaze_key]
            gaze = self.gaze_transform.corrupt_gaze(gaze)  # miscalibrates the gaze
            batch_input[gaze_key] = gaze
            # add field to indicate that the gaze should NOT be detached during training (in order to ensure that gradients flow all the way back to the correction transform)
            batch_input["no_detach_gaze"] = torch.tensor([True])
            return batch_input, aux_info_list, should_use_batch

        def chm_gaze_correction_input_functor(batch_input, aux_info_list, input_process_params):
            # this functor is used to apply the correction transform after parse_data_item.
            should_use_batch = True
            gaze_key = "normalized_input_gaze"
            gaze_key_before_correction = gaze_key + "_before_correction"
            batch_input[gaze_key_before_correction] = batch_input[gaze_key]
            # The transform happens on the the device on which training happens (typically the GPU).
            batch_input[gaze_key] = self.gaze_correction_transform(batch_input[gaze_key])
            return batch_input, aux_info_list, should_use_batch

        self.model_wrapper.input_process_dict = {}
        self.model_wrapper.input_process_dict["functor"] = functools.partial(chm_calibration_input_functor)
        self.model_wrapper.input_process_dict["post_parse_data_item"] = {}
        self.model_wrapper.input_process_dict["post_parse_data_item"]["functor"] = functools.partial(
            chm_gaze_correction_input_functor
        )
        self.model_wrapper.input_process_dict["post_parse_data_item"]["params"] = {}
        self.model_wrapper.input_process_dict["params"] = {}

        # initialize output process dict
        def chm_calibration_output_functor(
            training_output_dict, output_process_params, global_step, model=None, experiment_results_aggregator=None
        ):
            # error between the linear matrix of the correction transform and the true ground truth inverted linear matrix.
            lin_error = (
                self.gaze_correction_transform.lin_trans.weight
                - model.gaze_transform.lin_trans.weight.new_tensor(
                    output_process_params["ground_truth"]["inverted_tform"].lin_trans.weight.clone().detach().numpy()
                )
            )
            # error between the bias vector of the correction transform and the true ground truth inverted linear matrix.
            trans_error = (
                self.gaze_correction_transform.lin_trans.bias
                - model.gaze_transform.lin_trans.bias.new_tensor(
                    output_process_params["ground_truth"]["inverted_tform"].lin_trans.bias.clone().detach().numpy()
                )
            )
            # extract predicted output and the gaze input coordinates.
            gaze_map = training_output_dict["gaze_predicted_output"]["gaze_density_map"]
            log_normalized_gaze_map = training_output_dict["gaze_predicted_output"]["log_gaze_density_map"]
            gaze_coordinates = training_output_dict["gaze_batch_input"]["normalized_input_gaze"]
            groundtruth_gaze_coordinates = training_output_dict["gaze_batch_input"][
                "normalized_input_gaze_before_corruption"
            ]

            # metrics variables
            combined_nll = 0.0
            gaze_info_list = []
            nll_at_ground_truth_xy_list = []

            for b in range(gaze_coordinates.shape[0]):  # batch size
                gaze_info_list_b = []
                nll_at_ground_truth_xy_list_b = []
                for t in range(gaze_coordinates.shape[1]):  # sequence length
                    gaze_info_dict = {}
                    # get the 0th 'noisy' gaze point from the list of L gaze points
                    gaze = gaze_coordinates[b, t, 0, :]
                    # get the corresponding ground truth gaze (which the network has not seen)
                    groundtruth_gaze = groundtruth_gaze_coordinates[b, t, 0, :]
                    # scale the noisy gaze x properly to the size of the gaze map width (240)
                    x = int(gaze[0].cpu().item() * gaze_map.shape[-1])
                    # scale the noisy gaze y value to the size of the gaze map height( 135)
                    y = int(gaze[1].cpu().item() * gaze_map.shape[-2])
                    if x < 0 or y < 0 or x >= gaze_map.shape[-1] or y >= gaze_map.shape[-2]:
                        continue
                    x_gt = int(groundtruth_gaze[0].cpu().item() * gaze_map.shape[-1])  # scale to gaze map width
                    y_gt = int(groundtruth_gaze[1].cpu().item() * gaze_map.shape[-2])  # scale to gaze map height
                    if x_gt < 0 or y_gt < 0 or x_gt >= gaze_map.shape[-1] or y_gt >= gaze_map.shape[-2]:
                        continue

                    gaze_info_dict["groundtruth_coord"] = (x, y)
                    gaze_info_dict["noisy_coord"] = (x_gt, y_gt)
                    gaze_info_dict["groundtruth_gaze"] = groundtruth_gaze.detach().cpu().numpy().tolist()
                    gaze_info_dict["noisy_gaze"] = gaze.detach().cpu().numpy().tolist()
                    gaze_info_list_b.append(gaze_info_dict)

                    log_img_with_gaze = log_normalized_gaze_map[b, t, 0, :, :]
                    nll_at_ground_truth_xy = -log_img_with_gaze[y_gt, x_gt].detach().cpu().numpy().item()
                    nll_at_ground_truth_xy_list_b.append(float(nll_at_ground_truth_xy))
                    combined_nll += nll_at_ground_truth_xy

                gaze_info_list.append(gaze_info_list_b)
                nll_at_ground_truth_xy_list.append(nll_at_ground_truth_xy_list_b)

            # this element-wise transform diff will change during the training.
            transform_diff = torch.cat([lin_error, trans_error.unsqueeze(1)], dim=1).detach().cpu().numpy()
            # total error is computed as the sum of the squared error of the linear as well as the translation difference.
            error = ((lin_error ** 2).sum() + (trans_error ** 2).sum()).sqrt().detach().cpu().item()

            # logging the rotation matrix and translation vector
            current_correction_rotation = (
                self.gaze_correction_transform.lin_trans.weight.detach().cpu().numpy().tolist()
            )
            current_correction_translation = (
                self.gaze_correction_transform.lin_trans.bias.detach().cpu().numpy().tolist()
            )
            ground_truth_rotation = (
                self.output_process_dict["params"]["ground_truth"]["inverted_tform"]
                .lin_trans.weight.detach()
                .cpu()
                .numpy()
                .tolist()
            )
            ground_truth_translation = (
                self.output_process_dict["params"]["ground_truth"]["inverted_tform"]
                .lin_trans.bias.detach()
                .cpu()
                .numpy()
                .tolist()
            )

            print("Calibration Error", error)

            if experiment_results_aggregator is None:
                experiment_results_aggregator = {}
            if experiment_results_aggregator is not None:
                if "errors" not in experiment_results_aggregator:
                    experiment_results_aggregator["errors"] = []
                if GAZE_INFO not in experiment_results_aggregator:
                    experiment_results_aggregator[GAZE_INFO] = []
                if "nll_loss" not in experiment_results_aggregator:
                    experiment_results_aggregator["nll_loss"] = []
                if "nll_loss_bt" not in experiment_results_aggregator:
                    experiment_results_aggregator["nll_loss_bt"] = []

                experiment_results_aggregator["errors"].append(
                    {
                        "reference": output_process_params["reference"],
                        "error": error,
                        "transform_error": transform_diff.tolist(),
                        "current_correction_rotation": current_correction_rotation,
                        "current_correction_translation": current_correction_translation,
                        "ground_truth_rotation": ground_truth_rotation,
                        "ground_truth_translation": ground_truth_translation,
                    }
                )

                experiment_results_aggregator["nll_loss"].append(combined_nll)
                experiment_results_aggregator["nll_loss_bt"].append(nll_at_ground_truth_xy_list)
                experiment_results_aggregator[GAZE_INFO].append(gaze_info_list)

            experiment_results_aggregator["corruption_noise_std"] = self.gaze_transform.noise_std
            experiment_results_aggregator["corruption_bias_std"] = self.gaze_transform.bias_std

            return experiment_results_aggregator

        self.model_wrapper.output_process_dict = {}
        self.model_wrapper.output_process_dict["functor"] = functools.partial(chm_calibration_output_functor)
        self.model_wrapper.output_process_dict["params"] = {}
        self.model_wrapper.output_process_dict["params"]["ground_truth"] = {}
        inverted_tform_mtx, inverted_tform_vec = compute_inverted_affine_transform(
            self.gaze_corruption_transform_linear_matrix, self.gaze_corruption_transform_translation_vec
        )
        self.model_wrapper.output_process_dict["params"]["ground_truth"]["inverted_tform"] = GazeTransform(
            linear_transform=inverted_tform_mtx, translation=inverted_tform_vec
        )

        def param_grad_setter(model, correction_transform=None):
            device = next(model.parameters()).device
            correction_transform.to(device)
            for p in correction_transform.parameters():
                p.requires_grad = True
            params = correction_transform.parameters()
            return model, list(params)

        self.params_dict["param_grad_setter"] = functools.partial(
            param_grad_setter, correction_transform=self.gaze_correction_transform
        )
        if self.params_dict["max_overall_batch_during_training"] is None:
            self.params_dict["max_overall_batch_during_training"] = 1000
        self.params_dict["no_save_model"] = True
        self.params_dict["no_run_test"] = True


def arg_setter(parser):
    parser.add_argument(
        "--miscalibration_noise_levels",
        action="store",
        nargs="*",
        type=float,
        default=[1.0e-3, 1.0e-2, 1.0e-1],
        help="Noise levels for different experiments",
    )

    parser.add_argument(
        "--miscalibration_noise",
        action="store",
        nargs="*",
        type=float,
        default=0.01,
        help="Noise level for miscalibration",
    )

    parser.add_argument(
        "--num_optimization_runs",
        action="store",
        type=int,
        default=15,
        help="Number of times the optimization will be run for a single noise level",
    )

    parser.add_argument(
        "--weight_factor",
        action="store",
        nargs="*",
        type=float,
        default=[1.0, 1.3],
        help="Weight factor for x and y dimensions for spatially varying noise",
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

    for miscalibration_noise_level in args.miscalibration_noise_levels:
        for optimization_num in range(args.num_optimization_runs):
            print("Noise level: {} Run number : {}".format(miscalibration_noise_level, optimization_num))
            args_copy = copy.deepcopy(args)
            # very very small side channel noise
            args_copy.gaze_bias_std = 1e-8
            args_copy.gaze_noise_std = 1e-8
            args_copy.miscalibration_noise = miscalibration_noise_level

            # instantiate the experiment
            calibration_experiment = ChmCalibrationOptimizationExperiment(args_copy, session_hash)
            calibration_experiment.initialize_functors()
            calibration_experiment.perform_experiment()

            # file name components
            filename_variables = OrderedDict()
            filename_variables["type"] = "gaze_calibration"
            filename_variables["miscalibration_noise_level"] = miscalibration_noise_level
            filename_variables["optimization_run_num"] = optimization_num
            filename_variables["filename_append"] = args_copy.filename_append

            # save json after experiment
            calibration_experiment.save_experiment(name_values=filename_variables)


if __name__ == "__main__":
    main()