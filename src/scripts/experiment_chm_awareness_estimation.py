import uuid
import copy
import functools
import numpy as np
import torch

from collections import OrderedDict

from chm.configs.args_file import parse_arguments
from chm.experiments.chm_experiments import ChmExperiment

from chm.utils.chm_consts import InferenceMode
from chm.utils.experiment_result_keys import *

from chm.model.gaze_corruption import GazeCorruption

from chm.utils.experiment_utils import SpatioTemporalGaussianWithOpticFlowAwarenessEstimator


class AwarenessEstimationExperiment(ChmExperiment):
    def __init__(self, args, session_hash):
        super().__init__(args, session_hash, training_experiment=False)

        self.args = args
        self.gaze_corruption = GazeCorruption(
            bias_std=self.args.gaze_bias_std, noise_std=self.args.gaze_noise_std
        )  # noise to the side channel gaze

        # filter parameters
        self.gaussian_estimator_spatial_scale = self.args.gaussian_estimator_spatial_scale
        self.gaussian_estimator_temporal_scale = self.args.gaussian_estimator_temporal_scale
        self.sigma_kernel = self.args.sigma_kernel
        self.gaussian_estimator_temporal_decay = self.args.gaussian_estimator_temporal_decay
        self.temporal_filter_type = self.args.temporal_filter_type
        self.annotation_image_size = self.args.annotation_image_size

        self.result_keys = [
            AWARENESS_ERROR_CHM_KEY,
            AWARENESS_ERROR_OF_SPATIOTEMPORAL_GAUSSIAN_KEY,
            AWARENESS_ABS_ERROR_CHM_KEY,
            AWARENESS_ABS_ERROR_OF_SPATIOTEMPORAL_GAUSSIAN_KEY,
            AWARENESS_TARGET_KEY,
            AWARENESS_ESTIMATE_CHM_KEY,
            AWARENESS_ESTIMATE_OF_SPATIOTEMPORAL_GAUSSIAN_KEY,
            GAZE_INFO,
        ]

    def initialize_functors(self):

        # initialize input process dict and functor
        def awareness_input_functor(batch_input, aux_info_list, input_process_params):
            should_use_batch = True
            gaze_key = "normalized_input_gaze"
            gaze_key_before_corruption = gaze_key + "_before_corruption"
            # save the gaze point before corruption.
            batch_input[gaze_key_before_corruption] = batch_input[gaze_key]
            gaze = batch_input[gaze_key]
            gaze = self.gaze_corruption.corrupt_gaze(gaze)  # corrupt the gaze
            batch_input[gaze_key] = gaze  # replace 'normalized_input_gaze' with the noisy version.
            return batch_input, aux_info_list, should_use_batch

        self.model_wrapper.input_process_dict = {}
        self.model_wrapper.input_process_dict["functor"] = functools.partial(awareness_input_functor)
        self.model_wrapper.input_process_dict["params"] = {}
        self.model_wrapper.input_process_dict["inference_mode"] = InferenceMode.WITH_GAZE
        self.model_wrapper.input_process_dict["max_batch_num"] = self.args.max_inference_num_batches

        # initialize output process dict and functor
        def awareness_output_functor(
            inference_output_dict, output_process_params, global_step, model=None, experiment_results_aggregator=None
        ):
            gaze_map_with_gaze = inference_output_dict["predicted_gaze_with_gaze"]["gaze_density_map"]
            predicted_awareness_training_output = inference_output_dict["predicted_awareness_output_with_gaze"]

            awareness_batch_input = inference_output_dict["awareness_batch_input"]
            awareness_batch_annotation_data = inference_output_dict["awareness_batch_annotation_data"]
            awareness_aux_info_list = inference_output_dict["awareness_aux_info_list"]

            # (aB, T, L, 2) #the gaze points for the awareness label ds. aB refers to the awareness dataset batch size (--awareness_ds_batch_size).
            noisy_gaze = awareness_batch_input["normalized_input_gaze"]
            groundtruth_gaze = awareness_batch_input["normalized_input_gaze_before_corruption"]

            # Get awarenessmap and target coordinates from awareness labels batch
            # (B, H, W), -1 for T dim because the label is only for the last time stamp. 0 for the C dimension because, awareness heatmap is single channel image
            awareness_map = predicted_awareness_training_output["awareness_map"][:, -1, 0, :, :]
            awareness_x = awareness_map.new_tensor(
                (awareness_batch_annotation_data["query_x"] / self.annotation_image_size[2] * awareness_map.shape[-1])
                .clone()
                .detach()
                .numpy()
            ).int()  # (B)
            awareness_y = awareness_map.new_tensor(
                (awareness_batch_annotation_data["query_y"] / self.annotation_image_size[1] * awareness_map.shape[-2])
                .clone()
                .detach()
                .numpy()
            ).int()  # (B)

            # list of [0,1,2....aB]
            awareness_b = awareness_x.new_tensor(list(range(awareness_x.shape[0])))

            of_st_awareness_estimates = []
            chm_awareness_estimates = []

            # init error lists
            awareness_sq_err_chm = []
            awareness_abs_err_chm = []
            awareness_sq_err_of_spatiotemporal_gaussian = []
            awareness_abs_err_of_spatiotemporal_gaussian = []

            # init experiment results aggregator
            if experiment_results_aggregator is None:
                experiment_results_aggregator = OrderedDict()  # create a dict to aggregate results

            for RESULT_KEY in self.result_keys:
                if RESULT_KEY not in experiment_results_aggregator:
                    experiment_results_aggregator[RESULT_KEY] = []

            gaze_info_list = []
            awareness_target = awareness_batch_annotation_data["annotation_target"]

            for b, x, y in zip(awareness_b, awareness_x, awareness_y):
                gaze_info_list_b = []
                road_images = awareness_batch_input["road_image"][b, :, :, :, :]  # (T, 3, H, W)
                gaze_inputs = noisy_gaze[b, :, 0, :].cpu().float()
                # scale it up to the low res dimensions (for width dimension)
                gaze_inputs[:, 0] = gaze_inputs[:, 0] * awareness_map.shape[-1]
                # scale it up to the low res dimensions (for height dimension)
                gaze_inputs[:, 1] = gaze_inputs[:, 1] * awareness_map.shape[-2]
                gtruth_gaze_inputs = groundtruth_gaze[b, :, 0, :].cpu().float()
                gtruth_gaze_inputs[:, 0] = gtruth_gaze_inputs[:, 0] * awareness_map.shape[-1]
                gtruth_gaze_inputs[:, 1] = gtruth_gaze_inputs[:, 1] * awareness_map.shape[-2]

                for t in range(gaze_inputs.shape[0]):
                    gaze_info_dict = {}
                    gaze_info_dict["noisy_gaze"] = noisy_gaze[b, t, 0, :].cpu().numpy().tolist()
                    gaze_info_dict["groundtruth_gaze"] = groundtruth_gaze[b, t, 0, :].cpu().numpy().tolist()
                    gaze_info_dict["noisy_coord"] = gaze_inputs[t, :].cpu().numpy().tolist()
                    gaze_info_dict["groundtruth_coord"] = gtruth_gaze_inputs[t, :].cpu().numpy().tolist()
                    gaze_info_list_b.append(gaze_info_dict)

                gaze_info_list.append(gaze_info_list_b)
                should_train_bit_sequence = (
                    awareness_batch_input["should_train_input_gaze"][b, :, 0, 0].cpu().float()
                )  # (T)
                # (T, 2, H, W)
                optic_flow_sequence = awareness_batch_input["optic_flow_image"][b, :, :, :, :].cpu()
                spatio_temporal_gaussian_filter_with_of_estimator = (
                    SpatioTemporalGaussianWithOpticFlowAwarenessEstimator(
                        gaze_inputs=gaze_inputs,
                        should_train_bit_sequence=should_train_bit_sequence,
                        optic_flow_sequence=optic_flow_sequence,
                        sigma_kernel=self.sigma_kernel,
                        spatial_scale=self.gaussian_estimator_spatial_scale,
                        temporal_scale=self.gaussian_estimator_temporal_scale,
                        temporal_decay=self.gaussian_estimator_temporal_decay,
                        temporal_filter_type=self.temporal_filter_type,
                    )
                )
                # length of slice
                frame_num = predicted_awareness_training_output["awareness_map"].shape[1]
                coordinate = torch.tensor([x, y]).float()  # annotation target for bth batch
                # get the estimate at coordinate in the last frame using the baseline gaussian
                (
                    st_filtered_awareness_sequence_overall,
                    of_based_spatio_temporal_gaussian_baseline,
                ) = spatio_temporal_gaussian_filter_with_of_estimator.estimate_awareness(
                    frame=frame_num - 1, coordinate=coordinate
                )
                # get estimate at coordinate using CHM
                chm_estimate = float(awareness_map[b, y, x].cpu().float())

                # append results
                of_st_awareness_estimates.append(of_based_spatio_temporal_gaussian_baseline)
                chm_awareness_estimates.append(chm_estimate)
                target = float(awareness_target[b].cpu().float())

                # append metrics
                awareness_sq_err_chm.append((chm_estimate - target) ** 2)
                awareness_abs_err_chm.append(chm_estimate - target)

                awareness_sq_err_of_spatiotemporal_gaussian.append(
                    (of_based_spatio_temporal_gaussian_baseline - target) ** 2
                )
                awareness_abs_err_of_spatiotemporal_gaussian.append(of_based_spatio_temporal_gaussian_baseline - target)

            print(
                "Sq Errors: chm: {},  OF SpatioTemporal {}".format(
                    np.average(awareness_sq_err_chm), np.average(awareness_sq_err_of_spatiotemporal_gaussian)
                )
            )
            print(
                "Abs Errors: chm: {},  OF SpatioTemporal {}".format(
                    np.average(awareness_abs_err_chm), np.average(awareness_abs_err_of_spatiotemporal_gaussian)
                )
            )

            # if any of the errors are greater than one then there is an error.
            if (
                np.max(np.abs(awareness_sq_err_chm)) > 1
                or np.max(np.abs(awareness_sq_err_of_spatiotemporal_gaussian)) > 1
                or np.max(np.abs(awareness_abs_err_chm)) > 1
                or np.max(np.abs(awareness_abs_err_of_spatiotemporal_gaussian)) > 1
            ):
                import IPython

                IPython.embed(header="check error")

            # append results to the experiment_results_aggregator dict
            experiment_results_aggregator[GAZE_INFO].append(gaze_info_list)
            experiment_results_aggregator[AWARENESS_ERROR_CHM_KEY].extend(awareness_sq_err_chm)
            experiment_results_aggregator[AWARENESS_ERROR_OF_SPATIOTEMPORAL_GAUSSIAN_KEY].extend(
                awareness_sq_err_of_spatiotemporal_gaussian
            )

            experiment_results_aggregator[AWARENESS_ABS_ERROR_CHM_KEY].extend(awareness_abs_err_chm)
            experiment_results_aggregator[AWARENESS_ABS_ERROR_OF_SPATIOTEMPORAL_GAUSSIAN_KEY].extend(
                awareness_abs_err_of_spatiotemporal_gaussian
            )

            experiment_results_aggregator[AWARENESS_TARGET_KEY].extend(awareness_target.cpu().float().tolist())
            experiment_results_aggregator[AWARENESS_ESTIMATE_CHM_KEY].extend(chm_awareness_estimates)
            experiment_results_aggregator[AWARENESS_ESTIMATE_OF_SPATIOTEMPORAL_GAUSSIAN_KEY].extend(
                of_st_awareness_estimates
            )

            # compute mean and std for the errors
            for key in [
                AWARENESS_ERROR_CHM_KEY,
                AWARENESS_ABS_ERROR_CHM_KEY,
                AWARENESS_ERROR_OF_SPATIOTEMPORAL_GAUSSIAN_KEY,
                AWARENESS_ABS_ERROR_OF_SPATIOTEMPORAL_GAUSSIAN_KEY,
            ]:
                experiment_results_aggregator["overall_" + key + "_mean"] = np.mean(experiment_results_aggregator[key])
                experiment_results_aggregator["overall_" + key + "_std"] = np.std(experiment_results_aggregator[key])
                experiment_results_aggregator["overall_" + key + "_median"] = np.median(
                    experiment_results_aggregator[key]
                )
                experiment_results_aggregator["overall_size"] = len(experiment_results_aggregator[key])

            experiment_results_aggregator["corruption_bias_std"] = self.gaze_corruption.bias_std
            experiment_results_aggregator["corruption_noise_std"] = self.gaze_corruption.noise_std
            experiment_results_aggregator["gaussian_estimator_spatial_scale"] = self.gaussian_estimator_spatial_scale
            experiment_results_aggregator["gaussian_estimator_temporal_scale"] = self.gaussian_estimator_temporal_scale
            experiment_results_aggregator["sigma_kernel"] = self.sigma_kernel

            return experiment_results_aggregator

        self.model_wrapper.output_process_dict = {}
        self.model_wrapper.output_process_dict["functor"] = functools.partial(awareness_output_functor)
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
        help="Noise levels for experiments",
    )
    parser.add_argument(
        "--gaussian_estimator_spatial_scale",
        action="store",
        type=int,
        default=20,
        help="Spatial scale for spatio-temporal gaussian filter in pixels",
    )
    parser.add_argument(
        "--annotation_image_size",
        type=int,
        default=[3, 1080, 1920],
        help="Dimensions of the full original road_image in pixels used for annotation",
    )

    parser.add_argument(
        "--gaussian_estimator_temporal_scale",
        action="store",
        type=float,
        default=3.0,
        help="Temporal scale for spatio-temporal gaussian filter in frame count",
    )
    parser.add_argument(
        "--gaussian_estimator_temporal_decay",
        action="store",
        type=float,
        default=0.8,
        help="Temporal decay when using geometric filter",
    )
    parser.add_argument(
        "--temporal_filter_type",
        type=str,
        default="exponential",
        help="Type for temporal filtering used in the baseline estimator",
    )
    parser.add_argument(
        "--sigma_kernel",
        action="store",
        type=int,
        default=10,
        help="Sigma for kernel in optic flow based spatiotemporal filter",
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
        args_copy.gaze_bias_std = 1e-8
        args_copy.gaze_noise_std = noise_level
        args_copy.experiment_noise_levels = []

        # instantiate the experiment
        awareness_estimation_experiment = AwarenessEstimationExperiment(args_copy, session_hash)
        awareness_estimation_experiment.initialize_functors()
        awareness_estimation_experiment.perform_experiment()

        # file name components
        filename_variables = OrderedDict()
        filename_variables["type"] = "awareness_estimation"
        filename_variables["noise_level"] = noise_level
        filename_variables["gaussian_spatial_scale"] = args_copy.gaussian_estimator_spatial_scale
        filename_variables["gaussian_temporal_scale"] = args_copy.gaussian_estimator_temporal_scale
        filename_variables["sigma_kernel"] = args_copy.sigma_kernel
        filename_variables["filename_append"] = args_copy.filename_append
        awareness_estimation_experiment.save_experiment(name_values=filename_variables)


if __name__ == "__main__":
    main()
