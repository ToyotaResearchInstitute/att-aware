# Copyright 2020 Toyota Research Institute.  All rights reserved.
import uuid
import copy
import functools
import numpy as np
import cv2

from maad.configs.args_file import parse_arguments
from maad.experiments.maad_experiments import MAADExperiment

from maad.model.gaze_transform import GazeTransform
from maad.model.gaze_corruption import GazeCorruption
from maad.utils.maad_consts import *

# (NOTE) When running this script please make sure that you are using --batch_size = 1

# Here is an example of how to use this script to generate saliency maps for video_id=6, task=control and subject=2
# for frames 1800 to 2700


# python generate_saliency_maps_using_model.py --batch_size 1 --inference_ds_type train --train_sequence_ids 6
# --train_task_ids control --train_subject_ids 2 --add_optic_flow --use_s3d --enable_amp
# --batch_inference_frame_id_range 1800 2700 --load_model_path ~/maad/models/TRAINING_HASH/MODEL.pt

# Note that by NOT using --use_std_train_test_split arg, the dataset is split according to the original dreyeve data split.
# Therefore for generating saliency maps for video ids 53 and 60 you will need to use --inference_ds_type=test


class GenerateSaliencyMapsUsingModel(MAADExperiment):
    """
    Class to generate saliency maps using the model. 
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
        self.force_dropout = True  # For pure gaze saliency dropout the side channel gaze completely
        self.gaze_corruption = GazeCorruption(bias_std=args.gaze_bias_std, noise_std=args.gaze_noise_std)

    def initialize_functors(self):

        # input functor to alter gaze points for specific frames for testing if the awareness map is indeed using the gaze information
        # only effective if 'gaze_alteration_frame_id_range' is not = []. Default is [] and therefore will be bypassed
        def alter_gaze_points_for_frame_ids_input_functor(batch_input, aux_info_list, input_process_params):
            """
            This functor is responsible for artificially altering the input gaze points to a fixed point in the image
            if the frame_id of the frame falls within a predefined range
            """

            def check_if_batch_should_be_used(batch_input, aux_info_list, batch_inference_frame_id_range):
                """
                Used for selecting specific frames for inference. Note that batch size is always set to be 1. 
                """
                image_ids = np.array(
                    [ai["road_id"][1].cpu().detach().numpy() for ai in aux_info_list]
                ).flatten()  # (T, )

                should_use_batch = True
                # check if the last idx is in the batch_inference_frame_id_range
                if image_ids[-1] in list(range(batch_inference_frame_id_range[0], batch_inference_frame_id_range[1])):
                    should_use_batch = True
                else:
                    should_use_batch = False

                return should_use_batch

            batch_inference_frame_id_range = input_process_params["batch_inference_frame_id_range"]
            assert len(batch_inference_frame_id_range) > 0
            should_use_batch = check_if_batch_should_be_used(batch_input, aux_info_list, batch_inference_frame_id_range)

            gaze_key = "normalized_input_gaze"
            gaze_key_before_corruption = gaze_key + "_before_corruption"
            batch_input[gaze_key_before_corruption] = batch_input[gaze_key]
            gaze = batch_input[gaze_key]
            gaze = self.gaze_corruption.corrupt_gaze(gaze)
            batch_input[gaze_key] = gaze

            # the frame id range for which the gaze will be altered
            gaze_alteration_frame_id_range = input_process_params["gaze_alteration_frame_id_range"]
            if len(gaze_alteration_frame_id_range) > 0 and should_use_batch:
                # the fixed gaze point to which the gaze point
                gaze_fixed_point = input_process_params["gaze_fixed_point"]
                gaze_key = "normalized_input_gaze"
                gaze_key_before_alteration = gaze_key + "_before_alteration"
                gaze_points = batch_input[gaze_key]  # (B, T, L, 2)
                batch_input[gaze_key_before_alteration] = batch_input[gaze_key]
                # grab the image ids from the aux_info_list. For all the frame_ids within the range, alter
                image_ids = np.array(
                    [ai["road_id"][1].cpu().detach().numpy() for ai in aux_info_list]
                ).flatten()  # (T, ) #works because B = 1 for generating saliency maps,
                image_id_within_frame_id_range = np.logical_and(
                    image_ids > gaze_alteration_frame_id_range[0], image_ids < gaze_alteration_frame_id_range[1]
                )  # (T, )

                index_at_which_true = list(
                    np.where(
                        np.logical_and(
                            image_ids > gaze_alteration_frame_id_range[0], image_ids < gaze_alteration_frame_id_range[1]
                        )
                    )[0]
                )

                if len(index_at_which_true) > 0:
                    altered_gaze = torch.ones_like(gaze_points[:, index_at_which_true, :, :])  # (B, t, L, 2)
                    altered_gaze[:, :, :, 0:1] = gaze_fixed_point[0]  # (B, t, L, 1)
                    altered_gaze[:, :, :, 1:2] = gaze_fixed_point[1]  # (B, t, L, 1)
                    gaze_points[:, index_at_which_true, :, :] = altered_gaze  # (B,t,L,2)
                    # replace normalized_input_gaze with altered gaze
                    batch_input[gaze_key] = gaze_points

            return batch_input, aux_info_list, should_use_batch

        def save_images_in_folder_output_functor(
            inference_output_dict, output_process_params, global_step, model=None, experiment_results_aggregator=None
        ):

            assert "output_folder" in output_process_params
            assert "is_save_only_last_frame" in output_process_params
            assert "is_save_gaze_points_on_image" in output_process_params

            def convert_heatmap_to_image(
                img, batch_input, batch_target, batch_idx, time_idx, is_save_gaze_points_on_image, color_range=None
            ):
                if color_range is None:
                    img_qs = np.percentile(img, [1, 99])
                    img[img <= img_qs[0]] = img_qs[0]
                    img[img > img_qs[1]] = img_qs[1]
                else:
                    img_qs = color_range

                img = (img - img_qs[0]) / (img_qs[1] - img_qs[0])
                # Normalize to 0-1.0
                # img = (img - np.min(img)) / (np.max(img) - np.min(img))
                # img = 1-img
                img_cv_bgr = cv2.applyColorMap(np.uint8(img * 255), cv2.COLORMAP_JET)  # (H, W, 3) BGR

                # img_cv_bgr = np.float32(img_cv_bgr) / 255
                # img_cv_bgr = np.array(img_cv_bgr * 255, dtype=np.uint8)
                if is_save_gaze_points_on_image:
                    train_bits = batch_input["should_train_input_gaze"][batch_idx, time_idx, :, :]  # (L, 1)
                    target = batch_target[batch_idx, time_idx, :, :].cpu().detach()  # [L, 2]
                    target = target.numpy()

                    noisy_gaze = (
                        batch_input["normalized_input_gaze"][batch_idx, time_idx, :, :].cpu().detach().numpy()
                    )  # (L, 2)

                    img_height, img_width = img_cv_bgr.shape[:2]
                    img_dim = np.array((img_width, img_height))
                    img_dim = np.tile(img_dim, reps=(noisy_gaze.shape[0], 1))  # (TL, 2)

                    noisy_gaze = np.multiply(noisy_gaze, img_dim)

                    assert target.shape[0] == noisy_gaze.shape[0]
                    true_inds = (train_bits[:, 0] == 1).nonzero()[:, 0]
                    true_inds = true_inds.cpu().detach().numpy()
                    circle_size = 4
                    if true_inds.shape[0] > 0:
                        noisy_gaze = noisy_gaze[true_inds, :]  # (M, 2) with M > 0
                        target = target[true_inds, :]
                        img_cv_bgr = cv2.circle(
                            img_cv_bgr,
                            tuple(np.float32(target[0, :].flatten())),
                            radius=circle_size + 2,
                            color=(50, 205, 50),  # lime for true
                            thickness=2,
                        )  # plot true gaze

                        img_cv_bgr = cv2.circle(
                            img_cv_bgr,
                            tuple(np.float32(noisy_gaze[0, :].flatten())),
                            radius=circle_size,
                            color=(255, 255, 255),  # white for noisy
                            thickness=2,
                        )  # noisy gaze point as white. Make sure that the 'center' parameters for cv2.circle is np.float32. Doesn't work if it is np.float64

                return img_cv_bgr

            batch_input = inference_output_dict["batch_input"]
            aux_info_list = inference_output_dict["aux_info_list"]
            batch_target = inference_output_dict["batch_target"]
            data_item = inference_output_dict["data_item"]
            predicted_gaze_with_gaze = inference_output_dict["predicted_gaze_with_gaze"]
            predicted_gaze_without_gaze = inference_output_dict["predicted_gaze_without_gaze"]

            text_desc = output_process_params["text_desc"] if "text_desc" in output_process_params else "inference"
            output_folder = output_process_params["output_folder"]
            is_save_only_last_frame = output_process_params["is_save_only_last_frame"]
            is_save_gaze_points_on_image = output_process_params["is_save_gaze_points_on_image"]

            assert batch_target.shape[0] == 1  # (B = 1)?

            for b in range(batch_target.shape[0]):
                seq_len = batch_target.shape[1]  # len of the slice used as input for prediction
                if is_save_only_last_frame:
                    t_list = [-1]
                else:
                    t_list = range(seq_len)

                for t in t_list:
                    # Weird ordering due to the "list nature of aux_info_list"
                    print("Time step ", t)
                    frame_idx = aux_info_list[t][AUXILIARY_INFO_FRAME_IDX][b].numpy().item()
                    # for example, '06'
                    video_id_num = "{0:02d}".format(aux_info_list[t][AUXILIARY_INFO_VIDEO_ID][b])
                    gaze_hmap_with_gaze = (
                        predicted_gaze_with_gaze["gaze_density_map"][b, t, 0, :, :].cpu().detach().numpy()
                    )
                    gaze_hmap_with_gaze_bgr = convert_heatmap_to_image(
                        gaze_hmap_with_gaze,
                        batch_input,
                        batch_target,
                        b,
                        t,
                        is_save_gaze_points_on_image,
                        color_range=None,
                    )
                    gaze_hmap_without_gaze = (
                        predicted_gaze_without_gaze["gaze_density_map"][b, t, 0, :, :].cpu().detach().numpy()
                    )
                    gaze_hmap_without_gaze_bgr = convert_heatmap_to_image(
                        gaze_hmap_without_gaze,
                        batch_input,
                        batch_target,
                        b,
                        t,
                        is_save_gaze_points_on_image,
                        color_range=None,
                    )
                    awareness_hmap_with_gaze = (
                        predicted_gaze_with_gaze["awareness_map"][b, t, 0, :, :].cpu().detach().numpy()
                    )
                    awareness_hmap_with_gaze_bgr = convert_heatmap_to_image(
                        awareness_hmap_with_gaze,
                        batch_input,
                        batch_target,
                        b,
                        t,
                        is_save_gaze_points_on_image,
                        color_range=[0, 1],
                    )

                    awareness_hmap_without_gaze = (
                        predicted_gaze_without_gaze["awareness_map"][b, t, 0, :, :].cpu().detach().numpy()
                    )
                    awareness_hmap_without_gaze_bgr = convert_heatmap_to_image(
                        awareness_hmap_without_gaze,
                        batch_input,
                        batch_target,
                        b,
                        t,
                        is_save_gaze_points_on_image,
                        color_range=[0, 1],
                    )

                    gaze_hmap_with_gaze_path = os.path.join(
                        output_folder,
                        "VIDEO_ID_{0}_frame_num_{1:08d}_input_seq_len_{2}_t_{3}_gaze_hmap_with_gaze.png".format(
                            video_id_num, frame_idx, seq_len, t
                        ),
                    )
                    gaze_hmap_without_gaze_path = os.path.join(
                        output_folder,
                        "VIDEO_ID_{0}_frame_num_{1:08d}_input_seq_len_{2}_t_{3}_gaze_hmap_without_gaze.png".format(
                            video_id_num, frame_idx, seq_len, t
                        ),
                    )
                    awareness_hmap_with_gaze_path = os.path.join(
                        output_folder,
                        "VIDEO_ID_{0}_frame_num_{1:08d}_input_seq_len_{2}_t_{3}_awareness_hmap_with_gaze.png".format(
                            video_id_num, frame_idx, seq_len, t
                        ),
                    )
                    awareness_hmap_without_gaze_path = os.path.join(
                        output_folder,
                        "VIDEO_ID_{0}_frame_num_{1:08d}_input_seq_len_{2}_t_{3}_awareness_hmap_without_gaze.png".format(
                            video_id_num, frame_idx, seq_len, t
                        ),
                    )

                    cv2.imwrite(gaze_hmap_with_gaze_path, gaze_hmap_with_gaze_bgr)
                    cv2.imwrite(gaze_hmap_without_gaze_path, gaze_hmap_without_gaze_bgr)
                    cv2.imwrite(awareness_hmap_with_gaze_path, awareness_hmap_with_gaze_bgr)
                    cv2.imwrite(awareness_hmap_without_gaze_path, awareness_hmap_without_gaze_bgr)
                    print("Saved hmaps for frame id ", frame_idx)

        self.input_process_dict = {}
        self.input_process_dict["functor"] = functools.partial(alter_gaze_points_for_frame_ids_input_functor)
        self.input_process_dict["params"] = {}
        self.input_process_dict["params"]["batch_inference_frame_id_range"] = args.batch_inference_frame_id_range
        self.input_process_dict["params"]["gaze_alteration_frame_id_range"] = args.gaze_alteration_frame_id_range
        # fixed point to which the gaze will be fixed
        self.input_process_dict["params"]["gaze_fixed_point"] = args.gaze_fixed_point
        self.input_process_dict["inference_mode"] = InferenceMode.BOTH

        self.output_process_dict = {}
        self.output_process_dict["functor"] = functools.partial(save_images_in_folder_output_functor)
        self.output_process_dict["params"] = {}
        self.output_process_dict["params"]["writer"] = self.writer
        self.output_process_dict["params"]["batch_size"] = args.batch_size
        self.output_process_dict["params"]["is_save_only_last_frame"] = True
        self.output_process_dict["params"]["is_save_gaze_points_on_image"] = True
        self.output_process_dict["params"]["text_desc"] = "saliency_maps_using_inference"
        self.output_process_dict["params"]["output_folder"] = os.path.join(self.log_dir, "saliency_maps")

    def perform_experiment(self):
        self._perform_experiment()


def arg_setter(parser):
    parser.add_argument(
        "--gaze_alteration_frame_id_range",
        nargs="*",
        type=int,
        default=[],
        help="list containing the min and max frame ids for artifical altering of gaze during generating saliency maps",
    )

    parser.add_argument(
        "--batch_inference_frame_id_range",
        nargs="*",
        type=int,
        default=[0, 7500],
        help="min and max id range for using the batch for inference",
    )
    parser.add_argument("--gaze_fixed_point", nargs="*", type=float, default=[0.1, 0.1], help="artifical gaze value")


if __name__ == "__main__":
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash, [arg_setter])
    experiment = GenerateSaliencyMapsUsingModel(session_hash, args)
    experiment.initialize_functors()
    experiment.perform_experiment()
