# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch
import numpy as np
import itertools

from maad.losses.regularizations import EPSpatialRegularization, EPTemporalRegularization
from maad.losses.awareness_label_loss import AwarenessPointwiseLabelLoss


class MAADLoss(object):
    def __init__(self, params_dict, gt_prior_loss=None):
        """
        Class for computing the total loss for MAAD.
        Parameters:
        ----------
        params_dict: dict
            params dict containing the args from args_file.py
        gt_prior_loss: functools.partial
            function handle to compute the loss term on the gaze transform module
        """
        self.params_dict = params_dict

        self.aspect_ratio_reduction_factor = self.params_dict.get("aspect_ratio_reduction_factor", 8)

        # full size road image dimensions
        self.ORIG_ROAD_IMG_DIMS = self.params_dict.get("orig_road_image_dims")
        self.ORIG_ROAD_IMAGE_HEIGHT = self.ORIG_ROAD_IMG_DIMS[1]
        self.ORIG_ROAD_IMAGE_WIDTH = self.ORIG_ROAD_IMG_DIMS[2]

        # network image dimensions
        self.image_width = int(round(self.ORIG_ROAD_IMAGE_WIDTH / self.aspect_ratio_reduction_factor))
        self.image_height = int(round(self.ORIG_ROAD_IMAGE_HEIGHT / self.aspect_ratio_reduction_factor))
        self.add_optic_flow = self.params_dict.get("add_optic_flow", False)
        self.gaussian_kernel_size = self.params_dict.get("gaussian_kernel_size", 2)
        self.NEGATIVE_DIFFERENCE_COEFFICIENT = self.params_dict.get("negative_difference_coeff", 10.0)
        self.POSITIVE_DIFFERENCE_COEFFICIENT = self.params_dict.get("positive_difference_coeff", 1.0)
        regularization_eps = self.params_dict.get("regularization_eps", 1e-3)
        sig_scale_factor = self.params_dict.get("sig_scale_factor", 1)

        # spatial and temporal regularization
        self.spatial_regularization = EPSpatialRegularization(
            image_width=self.image_width,
            image_height=self.image_height,
            eps=regularization_eps,
            sig_scale_factor=sig_scale_factor,
        )
        self.temporal_regularization = EPTemporalRegularization(
            image_width=self.image_width,
            image_height=self.image_height,
            eps=regularization_eps,
            sig_scale_factor=sig_scale_factor,
            negative_difference_coeff=self.NEGATIVE_DIFFERENCE_COEFFICIENT,
            positive_difference_coeff=self.POSITIVE_DIFFERENCE_COEFFICIENT,
        )

        # Cost coeffs and parameters

        # main cost term coeffs
        self.gaze_data_coeff = self.params_dict.get("gaze_data_coeff", 1.2)
        self.logprob_gap = self.params_dict.get("logprob_gap", 10)
        self.awareness_at_gaze_points_loss_coeff = self.params_dict.get("awareness_at_gaze_points_loss_coeff", 1)

        # temporal and spatial regularizations parameters
        self.gaze_spatial_regularization_coeff = self.params_dict.get("gaze_spatial_regularization_coeff", 1.0)
        self.gaze_temporal_regularization_coeff = self.params_dict.get("gaze_temporal_regularization_coeff", 1.0)
        self.awareness_spatial_regularization_coeff = self.params_dict.get(
            "awareness_spatial_regularization_coeff", 1.0
        )
        self.awareness_temporal_regularization_coeff = self.params_dict.get(
            "awareness_temporal_regularization_coeff", 1.0
        )
        # decay loss parameters
        self.awareness_decay_coeff = self.params_dict.get("awareness_decay_coeff", 1.0)
        self.awareness_decay_alpha = self.params_dict.get("awareness_decay_alpha", 1.0)

        # steady state loss parameters
        self.awareness_steady_state_coeff = self.params_dict.get("awareness_steady_state_coeff", 0.01)

        # consistency loss parameters
        self.consistency_coeff_gaze = self.params_dict.get("consistency_coeff_gaze", 10)
        self.consistency_coeff_awareness = self.params_dict.get("consistency_coeff_awareness", 10)

        # annotation label loss parameters
        self.awareness_loss_type = self.params_dict.get("awareness_loss_type", "huber_loss")
        self.awareness_label_loss_patch_half_size = self.params_dict.get("awareness_label_loss_patch_half_size", 4)
        self.awareness_label_loss = AwarenessPointwiseLabelLoss(
            loss_type=self.awareness_loss_type,
            patch_half_size=self.awareness_label_loss_patch_half_size,
            annotation_image_size=self.ORIG_ROAD_IMG_DIMS,
        )
        self.awareness_label_coeff = self.params_dict.get("awareness_label_coeff", 1.0)

        # optic flow based temporal regularization parameters
        self.optic_flow_temporal_smoothness_decay = self.params_dict.get("optic_flow_temporal_smoothness_decay", 1.0)
        self.optic_flow_temporal_smoothness_coeff = self.params_dict.get("optic_flow_temporal_smoothness_coeff", 10.0)

        # auxiliary loss parameters
        self.gt_prior_loss = gt_prior_loss
        self.gt_prior_loss_coeff = self.params_dict.get("gt_prior_loss_coeff", 0.0)
        self.unnormalized_gaze_loss_coeff = self.params_dict.get("unnormalized_gaze_loss_coeff", 1e-5)
        self.common_predictor_map_loss_coeff = self.params_dict.get("common_predictor_map_loss_coeff", 1e-5)

    def _compute_common_loss(self, predicted_output, batch_input, batch_target):
        """
        Computes common loss terms and regularization costs

        Parameters:
        -----------
        predicted_output: dict
            Dictionary containing the outputs from MAADPredictor
        batch_input: dict
            Dictionary containing the tensor inputs to MAAD
        batch_target: dict
            Dictionary containing the network training target (ground truth)

        Returns:
        --------
        loss_tuple_and_stats: tuple
            ((individual cost terms....), stats)
        """
        # extract the gaze and awareness maps from the output dictionary
        normalized_gaze_map = predicted_output["gaze_density_map"]
        log_normalized_gaze_map = predicted_output["log_gaze_density_map"]
        awareness_map = predicted_output["awareness_map"]
        unnormalized_gaze_map = predicted_output["unnormalized_gaze"]
        common_predictor_map = predicted_output["common_predictor_map"]

        # initialize stats
        stats = {}

        # main gaze cost term
        negative_logprob = self._compute_nll(normalized_gaze_map, log_normalized_gaze_map, batch_input, batch_target)

        # aware at gaze points loss
        (
            awareness_at_gaze_points_loss,
            awareness_at_gaze_points_loss_pre_mult,
        ) = self._compute_awareness_at_gaze_points_loss(awareness_map, batch_input, batch_target)

        # spatial+temporal regularization for gaze map
        gaze_spatial_regularization, reg_stats = self.spatial_regularization(
            normalized_gaze_map, image=batch_input["road_image"],
        )
        gaze_temporal_regularization, treg_stats = self.temporal_regularization(
            normalized_gaze_map, image=batch_input["road_image"],
        )
        # mean over batch and time dimension
        gaze_spatial_reg_loss = self.gaze_spatial_regularization_coeff * torch.mean(gaze_spatial_regularization)
        gaze_temporal_reg_loss = self.gaze_temporal_regularization_coeff * torch.mean(gaze_temporal_regularization)

        # spatial+temporal regularization for awareness map
        awareness_spatial_regularization, reg_stats = self.spatial_regularization(
            awareness_map, image=batch_input["segmentation_mask_image"],
        )
        awareness_temporal_regularization, treg_stats = self.temporal_regularization(
            awareness_map, image=batch_input["segmentation_mask_image"],
        )
        awareness_spatial_reg_loss = self.awareness_spatial_regularization_coeff * torch.mean(
            awareness_spatial_regularization
        )
        awareness_temporal_reg_loss = self.awareness_temporal_regularization_coeff * torch.mean(
            awareness_temporal_regularization
        )

        # optic flow based temporal regularization
        optic_flow_awareness_temporal_smoothness = self._compute_optic_flow_based_temp_regularization(
            awareness_map, batch_input
        )

        # awareness decay loss
        aware_decay_loss = self.awareness_decay_coeff * torch.mean(
            (
                torch.mean(awareness_map[:, :-1, :, :, :], dim=(2, 3, 4)) * (1 - self.awareness_decay_alpha)
                - torch.mean(awareness_map[:, 1:, :, :, :], dim=(2, 3, 4))
            )
            ** 2
        )

        # steady state loss.
        # Total sum of each frame's awareness map should be fairly the same.
        awareness_map_sum0_tminus1 = torch.sum(awareness_map[:, :-1, :, :, :], dim=(2, 3, 4))  # (B, T-1)
        awareness_map_sum1_t = torch.sum(awareness_map[:, 1:, :, :, :], dim=(2, 3, 4))  # (B, T-1)

        awareness_map_sum_diff_along_t_sq = (
            awareness_map_sum0_tminus1 - awareness_map_sum1_t
        ) ** 2  # (B, T-1) # along the time dimension, the values represent the difference in the sum of awareness values for consecutive frames
        aware_steady_state_loss = self.awareness_steady_state_coeff * torch.mean(
            torch.mean(awareness_map_sum_diff_along_t_sq, dim=(1))
        )  # []

        # gaze transform prior loss. Unnormalized gaze loss, common predictor loss.
        # Regularizations to ensure that these modules don't blow up
        gaze_transform_prior_loss = self.gt_prior_loss_coeff * self.gt_prior_loss()
        unnormalized_gaze_loss = self.unnormalized_gaze_loss_coeff * (torch.mean(unnormalized_gaze_map)) ** 2
        common_predictor_map_loss = self.common_predictor_map_loss_coeff * torch.mean(common_predictor_map ** 2)

        loss_tuple_and_stats = (
            (
                negative_logprob,
                gaze_spatial_reg_loss,
                gaze_temporal_reg_loss,
                awareness_spatial_reg_loss,
                awareness_temporal_reg_loss,
                awareness_at_gaze_points_loss,
                awareness_at_gaze_points_loss_pre_mult,
                aware_steady_state_loss,
                aware_decay_loss,
                gaze_transform_prior_loss,
                unnormalized_gaze_loss,
                common_predictor_map_loss,
                optic_flow_awareness_temporal_smoothness,
            ),
            stats,
        )
        return loss_tuple_and_stats

    def _compute_nll(self, normalized_gaze_map, log_normalized_gaze_map, batch_input, batch_target):
        """
        Compute the negative log probability at each of the gaze points.

        Parameters:
        ----------
        normalized_gaze_map: torch.Tensor
            Gaze heatmap predicted by MAADNet
        log_normalized_gaze_map: torch.Tensor
            Logarithm of the gaze heatmap predicted by MAADNet
        batch_input: dict
            Dictionary containing the tensor inputs to MAAD
        batch_target: dict
            Dictionary containing the network training target (ground truth)

        Returns:
        --------
        negative_logprob: torch.Tensor 0-d
            Total negative log probability computed at the target gaze points that are valid.
        """
        logprob = normalized_gaze_map.new_zeros([1, 1])
        should_train_input_gaze = batch_input["should_train_input_gaze"]  # (B, T, L, 1)
        for b in range(log_normalized_gaze_map.shape[0]):  # batch dimension
            logprob_max = log_normalized_gaze_map[b, :, :, :, :].max().clone().detach()
            logprob_min = logprob_max - self.logprob_gap
            for t in range(log_normalized_gaze_map.shape[1]):  # time dimension
                log_gaze_map_bt = log_normalized_gaze_map[b, t, 0, :, :]  # grab the 2d heatmap
                lp = normalized_gaze_map.new_zeros([1, 1])  # accumulator of negative log probs
                # iterate through all the gaze points for the specific frame
                for l in range(batch_target.shape[2]):
                    x = max(0, min(batch_target[b, t, l, 0].int(), log_gaze_map_bt.shape[1] - 1))
                    y = max(0, min(batch_target[b, t, l, 1].int(), log_gaze_map_bt.shape[0] - 1))
                    # only consider valid gaze points for computing negative log prob
                    if should_train_input_gaze[b, t, l, 0]:
                        lp += max(log_gaze_map_bt[y, x], logprob_min)
                logprob += lp * self.gaze_data_coeff  # scale factor for gaze cost.

        negative_logprob = -logprob
        return negative_logprob

    def _compute_awareness_at_gaze_points_loss(self, awareness_map, batch_input, batch_target):
        """
        Compute awareness at the gaze points. Applies a gaussian filter around the gaze points to compute expected awareness estimates
        around the gaze points

        Parameters:
        ----------
        awareness_map: torch.Tensor
            Awareness heatmap predicted by MAADNet
        batch_input: dict
            Dictionary containing the tensor inputs to MAAD
        batch_target: dict
            Dictionary containing the network training target (ground truth)

        Returns:
        --------
        awareness_at_gaze_points_loss: torch.Tensor 0-d
            Total awareness computed at target gaze points (with gaussian kernel) that are valid. Post multiplication with coeff.
        awareness_at_gaze_points_loss_pre_mult: torch.Tensor 0-d
            Total awareness computed at target gaze points (with gaussian kernel) that are valid. Pre multiplication with coeff.
        """
        awareness_at_gaze_points_loss = awareness_map.new_zeros([1, 1])
        sig = 3  # std range for the gaussian kernel size
        should_train_input_gaze = batch_input["should_train_input_gaze"]
        N = self.gaussian_kernel_size
        for b in range(awareness_map.shape[0]):
            for t in range(awareness_map.shape[1]):
                awareness_map_bt = awareness_map[b, t, 0, :, :]
                a_at_g_loss = awareness_map.new_zeros([1, 1])
                for l in range(batch_target.shape[2]):
                    x = max(0, min(batch_target[b, t, l, 0].int(), awareness_map_bt.shape[1] - 1))
                    y = max(0, min(batch_target[b, t, l, 1].int(), awareness_map_bt.shape[0] - 1))
                    if should_train_input_gaze[b, t, l, 0]:
                        shift_x = 0
                        shift_y = 0
                        # make necessary shifts to the kernel based on where the gaze is
                        # check if gaze is close to left edge
                        shift_x = max(0, int(-(x - N)))
                        min_x = x - max(0, x - N)
                        max_x = 2 * N - min_x + 1
                        # if not near left edge, check if close to right edge
                        if shift_x == 0:
                            shift_x = min(0, -(int(x + N) - (awareness_map_bt.shape[1] - 1)))  # negative value
                            min_x = x - max(0, x - N) - shift_x
                            max_x = 2 * N - min_x + 1

                        # check if gaze is close to top edge
                        shift_y = max(0, int(-(y - N)))
                        min_y = y - max(0, y - N)
                        max_y = 2 * N - min_y + 1
                        # if not near top edge, check for near bottom edge
                        if shift_y == 0:
                            shift_y = min(0, -(int(y + N) - (awareness_map_bt.shape[0] - 1)))  # negative value
                            min_y = y - max(0, y - N) - shift_y
                            max_y = 2 * N - min_y + 1

                        kernel_x, kernel_y = np.meshgrid(
                            range(-N + shift_x, N + 1 + shift_x), range(-N + shift_y, N + 1 + shift_y)
                        )

                        # create the shifted gaussian kernel
                        kernel = np.exp(-(kernel_x ** 2 + kernel_y ** 2) / (sig ** 2))
                        kernel = awareness_map_bt.new_tensor(kernel / np.sum(kernel))
                        awareness_map_bt_patch = awareness_map_bt[y - min_y : y + max_y, x - min_x : x + max_x]
                        a_at_g_loss += torch.sum(
                            (
                                awareness_map_bt_patch
                                - (torch.ones_like(awareness_map_bt_patch) * kernel / torch.max(kernel))
                            )
                            ** 2
                        )
                awareness_at_gaze_points_loss += a_at_g_loss

        awareness_at_gaze_points_loss_pre_mult = awareness_at_gaze_points_loss
        awareness_at_gaze_points_loss = self.awareness_at_gaze_points_loss_coeff * awareness_at_gaze_points_loss
        return awareness_at_gaze_points_loss, awareness_at_gaze_points_loss_pre_mult

    def _compute_optic_flow_based_temp_regularization(self, awareness_map, batch_input):
        """
        Compute temporal regularization loss for awareness map using optic flow.

        Parameters:
        ----------
        awareness_map: torch.Tensor
            Awareness heatmap predicted by MAADNet
        batch_input: dict
            Dictionary containing the tensor inputs to MAAD

        Returns:
        -------
        optic_flow_awareness_temporal_smoothness: torch.Tensor
            Temporal smoothness cost computed on awareness map (post multiplying with coeff)
        """
        optic_flow_awareness_temporal_smoothness = awareness_map.new_zeros([1, 1])
        # scaled optic flow image (B, T, C=2, H, W), C=0 channel is ux and C=1 is uy, channel
        if self.add_optic_flow:
            optic_flow_image = batch_input["optic_flow_image"]

            # coordinate grid. xv is the row indices, yv is the column indices
            xv, yv = torch.meshgrid(
                [
                    awareness_map.new_tensor(torch.arange(0, awareness_map.shape[3]).clone().detach().numpy()),
                    awareness_map.new_tensor(torch.arange(0, awareness_map.shape[4]).clone().detach().numpy()),
                ]
            )
            # x_disp and y_disp indicate the delta to add from the displaced pixel computed from the flow
            # currently computing only at the displaced pixel, hence 0
            x_disp = [0]
            y_disp = [0]
            for b in range(awareness_map.shape[0]):
                for t in range(awareness_map.shape[1] - 1):  # 0, T-1
                    # h, w channel 0 contains displacement along the column dimension (index 1) (ux)
                    # and channel 1 contains displacement along the row dimension (index 0) (uy)
                    for (x_d, y_d) in list(itertools.product(x_disp, y_disp)):
                        optic_flow_image_t_ux_uy = optic_flow_image[b, t + 1, :, :, :]  # (2, h, w)
                        # (h, w) #awarness_map at t
                        awareness_map_t = awareness_map[b, t, 0, :, :]
                        # (h, w) awareness map at t + 1
                        awareness_map_tp1 = awareness_map[b, t + 1, 0, :, :]

                        # compute displaced pixels rounded from previous frame
                        xv_new_p = torch.floor(xv + torch.floor(optic_flow_image_t_ux_uy[1, :, :]) + x_d)
                        xv_new_p[xv_new_p < 0] = 0.0  # limit the index to stay within bounds
                        xv_new_p[xv_new_p >= awareness_map_t.shape[0]] = awareness_map_t.shape[0] - 1
                        yv_new_p = torch.floor(yv + torch.floor(optic_flow_image_t_ux_uy[0, :, :]) + y_d)
                        yv_new_p[yv_new_p < 0] = 0.0
                        yv_new_p[yv_new_p >= awareness_map_t.shape[1]] = awareness_map_t.shape[1] - 1

                        # grab the awareness values at time t corresponding to the displaced pixels.
                        # Flatten the x and y indices and then reshape it back to the original shape.
                        displaced_awareness_t_p = awareness_map_t[
                            xv_new_p.flatten().long(), yv_new_p.flatten().long()
                        ].reshape(awareness_map_t.shape[0], awareness_map_t.shape[1])

                        # (H, W)
                        eps_temporal_difference = (
                            awareness_map_tp1 - self.optic_flow_temporal_smoothness_decay * displaced_awareness_t_p
                        )
                        # separate the positive and negative change
                        positive_temporal_difference = eps_temporal_difference.clamp(min=0)  # (H, W)
                        negative_temporal_difference = -eps_temporal_difference.clamp(max=0)  # (H, W)

                        # penalize decrease more drastically, hence the squared.
                        negative_temporal_difference = (
                            negative_temporal_difference ** 2 * self.NEGATIVE_DIFFERENCE_COEFFICIENT
                        )
                        # positive change in awareness (increase) is penalized less, hence linear
                        positive_temporal_difference = (
                            positive_temporal_difference * self.POSITIVE_DIFFERENCE_COEFFICIENT
                        )
                        # (H, W)
                        temporal_difference_penalty = positive_temporal_difference + negative_temporal_difference
                        # weight based on displacement. Same for all pixels as the shift is the same
                        temporal_difference_penalty = np.exp(-np.linalg.norm((x_d, y_d))) * temporal_difference_penalty
                        # mean over the image dimensions (or number of pixels)
                        optic_flow_awareness_temporal_smoothness += torch.mean(temporal_difference_penalty)

            # average over B*(T-1) the number of outer loops
            optic_flow_awareness_temporal_smoothness = optic_flow_awareness_temporal_smoothness / (
                awareness_map.shape[0] * awareness_map.shape[1] - 1
            )
            optic_flow_awareness_temporal_smoothness = (
                self.optic_flow_temporal_smoothness_coeff * optic_flow_awareness_temporal_smoothness
            )

        return optic_flow_awareness_temporal_smoothness

    def _compute_consistency_smoothness(self, predicted_gaze_t, predicted_gaze_tp1):
        """
        Compute consistency loss between predicted two sequences offset by one hopsize (temporal_downsample_factor)

        Parameters:
        ----------
        predicted_gaze_t: dict
            Dictionary containing the output from MAADPredictor computed on gaze sequence at t
        predicted_gaze_tp1: dict
            Dictionary containing the outputs from MAADPredictor computed on gaze sequence at t+self.temporal_downsample_factor

        Returns:
        --------
        consistency_smoothness_awareness: torch.Tensor
            Consistency smoothness cost computed on awareness map (post multiplying with coeff)
        consistency_smoothness_gaze: torch.Tensor
            Consistency smoothness cost computed on gaze map (post multiplying with coeff)
        """
        normalized_gaze_map_t = predicted_gaze_t["gaze_density_map"]  # (B, T, 1, H, W)
        awareness_map_t = predicted_gaze_t["awareness_map"]  # (B, T, 1, H, W)

        normalized_gaze_map_tp1 = predicted_gaze_tp1["gaze_density_map"]  # (B, T, 1, H, W)
        awareness_map_tp1 = predicted_gaze_tp1["awareness_map"]  # (B, T, 1, H, W)
        T = normalized_gaze_map_t.shape[1]  # number of time steps

        consistency_smoothness_awareness = awareness_map_t.new_zeros([1, 1])
        consistency_smoothness_gaze = normalized_gaze_map_t.new_zeros([1, 1])

        # only consider the time stamps towrads the end of the slice so that sufficient temporal history is captured.
        # (B, T/2, C, H, W)
        awareness_diff_at_later_common_frame_idxs = (
            awareness_map_t[:, round(T / 2) :, :, :, :] - awareness_map_tp1[:, (round(T / 2) - 1) : -1, :, :, :]
        )
        # (B, T/2, C, H, W)
        normalized_gaze_diff_at_later_common_frame_idxs = (
            normalized_gaze_map_t[:, round(T / 2) :, :, :, :]
            - normalized_gaze_map_tp1[:, (round(T / 2) - 1) : -1, :, :, :]
        )
        consistency_smoothness_awareness = self.consistency_coeff_awareness * torch.mean(
            torch.mean(torch.sum((awareness_diff_at_later_common_frame_idxs ** 2), dim=(2, 3, 4)), dim=(1))
        )
        consistency_smoothness_gaze = self.consistency_coeff_gaze * torch.mean(
            torch.mean(torch.sum((normalized_gaze_diff_at_later_common_frame_idxs ** 2), dim=(2, 3, 4)), dim=(1))
        )

        return consistency_smoothness_awareness, consistency_smoothness_gaze

    def loss(
        self,
        predicted_gaze_output,
        gaze_batch_input,
        gaze_batch_target,
        predicted_awareness_output,
        awareness_batch_input,
        awareness_batch_target,
        awareness_batch_annotation_data,
        predicted_pairwise_gaze_t,
        pairwise_gaze_batch_input_t,
        pairwise_gaze_batch_target_t,
        predicted_pairwise_gaze_tp1,
        pairwise_gaze_batch_input_tp1,
        pairwise_gaze_batch_target_tp1,
    ):
        """
        Computes the total network loss for MAAD

        Parameters:
        ---------
        In the following * is in [gaze, awareness, pairwise_gaze]. Pairwise gaze is also for time t and t+1 (tp1), appended to the variable name.
        predicted_*_output: dict
            dict containing the output heatmaps
        *_batch_input: dict
            dict containing the input to the network
        *_batch_target: dict
            dict containing the network target ground truth gaze points

        Returns:
        -------
        loss: torch.Tensor
            Total network loss
        stats: dict
            Dictionary containing the individual loss terms and other stats
        """
        # compute loss terms on the gaze_batch
        gaze_loss_tuple, stats = self._compute_common_loss(predicted_gaze_output, gaze_batch_input, gaze_batch_target)
        (
            gaze_ds_negative_logprob,
            gaze_ds_gaze_spatial_reg_loss,
            gaze_ds_gaze_temporal_reg_loss,
            gaze_ds_awareness_spatial_reg_loss,
            gaze_ds_awareness_temporal_reg_loss,
            gaze_ds_awareness_at_gaze_points_loss,
            gaze_ds_awareness_at_gaze_points_loss_pre_mult,
            gaze_ds_aware_steady_state_loss,
            gaze_ds_aware_decay_loss,
            gaze_ds_gaze_transform_prior_loss,
            gaze_ds_unnormalized_gaze_loss,
            gaze_ds_common_predictor_map_loss,
            gaze_ds_optic_flow_awareness_temporal_smoothness,
        ) = gaze_loss_tuple

        # compute loss terms on the awareness_batch
        awareness_loss_tuple, awareness_stats = self._compute_common_loss(
            predicted_awareness_output, awareness_batch_input, awareness_batch_target,
        )
        (
            awareness_ds_negative_logprob,
            awareness_ds_gaze_spatial_reg_loss,
            awareness_ds_gaze_temporal_reg_loss,
            awareness_ds_awareness_spatial_reg_loss,
            awareness_ds_awareness_temporal_reg_loss,
            awareness_ds_awareness_at_gaze_points_loss,
            awareness_ds_awareness_at_gaze_points_loss_pre_mult,
            awareness_ds_aware_steady_state_loss,
            awareness_ds_aware_decay_loss,
            awareness_ds_gaze_transform_prior_loss,
            awareness_ds_unnormalized_gaze_loss,
            awareness_ds_common_predictor_map_loss,
            awareness_ds_optic_flow_awareness_temporal_smoothness,
        ) = awareness_loss_tuple

        # add cost computed on the awareness data batch to the cost computed on the gaze data batch.
        gaze_ds_negative_logprob += awareness_ds_negative_logprob
        gaze_ds_gaze_spatial_reg_loss += awareness_ds_gaze_spatial_reg_loss
        gaze_ds_gaze_temporal_reg_loss += awareness_ds_gaze_temporal_reg_loss
        gaze_ds_awareness_spatial_reg_loss += awareness_ds_awareness_spatial_reg_loss
        gaze_ds_awareness_temporal_reg_loss += awareness_ds_awareness_temporal_reg_loss
        gaze_ds_awareness_at_gaze_points_loss += awareness_ds_awareness_at_gaze_points_loss
        gaze_ds_awareness_at_gaze_points_loss_pre_mult += awareness_ds_awareness_at_gaze_points_loss_pre_mult
        gaze_ds_aware_steady_state_loss += awareness_ds_aware_steady_state_loss
        gaze_ds_aware_decay_loss += awareness_ds_aware_decay_loss
        gaze_ds_gaze_transform_prior_loss += awareness_ds_gaze_transform_prior_loss
        gaze_ds_unnormalized_gaze_loss += awareness_ds_unnormalized_gaze_loss
        gaze_ds_common_predictor_map_loss += awareness_ds_common_predictor_map_loss
        gaze_ds_optic_flow_awareness_temporal_smoothness += awareness_ds_optic_flow_awareness_temporal_smoothness

        # attended awareness annotation label loss
        if predicted_awareness_output is not None and awareness_batch_annotation_data is not None:
            awareness_loss_pre_mult, awareness_loss_stats = self.awareness_label_loss.loss(
                predicted_awareness_output, awareness_batch_annotation_data
            )
            awareness_label_loss = self.awareness_label_coeff * awareness_loss_pre_mult
        else:
            awareness_label_loss = gaze_ds_aware_steady_state_loss.new_tensor(0.0)
            awareness_loss_pre_mult = gaze_ds_aware_steady_state_loss.new_tensor(0.0)
            awareness_loss_stats = {
                "awareness_mse": 0,
                "awareness_l1_loss": 0,
                "awareness_per_pixel_l1_loss": 0,
                "awareness_img_max": 0,
                "awareness_label_mean": 0,
                "awareness_predicted_label_mean": 0,
            }

        # temporal consistency loss
        if predicted_pairwise_gaze_t is not None and predicted_pairwise_gaze_tp1 is not None:
            (consistency_smoothness_awareness, consistency_smoothness_gaze,) = self._compute_consistency_smoothness(
                predicted_pairwise_gaze_t, predicted_pairwise_gaze_tp1
            )
        else:
            consistency_smoothness_gaze = gaze_ds_aware_steady_state_loss.new_tensor(0.0)
            consistency_smoothness_awareness = gaze_ds_aware_steady_state_loss.new_tensor(0.0)

        # store stats
        stats["negative_logprob"] = gaze_ds_negative_logprob
        stats["gaze_s_regularization"] = gaze_ds_gaze_spatial_reg_loss
        stats["gaze_t_regularization"] = gaze_ds_gaze_temporal_reg_loss
        stats["gaze_consistency_smoothness"] = consistency_smoothness_gaze

        stats["awareness_s_regularization"] = gaze_ds_awareness_spatial_reg_loss
        stats["awareness_t_regularization"] = gaze_ds_awareness_temporal_reg_loss

        stats["awareness_optic_flow_t_smoothness"] = gaze_ds_optic_flow_awareness_temporal_smoothness
        stats["awareness_consistency_smoothness"] = consistency_smoothness_awareness

        stats["awareness_at_gaze_points_loss"] = gaze_ds_awareness_at_gaze_points_loss
        stats["awareness_at_gaze_points_loss_pre_mult"] = gaze_ds_awareness_at_gaze_points_loss_pre_mult

        stats["aware_steady_state_loss"] = gaze_ds_aware_steady_state_loss
        stats["aware_decay_loss"] = gaze_ds_aware_decay_loss
        stats["awareness_label_loss"] = awareness_label_loss
        stats["awareness_label_loss_pre_coeff_mult"] = awareness_loss_pre_mult

        stats["awareness_mse"] = awareness_loss_stats["awareness_mse"]
        stats["awareness_label_mean"] = awareness_loss_stats["awareness_label_mean"]
        stats["awareness_predicted_label_mean"] = awareness_loss_stats["awareness_predicted_label_mean"]
        stats["awareness_l1_loss"] = awareness_loss_stats["awareness_l1_loss"]
        stats["awareness_per_pixel_l1_loss"] = awareness_loss_stats["awareness_per_pixel_l1_loss"]
        stats["awareness_img_max"] = awareness_loss_stats["awareness_img_max"]

        stats["unnormalized_gaze_loss"] = gaze_ds_unnormalized_gaze_loss
        stats["common_predictor_map_loss"] = gaze_ds_common_predictor_map_loss
        stats["gaze_transform_prior_loss"] = gaze_ds_gaze_transform_prior_loss

        # total loss
        loss = (
            gaze_ds_negative_logprob
            + gaze_ds_gaze_spatial_reg_loss
            + gaze_ds_gaze_temporal_reg_loss
            + gaze_ds_awareness_spatial_reg_loss
            + gaze_ds_awareness_temporal_reg_loss
            + gaze_ds_awareness_at_gaze_points_loss
            + gaze_ds_aware_steady_state_loss
            + gaze_ds_aware_decay_loss
            + consistency_smoothness_gaze
            + consistency_smoothness_awareness
            + gaze_ds_optic_flow_awareness_temporal_smoothness
            + gaze_ds_gaze_transform_prior_loss
            + gaze_ds_unnormalized_gaze_loss
            + gaze_ds_common_predictor_map_loss
            + awareness_label_loss
        )

        return loss, stats

    def to(self, device):
        """
        Moves the regularization variables onto the device
        """
        self.spatial_regularization.to(device)
        self.temporal_regularization.to(device)
