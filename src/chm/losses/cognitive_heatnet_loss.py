import torch

from chm.losses.regularizations import EPSpatialRegularization, EPTemporalRegularization
from chm.losses.awareness_label_loss import AwarenessPointwiseLabelLoss


class CognitiveHeatNetLoss(object):
    def __init__(self, params_dict, gt_prior_loss=None):

        self.params_dict = params_dict

        self.aspect_ratio_reduction_factor = self.params_dict.get("aspect_ratio_reduction_factor", 8)
        self.ORIG_ROAD_IMG_DIMS = self.params_dict.get("orig_road_image_dims")
        self.image_width = int(round(self.ORIG_ROAD_IMAGE_WIDTH / self.aspect_ratio_reduction_factor))
        self.image_height = int(round(self.ORIG_ROAD_IMAGE_HEIGHT / self.aspect_ratio_reduction_factor))
        self.add_optic_flow = self.params_dict.get("add_optic_flow", False)
        regularization_eps = self.params_dict.get("regularization_eps", 1e-3)
        sig_scale_factor = self.params_dict.get("sig_scale_factor", 1)

        # spatial and temporal regularization
        self.spatial_regularization = EPSpatialRegularization(
            image_width=image_width,
            image_height=image_height,
            eps=regularization_eps,
            sig_scale_factor=sig_scale_factor,
        )
        self.temporal_regularization = EPTemporalRegularization(
            image_width=image_width,
            image_height=image_height,
            eps=regularization_eps,
            sig_scale_factor=sig_scale_factor,
        )

        # Cost coeffs and parameters

        # main cost term coeffs
        self.gaze_data_coeff = self.params_dict.get("gaze_data_coeff", 1.0)
        self.awareness_at_gaze_points_loss_coeff = self.params_dict.get("awareness_at_gaze_points_loss_coeff", 1)

        # temporal and spatial regularizations
        self.gaze_spatial_regularization_coeff = self.params_dict.get("gaze_spatial_regularization_coeff", 1.0)
        self.gaze_temporal_regularization_coeff = self.params_dict.get("gaze_temporal_regularization_coeff", 1.0)
        self.awareness_spatial_regularization_coeff = self.params_dict.get(
            "awareness_spatial_regularization_coeff", 1.0
        )
        self.awareness_temporal_regularization_coeff = self.params_dict.get(
            "awareness_temporal_regularization_coeff", 1.0
        )
        self.awareness_decay_coeff = self.params_dict.get("awareness_decay_coeff", 1.0)
        self.awareness_decay_alpha = self.params_dict.get("awareness_decay_alpha", 1.0)

        self.consistency_coeff_gaze = self.params_dict.get("consistency_coeff_gaze", 10)
        self.consistency_coeff_awareness = self.params_dict.get("consistency_coeff_awareness", 10)

        self.NEGATIVE_DIFFERENCE_COEFFICIENT = self.params_dict.get("negative_difference_coeff", 10.0)
        self.POSITIVE_DIFFERENCE_COEFFICIENT = self.params_dict.get("positive_difference_coeff", 1.0)

        self.awareness_loss_type = self.params_dict.get("awareness_loss_type", "huber_loss")
        self.awareness_label_loss_patch_half_size = self.params_dict.get("awareness_label_loss_patch_half_size", 4)

        self.awareness_label_loss = AwarenessPointwiseLabelLoss(type=self.awareness_loss_type)
        self.awareness_label_coeff = self.params_dict.get("awareness_label_coeff", 1.0)

        self.optic_flow_temporal_smoothness_decay = self.params_dict.get("optic_flow_temporal_smoothness_decay", 1.0)
        self.optic_flow_temporal_smoothness_coeff = self.params_dict.get("optic_flow_temporal_smoothness_coeff", 10.0)

        self.logprob_gap = self.params_dict.get("logprob_gap", 10)

        self.gt_prior_loss = gt_prior_loss
        self.gt_prior_loss_coeff = self.params_dict.get("gt_prior_loss_coeff", 1.0)
        self.unnormalized_gaze_loss_coeff = self.params_dict.get("unnormalized_gaze_loss_coeff", 1e-5)
        self.common_predictor_map_loss_coeff = self.params_dict.get("common_predictor_map_loss_coeff", 1e-5)

    def _compute_consistency_smoothness(self):
        pass

    def _compute_common_loss(self, predicted_output, batch_input, batch_target):
        # extract the gaze and awareness maps from the output dictionary
        normalized_gaze_map = predicted_output["gaze_density_map"]
        log_normalized_gaze_map = predicted_output["log_gaze_density_map"]
        awareness_map = predicted_output["awareness_map"]
        unnormalized_gaze_map = predicted_output["unnormalized_gaze"]
        common_predictor_map = predicted_output["common_predictor_map"]

        # initialize stats
        stats = {}

        # main gaze cost term
        negative_logprob = self.compute_nll(normalized_gaze_map, log_normalized_gaze_map, batch_input, batch_target)

        # aware at gaze points
        (
            awareness_at_gaze_points_loss,
            awareness_at_gaze_points_loss_pre_mult,
        ) = self.compute_awareness_at_gaze_points_loss(awareness_map, batch_input, batch_target)

        # spatial+temporal regularization for gaze map
        gaze_spatial_regularization, reg_stats = self.spatial_regularization(
            normalized_gaze_map,
            image=batch_input["road_image"],
        )
        gaze_temporal_regularization, treg_stats = self.temporal_regularization(
            normalized_gaze_map,
            image=batch_input["road_image"],
        )
        # mean over batch and time dimension
        gaze_spatial_reg_loss = self.gaze_spatial_regularization_coeff * torch.mean(gaze_spatial_regularization)
        gaze_temporal_reg_loss = self.gaze_temporal_regularization_coeff * torch.mean(gaze_temporal_regularization)

        # spatial+temporal regularization for awareness map
        awareness_spatial_regularization, reg_stats = self.spatial_regularization(
            awareness_map,
            image=batch_input["segmentation_mask_image"],
        )
        awareness_temporal_regularization, treg_stats = self.temporal_regularization(
            awareness_map,
            image=batch_input["segmentation_mask_image"],
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
        unnormalized_gaze_loss = self.unnormalized_gaze_loss_coeff * (torch.mean(unnormalized_gaze)) ** 2
        common_predictor_map_loss = self.common_predictor_map_loss_coeff * torch.mean(common_predictor_map ** 2)

        return (
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

    def _compute_nll(self):
        pass

    def _compute_awareness_at_gaze_points_loss(self):
        pass

    def _compute_optic_flow_based_temp_regularization(self):
        pass

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
        Computes the total network loss for CHM
        Parameters:
        ---------
        In the following * is in [gaze, awareness, pairwise_gaze]. Pairwise gaze is also for time t and t+1 (tp1)
        predicted_*_output: dict
            dict containing the output heatmaps
        *_batch_input: dict
            dict containing the input to the network
        *_batch_target: dict
            dict containing the network target ground truth gaze points

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
            predicted_awareness_output,
            awareness_batch_input,
            awareness_batch_target,
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

        # Label Loss
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

        # Consistency Loss
        if predicted_pairwise_gaze_t is not None and predicted_pairwise_gaze_tp1 is not None:
            (
                consistency_smoothness_awareness,
                consistency_smoothness_gaze,
            ) = self._compute_consistency_smoothness(predicted_pairwise_gaze_t, predicted_pairwise_gaze_tp1)
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
            negative_logprob
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
        self.spatial_regularization.to(device)
        self.temporal_regularization.to(device)