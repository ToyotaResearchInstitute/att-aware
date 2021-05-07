# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch
import numpy as np
import collections


def create_chm_predictor(chm_predictor_input_dim_dict, predictor_output_size):
    return CHMPredictorNet(chm_predictor_input_dim_dict, predictor_output_size)


class CHMPredictorNet(torch.nn.Module):
    def __init__(self, chm_predictor_input_dim_dict, predictor_output_size):
        """
        Parameters:
        ----------
        chm_predictor_input_dim_dict: dict
            Dict containing the output resolution of each of the enc-dec backbone in the FusionNet
        predictor_output_size: [int, int, int]
            Size of predictor output = [num_latent_layers, H, W]
        """
        super().__init__()
        # sum up the number of channels for output for each of enc-dec backbone
        self.predictor_input_num_features = np.sum([x[0] for x in chm_predictor_input_dim_dict.values()])

        self.chm_predictor_input_dim_dict = chm_predictor_input_dim_dict
        self.predictor_output_size = predictor_output_size
        self.predictor_output_num_features = self.predictor_output_size[0]

        # kernel size used for conv2d
        kernel_size = 5
        self.common_predictor = torch.nn.Conv2d(
            self.predictor_input_num_features,
            self.predictor_output_num_features,  # = num_latent_layers argument
            kernel_size,
            padding=kernel_size // 2,
        )
        # gaze_predictor_output_feature = 1
        self.gaze_predictor = torch.nn.Conv1d(self.predictor_output_num_features, 1, [1], padding=0)
        # awareness_predictor_output_features = 1
        self.awareness_predictor = torch.nn.Conv1d(self.predictor_output_num_features, 1, [1], padding=0)
        # dim=3 works because before applying the softmax we flatten the 2d heatmap. The softmax makes sure the probability sums to one over the entire heatmap
        self.softmax = torch.nn.Softmax(dim=3)

    def forward(self, predictor_input):
        """
        Parameters:
        ----------
        predictor_input: torch.Tensor
            Output of the Fusion Net which will be fed into the CHMPredictorNet

        Returns:
        --------
        gaze_awareness_maps_output: OrderedDict
            Containing the gaze density map, log_gaze_density_map, awareness estimate, unnormalized_gaze, common predictor output
        """
        # initialize a zero tensor to hold the output of the common predictor
        common_predictor_output = predictor_input.new_zeros(
            predictor_input.shape[0],  # batch
            predictor_input.shape[1],  # time
            # only change the number of channels to match output dimensions.
            self.predictor_output_num_features,
            predictor_input.shape[3],  # height
            predictor_input.shape[4],  # width
        )

        for t in range(predictor_input.shape[1]):
            # run predictor over each image in sequence
            common_predictor_output[:, t, :, :, :] = self.common_predictor(predictor_input[:, t, :, :, :])

        # common_predictor_output is of shape (B, T, C=num_latent_layers, H, W)

        # Gaze estimate
        # reshaped_common ---> gaze_estimate ---> softmax = gaze_heatmap
        reshaped_common = common_predictor_output.permute(0, 2, 1, 3, 4)  # (B, C=num_latent_layers, T, H, W)
        # (B, C=num_latent_layers, T*H*W)
        flattened_reshaped_common = reshaped_common.contiguous().view(
            reshaped_common.shape[0], reshaped_common.shape[1], -1
        )
        # reshape in order to allow softmax to run (only used on the gaze estimate for now)
        # (B, T, C=1, H, W)
        gaze_estimate = (
            self.gaze_predictor(flattened_reshaped_common)
            .view(reshaped_common.shape[0], 1, reshaped_common.shape[2], reshaped_common.shape[3], -1)
            .permute(0, 2, 1, 3, 4)
        )
        # (B, T, C=1, H, W). Probability distribution
        gaze_density_map = self.softmax(
            gaze_estimate.contiguous().view(gaze_estimate.shape[0], gaze_estimate.shape[1], 1, -1)
        ).view_as(gaze_estimate)

        # Awareness Map Estimation
        # reshape_common ---> awareness_map
        # (B, T, C=1, H, W)
        awareness_map = torch.nn.Sigmoid()(
            self.awareness_predictor(flattened_reshaped_common)
            .view(reshaped_common.shape[0], 1, reshaped_common.shape[2], reshaped_common.shape[3], -1)
            .permute(0, 2, 1, 3, 4)
        )

        gaze_awareness_maps_output = collections.OrderedDict()
        gaze_awareness_maps_output["gaze_density_map"] = gaze_density_map
        gaze_awareness_maps_output["log_gaze_density_map"] = torch.log(gaze_density_map)
        gaze_awareness_maps_output["awareness_map"] = awareness_map
        gaze_awareness_maps_output["unnormalized_gaze"] = gaze_estimate
        gaze_awareness_maps_output["common_predictor_map"] = reshaped_common

        if torch.isnan(gaze_density_map).sum():
            import IPython

            print(gaze_density_map)
            IPython.embed(header="predictor:forward.  gaze_density_map has nans")

        return gaze_awareness_maps_output
