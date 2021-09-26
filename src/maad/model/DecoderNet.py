# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch

from maad.model.DecoderUnit import DecoderUnit


def create_decoder(decoder_net_params=None):
    """
    Creates the encoder structure. Two possible encoders.
    1. With only ResNet18 encoder layers. 4 layers
    2. When use_s3d is True, then 3 layers of S3D will be stacked on top of two layers of ResNet18. Total of 5

    Parameters:
    ----------
    decoder_net_params: dict
        Dict containing various parameters (use_s3d, decoder_layer_features, output_dim_of_decoder, side_channel_input_dim, skip_layers) for the decoder net

    Returns:
    -------
    DecoderNet: torch.nn.Module
        Decoder network for MAAD
    """
    assert decoder_net_params is not None

    use_s3d = decoder_net_params["use_s3d"]
    decoder_layer_features = decoder_net_params["decoder_layer_features"]
    output_dim_of_decoder = decoder_net_params["output_dim_of_decoder"]
    side_channel_input_dim = decoder_net_params["side_channel_input_dim"]
    skip_layers = decoder_net_params["skip_layers"]

    decoder_layers = torch.nn.ModuleDict()

    # note that the layer names go backwards. Because the decoder layers mirror the encoder layers
    if not use_s3d:
        decoder_layers["layer4"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-1],  # 128
            output_dim=decoder_layer_features[-2],  # 64
            skip_dim=skip_layers[-1],  # 0
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )  # innermost DU. skip_layers[-1] = 0
        decoder_layers["layer3"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-2],  # 64
            output_dim=decoder_layer_features[-3],  # 32
            skip_dim=skip_layers[-2],  # 256
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
        decoder_layers["layer2"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-3],  # 32
            output_dim=decoder_layer_features[-4],  # 16
            skip_dim=skip_layers[-3],  # 32
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
        decoder_layers["layer1"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-4],  # 16
            output_dim=output_dim_of_decoder,  # 16
            skip_dim=skip_layers[-4],  # 16
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
    else:
        decoder_layers["s3d_net_3"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-1],  # 256 #
            output_dim=decoder_layer_features[-2],  # 128
            skip_dim=skip_layers[-1],  # 0, no skip connections for S3D
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
        decoder_layers["s3d_net_2"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-2],  # 128
            output_dim=decoder_layer_features[-3],  # 64
            skip_dim=skip_layers[-2],  # 0
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
        decoder_layers["s3d_net_1"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-3],  # 64
            output_dim=decoder_layer_features[-4],  # 32
            skip_dim=skip_layers[-3],  # 0
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
        decoder_layers["layer2"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-4],  # 32
            output_dim=decoder_layer_features[-5],  # 16
            skip_dim=skip_layers[-4],  # 32
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
        decoder_layers["layer1"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-5],  # 16
            output_dim=output_dim_of_decoder,  # 16
            skip_dim=skip_layers[-5],  # 16
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )

    # keys indicating the source of skip connections from the encoder.
    if not use_s3d:
        skip_layers_keys = ["layer3", "layer2", "layer1"]
    else:
        # if s3d units are being used, the only two layers have skip connections.
        skip_layers_keys = ["layer2", "layer1"]

    return DecoderNet(decoder_layers=decoder_layers, skip_layers_keys=skip_layers_keys,)


class DecoderNet(torch.nn.Module):
    """
    Class that encapsulates multiple DecoderUnits to form the decoder of MAAD
    """

    def __init__(self, decoder_layers, skip_layers_keys):
        """
        Parameters:
        ----------
        decoder_layers: dict
            Dict containing the DecoderUnits used for the decoder

        skip_layer_keys: list
            List of encoder layers' names that will have skip connections to the decoder.

        Returns:
        -------
        None
        """
        super().__init__()
        self.decoder_layers = decoder_layers  # these are the DecoderUnits.
        self.skip_layers_keys = skip_layers_keys
        self.upsm = torch.nn.Upsample(mode="bilinear", align_corners=False)

    def forward(self, encoder_output_dict, side_channel_input, enc_input_shape):
        """
        Decode an output image via Conv3d (or s3D) + upsampling + side channel information model

        Parameters:
        ----------
        encoder_output_dict: dict
            Dict containing all the intermediate outputs from the encoder layer
        side_channel_input: dict
            Dict containing the side channel input. Depending on the key, different processing is applied.
            For example, gaze points are transformed into voronoi maps
        enc_input_shape: torch.Tensor
            Tensor indicating the shape of the encoder input. Needed for making sure the upsampled output of the final decoder layer matches the original image dimension

        Returns:
        --------
        decoder_net_output: torch.Tensor
            Output of the Decoder Net after processing through all the DecoderUnit layers
        """
        du_keys = list(self.decoder_layers.keys())
        upsm_target_size_list = [[encoder_output_dict[x].shape[3], encoder_output_dict[x].shape[4]] for x in du_keys]
        upsm_target_size_list.append([int(round(enc_input_shape[3] / 2)), int(round(enc_input_shape[4] / 2))])

        # layer4, layer3, layer2, layer1 or
        # s3d_net_3, s3d_net_2, s3d_net_1, layer2, layer1 in that order.
        # so [0] is layer4 or s3d_net_3
        du_key = list(self.decoder_layers)[0]
        # initialize the previous decoder unit output variable as the last encoder layer output. For subsequent decoder layers, this will be the output of the previous decoder unit.
        # encoder output dict will other wise be used for skip connections
        previous_du_out = encoder_output_dict[du_key]
        du_outs = []
        for i, du_key in enumerate(self.decoder_layers):
            individual_sc_input = None  # individual side channel input. From gaze, optic flow etc.,
            individual_sc_inputs = []  # list to collate all the individual side channel inputs.
            # (H,W) are the resolution for the current layer
            H = encoder_output_dict[du_key].shape[-2]
            W = encoder_output_dict[du_key].shape[-1]
            if side_channel_input is not None:
                for side_key in side_channel_input:
                    if side_key == "driver_facing":
                        # CREATE VORONOI MAPS FOR SIDE CHANNEL GAZE INPUT
                        # side_channel_input is a tensor of sahpe (B, T, L, 4) when the driver_facing module only has linear0 module.
                        # (B, T, L, 1)
                        dropout_indicator = side_channel_input[side_key][:, :, :, 3:]
                        # (B, T, L, 1)
                        should_train_input_gaze_indicator = side_channel_input[side_key][:, :, :, 2:3]

                        # (B, T, 1) Since the dropout is applied for each item in the batch, the value at 0th position is the same
                        dropout_indicator = dropout_indicator[:, :, 0, :]
                        # (B,T, L, 2). Grab only the gaze points and ignore the dropout indicator
                        gaze_points = side_channel_input[side_key][:, :, :, 0:2]

                        # create meshgrid containing normalized coordinates for computing dx and dy at each pixel to the nearest gaze point
                        yv, xv = torch.meshgrid(
                            [
                                torch.linspace(
                                    start=0.0,
                                    end=1.0,
                                    steps=encoder_output_dict[du_key].shape[3],
                                    device=gaze_points.device,
                                ),
                                torch.linspace(
                                    start=0.0,
                                    end=1.0,
                                    steps=encoder_output_dict[du_key].shape[4],
                                    device=gaze_points.device,
                                ),
                            ]
                        )  # normalized coordinates along each dimension
                        yv = yv.unsqueeze(dim=2)  # ( H, W, 1)
                        xv = xv.unsqueeze(dim=2)  # (H, W, 1)
                        coord_grid = torch.cat([xv, yv], dim=2)  # (H, W, 2)

                        # create a tensor to hold the dxdy to nearest gaze point side channel information. Consists of two channels. one for dx and one for dy.
                        # gaze_side_channel_layer = torch.zeros(gaze_points.shape[0], gaze_points.shape[1], coord_grid.shape[0], coord_grid.shape[1], 2,
                        # device=gaze_points.device) #(B,T,H,W,2), C=2 To be permuted later into B, T, C, H, W where C = 2.
                        # first channel contains dx information and second channel contains dy information
                        gp_list_grid = (
                            gaze_points.unsqueeze(2)
                            .unsqueeze(2)
                            .repeat(1, 1, coord_grid.shape[0], coord_grid.shape[1], 1, 1)
                        )  # (B,T,H,W,L,2)
                        coord_grid_list = (
                            coord_grid.unsqueeze(2)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .repeat(gaze_points.shape[0], gaze_points.shape[1], 1, 1, gaze_points.shape[2], 1)
                        )  # (B, T, H, W, L, 2)

                        dx_dy_grid_gaze_list = coord_grid_list - gp_list_grid  # (B, T, H, W, L, 2)
                        # (B, T, H, W, L) #compute distances to all the L gaze points.
                        gp_list_distance_grid = torch.norm(dx_dy_grid_gaze_list, dim=-1)

                        # (B, T, H, W) #compute the index of gaze point which is of shortest distance to a particular coordinate point.
                        # since the "invalid" gaze points were kept at -10,-10, these points will never be the "smallest" distance gaze point,
                        # if there are other valid gaze points in the list.In the case all L points are invalid,
                        # then the multiplication (later) on with the train bit will take care of zeroing out the entire frame
                        (min_distances, min_indices) = torch.min(gp_list_distance_grid, dim=-1)

                        # (B, T, H, W, 1) #this is the index corresponding to which one of the L'th gaze point is closest to a particular coordinate
                        min_indices = min_indices.unsqueeze(-1)

                        # (B, T, H, W, L). Initialize the one-hot tensor with zeros
                        # the one hot vector for each coordinate position will be L dimensional.
                        # this is picking the proper gaze index for dx dy coordinate corresponding to the shortest distance gaze point
                        min_indices_one_hot = torch.zeros_like(dx_dy_grid_gaze_list[:, :, :, :, :, 0])

                        # (B, T, H, W, L) #scatter 1 at the specified indices along the L dimension of min_indices_one_hot
                        min_indices_one_hot.scatter_(4, min_indices, 1)

                        # (B, T, H, W, L, 2). Make one hot tensor same size as dx_dy_grid_gaze_list
                        # so that both dx and dy can be selected using the same index
                        min_indices_one_hot = min_indices_one_hot.unsqueeze(-1).repeat(
                            1, 1, 1, 1, 1, dx_dy_grid_gaze_list.shape[-1]
                        )

                        # (B, T, H, W, L, 2) Note that only one of L gaze points will be nonzero. Therefore, go ahead and sum up along L dimension
                        gaze_side_channel_layer = dx_dy_grid_gaze_list * min_indices_one_hot

                        # (B, T, H, W, 2) #dx,dy values to the nearest gaze point.
                        gaze_side_channel_layer = torch.sum(gaze_side_channel_layer, 4)
                        individual_sc_input = gaze_side_channel_layer.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W) C=2

                        # (B, T, 1, H, W) #add dimension at the end for H and W.
                        dropout_indicator = (
                            dropout_indicator.unsqueeze(-1)
                            .unsqueeze(-1)
                            .repeat(1, 1, 1, individual_sc_input.shape[3], individual_sc_input.shape[4])
                        )

                        # (B, T, H, W, L, 1)
                        min_indices_one_hot_single_dim = min_indices_one_hot[:, :, :, :, :, 1:]
                        # (B, T, H, W, L, 1)
                        should_train_input_gaze_indicator = (
                            should_train_input_gaze_indicator.unsqueeze(2)
                            .unsqueeze(2)
                            .repeat(
                                1,
                                1,
                                min_indices_one_hot_single_dim.shape[2],
                                min_indices_one_hot_single_dim.shape[3],
                                1,
                                1,
                            )
                        )

                        # (B, T, H, W, 1)
                        should_train_input_gaze_indicator = torch.sum(
                            should_train_input_gaze_indicator * min_indices_one_hot_single_dim, 4
                        )

                        # (B, T, 1, H, W)
                        should_train_input_gaze_indicator = should_train_input_gaze_indicator.permute(0, 1, 4, 2, 3)

                        # (B,T,2,H,W)
                        st_mult = should_train_input_gaze_indicator.repeat(1, 1, individual_sc_input.shape[2], 1, 1)

                        # (B, T, C, H, W) with C=2, same as individual_sc_input's channel dimension
                        di_mult = dropout_indicator.repeat(1, 1, individual_sc_input.shape[2], 1, 1)

                        # this multiplication with the di_mult will make sure that the dxdy layers that are dropped out are all zero'ed out.
                        # (B, T, C, H, W) C=2 #only pass through when should train and dropout indicator are true.
                        individual_sc_input = individual_sc_input * di_mult * st_mult

                        # (B, T, C=1, H, W)
                        min_distances = min_distances.unsqueeze(2)

                        # (B, T,1,H,W)
                        min_dx_layer_sq = ((gaze_side_channel_layer[:, :, :, :, 0:1]) ** 2).permute(0, 1, 4, 2, 3)
                        min_dx_layer_sq = min_dx_layer_sq * dropout_indicator * should_train_input_gaze_indicator

                        # (B, T, 1,H, W)
                        min_dy_layer_sq = ((gaze_side_channel_layer[:, :, :, :, 1:]) ** 2).permute(0, 1, 4, 2, 3)
                        min_dy_layer_sq = min_dy_layer_sq * dropout_indicator * should_train_input_gaze_indicator

                        # (B, T, 1,H, W)
                        min_dx_dy_layer = (
                            gaze_side_channel_layer[:, :, :, :, 0:1] * gaze_side_channel_layer[:, :, :, :, 1:]
                        ).permute(0, 1, 4, 2, 3)
                        min_dx_dy_layer = min_dx_dy_layer * dropout_indicator * should_train_input_gaze_indicator
                        individual_sc_input = torch.cat(
                            [
                                individual_sc_input,
                                min_dx_layer_sq,
                                min_dy_layer_sq,
                                min_dx_dy_layer,
                                min_distances,
                                should_train_input_gaze_indicator,
                                dropout_indicator,
                            ],
                            dim=2,
                        )  # (B, T, C=2+1+1+1+1+1+1, H, W) C=8
                        individual_sc_inputs.append(individual_sc_input)
                    elif side_key == "optic_flow":
                        # (B, T, C=2, H, W)
                        side_inp = side_channel_input[side_key]  # optic flow ux and uy layers

                        # apply adaptive avg pool to make optic flow to the correct resolution
                        # (BT, C=2, H, W)
                        side_inp = side_inp.view(-1, side_inp.shape[2], side_inp.shape[3], side_inp.shape[4])
                        adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d((H, W))

                        # (BT, C=2, h, w)
                        low_res_optic_flow_side_inp = adaptive_avg_pool(side_inp)
                        # (B, T, C=2, h, w)
                        low_res_optic_flow_side_inp = low_res_optic_flow_side_inp.reshape(
                            side_channel_input[side_key].shape[0], side_channel_input[side_key].shape[1], -1, H, W
                        )
                        individual_sc_inputs.append(low_res_optic_flow_side_inp)

            individual_sc_inputs = torch.cat(individual_sc_inputs, dim=2)
            # skip the skip connection for layers that are not specified for skip connection
            if not du_key in self.skip_layers_keys:
                # forward of decoder unit
                new_du_out = self.decoder_layers[du_key](
                    previous_du_out, None, individual_sc_inputs, upsm_size=upsm_target_size_list[i + 1]
                )
            else:
                new_du_out = self.decoder_layers[du_key](
                    previous_du_out,
                    encoder_output_dict[du_key],  # skip connection
                    individual_sc_inputs,
                    upsm_size=upsm_target_size_list[i + 1],
                )
            du_outs.append(new_du_out)
            previous_du_out = new_du_out

        # at this stage previous_du_out is the output of decoder_unit named layer1.
        self.upsm.size = [enc_input_shape[3], enc_input_shape[4]]
        # upsampling is done on tensor in which B and T are combined.
        # (BT, C, H, W)
        previous_du_out_tmp = self.upsm(
            previous_du_out.reshape(
                [
                    previous_du_out.shape[0] * previous_du_out.shape[1],  # BT
                    previous_du_out.shape[2],  # C
                    previous_du_out.shape[3],  # H
                    previous_du_out.shape[4],  # W
                ]
            )
        )
        # decouple B and T dimensions using reshape.
        decoder_net_output = previous_du_out_tmp.reshape(
            previous_du_out.shape[0],  # B
            previous_du_out.shape[1],  # T
            previous_du_out.shape[2],  # C
            previous_du_out_tmp.shape[2],  # H
            previous_du_out_tmp.shape[3],  # W
        )
        return decoder_net_output
