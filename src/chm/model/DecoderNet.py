import torch

from chm.model.S3D import STConv3d


def create_decoder(decoder_net_params=None):
    assert decoder_net_params is not None

    use_s3d = decoder_net_params["use_s3d"]
    decoder_layer_features = decoder_net_params["decoder_layer_features"]
    output_dim_of_decoder = decoder_net_params["output_dim_of_decoder"]
    side_channel_input_dim = decoder_net_params["side_channel_input_dim"]
    skip_layers = decoder_net_params["skip_layers"]

    post_proc = torch.nn.Sequential()
    decoder_layers = torch.nn.ModuleDict()

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
            last_out_dim=decoder_layer_features[-1],  # 128 #
            output_dim=decoder_layer_features[-2],  # 64
            skip_dim=skip_layers[-1],  # 0
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )  # innermost DU. skip_layers[-1] = 0
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
        skip_layers_keys = ["layer2", "layer1"]

    return DecoderNet(
        postproc=postproc,
        decoder_layers=decoder_layers,
        skip_layers_keys=skip_layers_keys,
    )  # since input gaze is normalized, the full_scale parameters is set to 1.0


class DecoderUnit(torch.nn.Module):
    def __init__(self, last_out_dim, output_dim, skip_dim, side_channel_input_dim, use_s3d=False):
        super().__init__()
        self.use_s3d = use_s3d
        self.replication_pad = torch.nn.ReplicationPad3d(1)
        # create main conv3d processing module for processing previous decoder unit output and side channel input
        if not self.use_s3d:
            self.conv3d_net = torch.nn.Conv3d(
                in_channels=last_out_dim + skip_dim + side_channel_input_dim,
                out_channels=output_dim,
                kernel_size=(3, 3, 3),
                padding=(0, 0, 0),
                stride=(1, 1, 1),
            )
            self.instance_norm = torch.nn.InstanceNorm3d(output_dim)
            self.nonlin = torch.nn.ReLU()
        else:
            self.sep_conv3d = STConv3d(
                in_channels=last_out_dim + skip_dim + side_channel_input_dim,
                out_channels=output_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.upsm = torch.nn.Upsample(mode="bilinear")

        # create conv3d module for processing skip connections
        if skip_dim != 0:
            if not self.use_s3d:
                self.skip_net = torch.nn.Conv3d(
                    in_channels=skip_dim,
                    out_channels=skip_dim,
                    kernel_size=(3, 3, 3),
                    padding=(0, 0, 0),
                    stride=(1, 1, 1),
                )
                self.skip_instance_norm = torch.nn.InstanceNorm3d(skip_dim)
                self.skip_nonlin = torch.nn.ReLU()
            else:
                self.skip_net_s3d = STConv3d(
                    in_channels=skip_dim,
                    out_channels=skip_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )

    def forward(self, last_out, skip_input, side_channel_input, upsm_size):
        """
        Parameters
        ----------
        last_out: torch.Tensor (B, T, C, H, W)
            output from the previous DecoderUnit in the DecoderNet
        skip_input: torch.Tensor (B, T, C, H, W) or None
            skip connection from corresponding layer in the EncoderNet
        side_channel_input: torch.Tensor (B, T, C, H, W)
            side channel input consisting of gaze Voronoi Maps, optic flow, train_indicator and dropout indicator bits
        upsm_size: torch.Tensor (2, 1)
            target resolution after upsampling.

        Returns
        -------
        du_output: torch.Tensor  (B, T, C', H', W'),
            output of the decoder unit with proper resolution for consumption by the next
            unit in the processing chain
        """
        if skip_input is None:
            joint_input = torch.cat(
                # side_channel_input is of size (B, T, side_channel_input_dim, H, W) - Voronoi maps + dropout + train_indicator+ dx2 , dy2, dxdy, optic flow
                [last_out, side_channel_input],
                2,
            )
        else:
            # process the temporal aspect of skip connection first.
            skip_input = skip_input.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            # replication pad so that after conv3d the size remains the same.
            if not self.use_s3d:
                skip_input = self.replication_pad(skip_input)
                skip_input = self.skip_nonlin(self.skip_instance_norm(self.skip_net(skip_input)))
            else:
                skip_input = self.skip_net_sep_conv3d(skip_input)

            skip_input = skip_input.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
            # concatenate along the channel dimension. (B, T, C, H, W)
            joint_input = torch.cat([last_out, skip_input, side_channel_input], 2)

        joint_input = joint_input.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        if not self.use_s3d:
            # (B, C, T+2, H+2, W+2), increased T, H, W dimensions due to replication padding. on both sides.
            joint_input = self.replication_pad(joint_input)
            # (B, C', T, H, W) C' is the output dim of the conv3d
            du_output = self.nonlin(self.instance_norm(self.conv3d_net(joint_input)))
        else:
            # (B, C', T, H, W)
            du_output = self.sep_conv3d(joint_input)

        # (BC', T, H, W) for applying upsm on H and W, the tensor has to be 4D
        du_output = du_output.view(du_output.shape[0] * du_output.shape[1], du_output.shape[2], du_output.shape[3], -1)

        # Upsampling the output
        self.upsm.size = upsm_size  # set the upsm size
        du_output = self.upsm(du_output)  # (BC', T, H', W')
        # (B, C, T, H', W'). Use B from joint_input and infer C dimension
        du_output = du_output.reshape(
            joint_input.shape[0], -1, du_output.shape[1], du_output.shape[2], du_output.shape[3]
        )
        # (B, T, C', H', W') So that the skip connection from encoder can be added properly along C, H', W' dimensions for subsequent decoder layer
        du_output = du_output.permute(0, 2, 1, 3, 4)
        return du_output


class DecoderNet(torch.nn.Module):
    def __init__(self, post_proc, decoder_layers, skip_layers_keys):
        super().__init__()
        self.postproc = postproc
        self.decoder_layers = decoder_layers  # these are the DecoderUnits.
        self.skip_layers_keys = skip_layers_keys
        self.upsm = torch.nn.Upsample(mode="bilinear")

    def forward(self, input, side_channel_input, enc_input_shape):
        pass