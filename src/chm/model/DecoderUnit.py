# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch

from chm.model.S3D import STConv3d


class DecoderUnit(torch.nn.Module):
    """
    Individual spatio-temporal convolutional units used in the DecoderNet.
    """

    def __init__(self, last_out_dim, output_dim, skip_dim, side_channel_input_dim, use_s3d=False):
        """
        Parameters:
        ----------
        last_out_dim: int
            Number of output features from the previous decoder unit, if it exists.
        output_dim: int
            Number of output features for this decoder unit
        skip_dim: int
            Number of features in the skip connections from the corresponding encoder layers
        side_channel_input_dim: int
            Number of layers in the side channel inputs
        use_s3d: bool
            Flag which indicates whether the separable conv3D should be used
        """
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

        self.upsm = torch.nn.Upsample(mode="bilinear", align_corners=False)

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
                skip_input = self.skip_net_s3d(skip_input)

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