# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch


class STConv3d(torch.nn.Module):
    """
    Adapted from https://github.com/qijiezhao/s3d.pytorch/blob/master/S3DG_Pytorch.py

    S3D structure used in encoder and decoder. Consists of separate spatial and temporal Conv3D with (1, k, k) and (k, 1,1) kernels.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        """
        Parameters
        ---------
        in_channels: int
            number of input features for the spatial conv3d
        out_channels: int
            number of output features for the spatial conv3d. This is also used as the number of
            input and output features for the temporal conv3d
        kernel_size: int
            Convolution kernel size
        stride: int
            Convolution stride length
        padding: int
            Convolution padding size
        """

        super(STConv3d, self).__init__()
        # for spatial conv only pad spatial dim
        self.spatial_replication_pad = torch.nn.ReplicationPad3d((0, 0, padding, padding, padding, padding))
        # spatial conv (1, k, k)
        self.spatial_conv3d = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(0, 0, 0),
        )
        # for temp dim only pad temporal dim
        self.temporal_replication_pad = torch.nn.ReplicationPad3d((padding, padding, 0, 0, 0, 0))
        # temporal conv (k, 1, 1)
        self.temporal_conv3d = torch.nn.Conv3d(
            out_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1), padding=(0, 0, 0)
        )
        self.spatial_instance_norm = torch.nn.InstanceNorm3d(out_channels)
        self.temporal_instance_norm = torch.nn.InstanceNorm3d(out_channels)

        self.spatial_nonlin = torch.nn.ReLU()
        self.temporal_nonlin = torch.nn.ReLU()

    def forward(self, x):
        """
        assumes x is (B, C, T, H, W)
        """

        x = self.spatial_nonlin(self.spatial_instance_norm(self.spatial_conv3d(self.spatial_replication_pad(x))))
        x = self.temporal_nonlin(self.temporal_instance_norm(self.temporal_conv3d(self.temporal_replication_pad(x))))
        return x