# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch
import numpy as np


class EPSpatialRegularization(torch.nn.Module):
    """
    Edge Preserving Spatial Regularization
    """

    def __init__(self, image_width, image_height, eps=1e-3, sig_scale_factor=1):
        """
        Parameters:
        ----------
        image_width: int
            Network image width
        image_height: int
            Network image height
        eps: float
            Epsilon to avoid divide by zero error in diffusivity computation
        sig_scale_factor: int
            Scaling factor for std deviation parameter for the gaussian blur filter
        """
        super(EPSpatialRegularization, self).__init__()

        self.eps = eps
        self.image_width = image_width
        self.image_height = image_height
        self.sig_scale_factor = sig_scale_factor

        # register difference filters (non-centered and centered versions)
        self.register_buffer("conv_filter", torch.Tensor([[[[0, 0, 0], [1, -1, 0], [0, 0, 0]]]]))
        self.register_buffer("conv_filter2", torch.Tensor([[[[0, 1, 0], [0, -1, 0], [0, 0, 0]]]]))
        self.register_buffer("conv_filter_centered", torch.Tensor([[[[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]]]]))
        self.register_buffer("conv_filter_centered2", torch.Tensor([[[[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]]]]))

        # create a gaussian blur filter.
        sig = round(0.3 * self.image_width / 60)
        N = int(np.ceil(sig) * self.sig_scale_factor + 1)  # 2sig + 1 = R

        self.blur_N = N
        kernel_x, kernel_y = np.meshgrid(range(-N, N + 1), range(-N, N + 1))
        # gaussian kernel
        kernel = np.exp(-(kernel_x ** 2 + kernel_y ** 2) / (sig ** 2))
        # normalize kernel
        kernel = torch.Tensor(kernel / np.sum(kernel)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("blur_kernel", kernel)

    def forward(self, heatmap, image=None):
        """
        Parameters:
        ----------
        heatmap: torch.Tensor
            Gaze heatmap or awareness map on which spatial regularization is computed
        image: torch.Tensor
            Mask image or road image used for computing diffusivity

        Returns:
        --------
        result: torch.Tensor
            Spatial regularization cost computed on heatmap
        stats: dict
            Dictionary containing diffusivity related stats
        """
        filter_c = self.conv_filter_centered  # (1,1,3,3)
        filter_c2 = self.conv_filter_centered2  # (1,1,3,3)

        # pad with reflective boundary conditions
        # image of shape (B, T, C=3, H, W) (mask or road image)
        # combine B and T dimensions and pad height and width dimensions. (BT, C, H+pad, W+pad)
        padded_image = torch.nn.ReflectionPad2d(self.blur_N)(
            image.view(-1, image.shape[2], image.shape[3], image.shape[4])
        )
        # (BTC, H+pad, W+pad) image.shape[0] = B, image.shape[1]=T, padded_image.shape[1] = channels
        # reshape it (BTC, H, W)
        padded_image = padded_image.view(
            image.shape[0] * image.shape[1] * padded_image.shape[1], padded_image.shape[2], padded_image.shape[3]
        )
        # (BTC, 1, H, W) Unsqueeze adds the missing channel dim for convolution. Blur the padded image with a gaussian filter.
        blurred_image = torch.nn.functional.conv2d(padded_image.unsqueeze(1), weight=self.blur_kernel)

        # compute gradients along x and y from the road/mask images.
        Ix = torch.nn.functional.conv2d(blurred_image, filter_c, padding=1).view_as(image)  # (B, T, C, H, W)
        Iy = torch.nn.functional.conv2d(blurred_image, filter_c2, padding=1).view_as(image)  # (B, T, C, H, W)

        # process heatmaps

        transposed_heatmap = heatmap.transpose(1, 2)  # (B, C=1, T, H, W)
        reflection_pad_2d = torch.nn.ReflectionPad2d(1)
        reflected_transposed_heatmap = reflection_pad_2d(
            transposed_heatmap.view(
                -1, transposed_heatmap.shape[2], transposed_heatmap.shape[3], transposed_heatmap.shape[4]
            )
        )  # (BC, T, H+pad, W+pad), pad = 1+1
        reflected_transposed_heatmap = reflected_transposed_heatmap.view(
            transposed_heatmap.shape[0],
            transposed_heatmap.shape[1],
            reflected_transposed_heatmap.shape[1],
            reflected_transposed_heatmap.shape[2],
            reflected_transposed_heatmap.shape[3],
        )
        filt = self.conv_filter.unsqueeze(0).repeat(1, 1, 1, 1, 1)  # (1,1,1,3,3)
        filt2 = self.conv_filter2.unsqueeze(0).repeat(1, 1, 1, 1, 1)  # (1,1,1,3,3)
        # (B, T, C=1, H, W)
        filter_output = torch.nn.functional.conv3d(reflected_transposed_heatmap, filt).transpose(1, 2)
        # (B, T, C=1, H, W)
        filter_output2 = torch.nn.functional.conv3d(reflected_transposed_heatmap, filt2).transpose(1, 2)

        stats = {}
        diffusivity = (self.eps + (Ix ** 2 + Iy ** 2).sum(dim=2)) ** (-0.5)  # (B, T, H, W)
        stats["diffusivity"] = diffusivity
        result = (((filter_output ** 2).sum(dim=2) + (filter_output2 ** 2).sum(dim=2)) ** (1.0)) * diffusivity

        return result, stats


class EPTemporalRegularization(torch.nn.Module):
    """
    Edge Preserving Temporal Regularization
    """

    def __init__(
        self,
        image_width,
        image_height,
        eps=1e-3,
        sig_scale_factor=1,
        negative_difference_coeff=20.0,
        positive_difference_coeff=1.0,
    ):
        """
        Parameters:
        ----------
        image_width: int
            Network image width
        image_height: int
            Network image height
        eps: float
            Epsilon to avoid divide by zero error in diffusivity computation
        sig_scale_factor: int
            Scaling factor for std deviation parameter for the gaussian blur filter
        negative_difference_coeff: float
            Weighting factor for decreasing temporal awareness.
        positive_difference_coeff: float
            Weight factor for increasing temporal awareness
        """
        super(EPTemporalRegularization, self).__init__()

        self.eps = eps
        self.image_width = image_width
        self.image_height = image_height
        self.sig_scale_factor = sig_scale_factor
        self.NEGATIVE_DIFFERENCE_COEFFICIENT = negative_difference_coeff
        self.POSITIVE_DIFFERENCE_COEFFICIENT = positive_difference_coeff

        # register temporal different filters and spatial
        self.register_buffer("temporal_filter", torch.Tensor([-1, 0, 1]))
        self.register_buffer("conv_filter_centered", torch.Tensor([[[[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]]]]))
        self.register_buffer("conv_filter_centered2", torch.Tensor([[[[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]]]]))

        self.reflection = torch.nn.ReflectionPad1d((1, 1))
        sig = round(0.3 * self.image_width / 60)
        N = int(np.ceil(sig) * self.sig_scale_factor + 1)  # fatcor*sig + 1 = R

        self.blur_N = N
        # create, normalize and register gaussian kernel
        kernel_x, kernel_y = np.meshgrid(range(-N, N + 1), range(-N, N + 1))
        kernel = np.exp(-(kernel_x ** 2 + kernel_y ** 2) / (sig ** 2))
        kernel = torch.Tensor(kernel / np.sum(kernel)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("blur_kernel", kernel)  # (1, 1, 2*N, 2*N)

    def forward(self, heatmap, image):
        """
        Parameters:
        ----------
        heatmap: torch.Tensor
            Gaze heatmap or awareness map on which temporal regularization is computed
        image: torch.Tensor
            Mask image or road image used for computing diffusivity

        Returns:
        --------
        result: torch.Tensor
            Temporal regularization cost computed on heatmap
        stats: dict
            Dictionary containing diffusivity related stats
        """

        filter_c = self.conv_filter_centered  # (1,1,3,3)
        filter_c2 = self.conv_filter_centered2  # (1,1,3,3)

        # pad with reflective boundary conditions
        # image.view(-1, image.shape[2], image.shape[3], image.shape[4]) - (BT, C=3, H, W)
        # (BT, C=3, H+2*N, W+2*N)
        padded_image = torch.nn.ReflectionPad2d(self.blur_N)(
            image.view(-1, image.shape[2], image.shape[3], image.shape[4])
        )
        # (BTC,  H+2*N, W+2*N) With C=3
        padded_image = padded_image.view(
            image.shape[0] * image.shape[1] * padded_image.shape[1], padded_image.shape[2], padded_image.shape[3]
        )
        # padded_image.unsqueeze(1) = (BTC, 1, H+2*N, W+2*N)
        # gaussian smooth all the channels of the padded image
        blurred_image = torch.nn.functional.conv2d(padded_image.unsqueeze(1), weight=self.blur_kernel)

        # gradients along x and y
        Ix = torch.nn.functional.conv2d(blurred_image, filter_c, padding=1).view_as(image)
        Iy = torch.nn.functional.conv2d(blurred_image, filter_c2, padding=1).view_as(image)

        # (BTC, 1, H, W), with C=3 Padding makes sure that the size is the same after convultion
        transposed_heatmap = heatmap[:, :, 0, :, :].transpose(1, -1)  # (B, W, H, T)
        reflection_pad_1d = torch.nn.ReflectionPad1d((1, 1, 0, 0))
        reflected_transposed_heatmap = reflection_pad_1d(transposed_heatmap)  # (B, W, H, T+pad), pad =1+1
        reflected_transposed_heatmap = reflected_transposed_heatmap.transpose(1, -1)  # (B, T+pad, H, W)
        reflected_transposed_heatmap = reflected_transposed_heatmap.unsqueeze(1)  # (B, C, T+pad, H, W)

        # apply temporal filter
        # (1,1,3,1,1) kernel is size 3 for time dimension
        temporal_flt = self.temporal_filter.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # (B,T,C=1, H, W)
        temporal_difference = torch.nn.functional.conv3d(reflected_transposed_heatmap, temporal_flt).transpose(1, 2)
        temporal_difference = temporal_difference[:, :, 0, :, :]  # (B, T, H, W)

        # split into positive, negative parts
        positive_temporal_difference = temporal_difference.clamp(min=0)  # (B, T, H, W)
        negative_temporal_difference = -temporal_difference.clamp(max=0)

        # penalize decrease more drastically, hence the squared.
        negative_temporal_difference = negative_temporal_difference ** 2 * self.NEGATIVE_DIFFERENCE_COEFFICIENT
        # positive change in awareness (increase) is penalized less, hence linear
        positive_temporal_difference = positive_temporal_difference * self.POSITIVE_DIFFERENCE_COEFFICIENT
        temporal_difference_penalty = positive_temporal_difference + negative_temporal_difference

        stats = {}
        diffusivity = (self.eps + (Ix ** 2 + Iy ** 2).sum(dim=2)) ** (-0.5)  # (B, T, H, W)
        stats["diffusivity"] = diffusivity
        result = ((temporal_difference_penalty) ** (1.0)) * diffusivity  # (B, T, H, W)

        return result, stats
