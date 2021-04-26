import torch
import numpy as np


class EPSpatialRegularization(torch.nn.Module):
    """
    Edge Preserving Spatial Regularization
    """

    def __init__(self, image_width, image_height, eps=1e-3, sig_scale_factor=1):
        super(EPSpatialRegularization, self).__init__()

        self.eps = eps
        self.image_width = image_width
        self.image_height = image_height
        self.sig_scale_factor = sig_scale_factor

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
        # normalize
        kernel = torch.Tensor(kernel / np.sum(kernel)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("blur_kernel", kernel)

    def forward(self, heatmap, image=None):
        """
        Parameters:
        ----------
        heatmap: torch.Tensor
            gaze heatmap or awareness map on which spatial regularization is computed
        image: torch.Tensor
            mask image or road image used for computing diffusivity
        """
        filter_c = self.conv_filter_centered  # (1,1,3,3)
        filter_c2 = self.conv_filter_centered2  # (1,1,3,3)

        # Pad with reflective boundary conditions
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
    def __init__(self, image_width, image_height, eps=1e-3, sig_scale_factor=1):
        super(EPTemporalRegularization, self).__init__()

    def forward(self, map, image):
        import IPython

        IPython.embed(banner1="in EP temp")
