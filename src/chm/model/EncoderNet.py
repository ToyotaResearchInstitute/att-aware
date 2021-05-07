# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch
import torchvision.models
import collections

from chm.model.S3D import STConv3d


def run_over_images(input, net, axis):
    """
    Parameters:
    ----------
    input: torch.Tensor
        (B, T, C, H, W). Image sequence
    net: torch.nn.Module
        network to be used for processing input
    axis: int
        axis along which the input needs to be extract for individual application of network

    Returns:
    --------
    output: torch.Tensor
        Result of applying net on input
    """
    output = []
    for i in range(input.shape[axis]):
        # for the entire batch grab the ith image along axis by using index_select,
        # remove the time dimension using squeeze, pass it through a network
        # that expects a batch of images, and then finally add back the removed dimension
        tmp = net(torch.index_select(input, dim=axis, index=input.new_tensor([i]).long()).squeeze(axis)).unsqueeze(axis)
        output.append(tmp)
    # concatenate the results along the time dimension.
    # so its back to B * T * C * H * W
    output = torch.cat(output, axis)
    return output


def create_encoder(reduced_middle_layer_size=512, use_s3d=False):
    """
    Creates the encoder structure. Two possible encoders.
    1. With only ResNet18 encoder layers. 4 layers
    2. When use_s3d is True, then 3 layers of S3D will be stacked on top of two layers of ResNet18. Total of 5

    Parameters:
    ----------
    reduced_middle_layer_size: int
        Number of feature layers for the Conv3D (1 x 1 x 1) after the encoder. Used for reduced the number of parameters
    use_s3d: bool
        Flag which decides the encoder structure

    Returns:
    -------
    EncoderNet: torch.nn.Module
        Encoder network for CHM
    """

    orig_net = torchvision.models.resnet18(pretrained=True)  # use resnet
    # initial layers of resnet for preprocessing
    preproc_net = torch.nn.Sequential()
    preproc_net.add_module("conv1", orig_net.conv1)
    preproc_net.add_module("bn1", orig_net.bn1)
    preproc_net.add_module("relu1", orig_net.relu)
    preproc_net.add_module("maxpool", orig_net.maxpool)

    # encoder layers from resnet for 2d convs. Present for both nonS3D and S3D architectures
    layered_net = torch.nn.ModuleDict()
    layered_net["layer1"] = orig_net.layer1
    layered_net["layer2"] = orig_net.layer2
    if not use_s3d:
        layered_net["layer3"] = orig_net.layer3
        layered_net["layer4"] = orig_net.layer4
        s3d_net = None
        # encoder layers will be [layer1, layer2, layer3, layer4], None
    else:
        s3d_net = torch.nn.ModuleDict()
        s3d_layer_3_out_channels = 512
        # the input dim for s3d net should match the output of layer2 of resnet.
        s3d_net["s3d_net_1"] = STConv3d(
            in_channels=list(orig_net.layer2.children())[-1].bn2.num_features,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        s3d_net["s3d_net_2"] = STConv3d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        s3d_net["s3d_net_3"] = STConv3d(
            in_channels=512,
            out_channels=s3d_layer_3_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # encoder layers will be [layer1, layer2], [s3d_net_1, s3d_net_2, s3d_net_3]

    post_processing = None
    post_processing_bn = None
    post_processing_nonlin = None
    if reduced_middle_layer_size is not None:
        if not use_s3d:
            orig_latent_dim = list(orig_net.layer4.children())[-1].conv2.out_channels
        else:
            orig_latent_dim = s3d_layer_3_out_channels  # out_channels in s3d_net['s3d_net_3']

        post_processing = torch.nn.Conv3d(orig_latent_dim, reduced_middle_layer_size, 1)
        post_processing_in = torch.nn.InstanceNorm3d(reduced_middle_layer_size)
        post_processing_nonlin = torch.nn.ReLU()

    post_layers = {}
    post_layers["conv3d"] = post_processing
    post_layers["in"] = post_processing_in
    post_layers["relu"] = post_processing_nonlin

    return EncoderNet(preproc_net=preproc_net, layered_outputs=layered_net, s3d_net=s3d_net, post_layers=post_layers)


class EncoderNet(torch.nn.Module):
    """
    Class for the Encoder Net in CHMNet
    """

    def __init__(self, preproc_net, layered_outputs, s3d_net=None, post_layers=None):
        """
        Parameters:
        -----------
        preproc_net: torch.nn.Sequential
            Initialize preprocessing modules for encoder
        layered_output: torch.nn.ModuleDict
            Initial layers of the encoder. Either [layer1, layer2] or [layer1, layer2...layer4]
        s3d_net: torch.nn.ModuleDict or None
            If use_s3d flag was true the s3d_net consists of [s3d_net_1, s3d_net_2, s3d_net_3]. else None
        post_layers: dict or None
            dict containing the various layers for post processing

        """
        super().__init__()
        self.preproc_net = preproc_net
        self.layered_outputs = layered_outputs
        self.s3d_net = s3d_net
        self.post_layers = post_layers
        if self.post_layers is not None:
            self.post_layer_conv3d = self.post_layers["conv3d"]
            self.post_layer_in = self.post_layers["in"]
            self.post_layer_relu = self.post_layers["relu"]

    def forward(self, input):
        """
        Encode an input (or a sequence of input) image(s) via a set of convolutions and subsamples, save the intermediate outputs for shortcut
        connections.

        Parameters:
        ----------
        input: torch.Tensor
            input (or sequence of) images

        Returns:
        -------
        encoder_output: OrderedDict()
            a dictionary containing all the intermediate results of applying fwd() on the image or batch of images
        """
        intermediate_output = run_over_images(input, self.preproc_net, axis=1)
        encoder_output = collections.OrderedDict()
        for key in self.layered_outputs:  # keys are [layer1, layer2, [layer3, layer4]]
            encoder_output[key] = run_over_images(intermediate_output, self.layered_outputs[key], axis=1)
            intermediate_output = encoder_output[key]

        if self.s3d_net is not None:
            for key in self.s3d_net:
                intermediate_output = intermediate_output.permute(0, 2, 1, 3, 4)  # B, T, C, H, W --> B, C, T, H, W
                out = self.s3d_net[key](intermediate_output)  # (B, C, T, H, W)
                encoder_output[key] = out.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
                intermediate_output = encoder_output[key]

        if self.post_layers is not None:
            # from [B,T,C,H,W] to [B,C,T,H,W] for 3D conv, and back.
            intermediate_output = self.post_layer_relu(
                self.post_layer_in(self.post_layer_conv3d(encoder_output[key].transpose(1, 2)))
            ).transpose(1, 2)
            encoder_output[key] = intermediate_output

        return encoder_output