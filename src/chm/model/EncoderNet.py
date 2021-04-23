import torch
import torchvision.models

from chm.model.S3D import STConv3d


def create_encoder(reduced_middle_layer_size=512, use_s3d_encoder=False):
    """
    Creates the encoder structure. Two possible encoders.
    1. With only ResNet18 encoder layers. 4 layers
    2. When use_s3d_encoder is True, then 3 layers of S3D will be stacked on top of two layers of ResNet18. Total of 5

    Parameters:
    ----------
    reduced_middle_layer_size: int
        Number of feature layers for the Conv3D (1 x 1 x 1) after the encoder. Used for reduced the number of parameters
    use_s3d_encoder: bool
        Flag which decides the encoder structure
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
    if not use_s3d_encoder:
        layered_net["layer3"] = orig_net.layer3
        layered_net["layer4"] = orig_net.layer4
        s3d_net = None
        # encoder layers will be [layer1, layer2, layer3, layer4], None
    else:
        s3d_net = torch.nn.ModuleDict()
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
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # encoder layers will be [layer1, layer2], [s3d_net_1, s3d_net_2, s3d_net_3]

    import IPython

    IPython.embed(banner1="check encoder")
