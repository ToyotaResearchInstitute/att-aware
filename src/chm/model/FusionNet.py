import torch

from chm.model.EncoderNet import create_encoder
from chm.model.DecoderNet import create_decoder


def create_enc_dec_backbone(params_dict, network_out_height, network_out_weight):

    use_s3d_encoder = params_dict.get("use_s3d_encoder", False)
    decoder_layer_features = params_dict.get("decoder_layer_features", [8, 32, 64, 128])
    reduced_middle_layer_size = params_dict.get("reduced_middle_layer_size", 512)
    if not use_s3d_encoder:
        assert len(decoder_layer_features) == 4
    else:
        assert len(decoder_layer_features) == 5

    # create encoder
    road_facing_encoder = create_encoder(
        reduced_middle_layer_size=reduced_middle_layer_size, use_s3d_encoder=use_s3d_encoder
    )


class FusionNet(torch.nn.Module):
    """
    FusionNet is the network class that combines the different modules
    (road facing, driver facing, decoder) to form the encoder-decoder backbone
    """

    def __init__(self, params_dict):
        pass
