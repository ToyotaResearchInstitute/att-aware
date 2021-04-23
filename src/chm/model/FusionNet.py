import torch

from chm.model.EncoderNet import create_encoder
from chm.model.DecoderNet import create_decoder


def create_enc_dec_backbone(params_dict, network_out_height, network_out_weight):

    use_s3d = params_dict.get("use_s3d", False)
    add_optic_flow = params_dict.get("add_optic_flow", False)
    # num_of_features in the decoder units. [Last...last-1,..,first]
    decoder_layer_features = params_dict.get("decoder_layer_features", [8, 32, 64, 128])
    reduced_middle_layer_size = params_dict.get("reduced_middle_layer_size", 512)

    if not use_s3d:
        assert len(decoder_layer_features) == 4
    else:
        assert len(decoder_layer_features) == 5

    # create encoder
    road_facing_encoder = create_encoder(reduced_middle_layer_size=reduced_middle_layer_size, use_s3d=use_s3d)

    # input to the cognitive predictor (more like the output size of the decoder that should match the input size of the cognitive predictor)
    output_dim_of_decoder = decoder_layer_features[0]
    # skip connection sizes are determined by the ResNet Features size. Hence, hardcoded
    if not use_s3d:
        skip_layers = [64, 128, 256, 0]
    else:
        # skip connections only for layer 1 and layer 2, s3d_net_1, s3d_net_2, s3d_net_3 don't have skip connections
        skip_layers = [64, 128, 0, 0, 0]

    # reduced_middle_layer_size is the output_num_channels of the 3d conv after the last the encoder layer
    # and before the first decoder
    decoder_layer_features[-1] = reduced_middle_layer_size
    dx_sq_layer_dim = 1
    dy_sq_layer_dim = 1
    dx_times_dy_layer_dim = 1
    dxdy_dist_layer_dim = 1
    optic_flow_output_dim = 2
    gaze_transform_output_dim = 4

    decoder_net_params = {}
    decoder_net_params["output_dim_of_decoder"] = output_dim_of_decoder
    decoder_net_params["decoder_layer_features"] = decoder_layer_features
    decoder_net_params["skip_layers"] = skip_layers
    decoder_net_params["side_channel_input_dim"] = (
        gaze_transform_output_dim + dx_sq_layer_dim + dy_sq_layer_dim + dx_times_dy_layer_dim + dxdy_dist_layer_dim
    )
    if add_optic_flow:
        decoder_net_params["side_channel_input_dim"] = (
            decoder_net_params["side_channel_input_dim"] + optic_flow_output_dim
        )
    decoder_net_params["use_s3d"] = use_s3d

    road_facing_decoder = create_decoder(decoder_net_params)


class FusionNet(torch.nn.Module):
    """
    FusionNet is the network class that combines the different modules
    (road facing, driver facing, decoder) to form the encoder-decoder backbone
    """

    def __init__(self, params_dict):
        pass
