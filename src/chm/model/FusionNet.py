import torch

from chm.model.EncoderNet import create_encoder
from chm.model.DecoderNet import create_decoder


def create_enc_dec_backbone(params_dict, network_out_height, network_out_width):

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

    # side channel input channel dimensions
    # voronoi mapa dimensions
    dx_sq_layer_dim = 1
    dy_sq_layer_dim = 1
    dx_times_dy_layer_dim = 1
    dxdy_dist_layer_dim = 1
    # optic flow dimensions
    optic_flow_output_dim = 2
    # side channel gaze module dim
    gaze_transform_output_dim = 4

    # decoder params dict.
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

    # create decoder
    road_facing_decoder = create_decoder(decoder_net_params)

    # Side channel modules
    side_channel_modules = torch.nn.ModuleDict()
    side_channel_modules["driver_facing"] = torch.nn.ModuleDict()
    side_channel_modules["driver_facing"]["linear0"] = None

    if add_optic_flow:
        side_channel_modules["optic_flow"] = torch.nn.ModuleDict()
        side_channel_modules["optic_flow"]["linear0"] = None

    # Road facing modules
    map_modules = torch.nn.ModuleDict()
    map_modules["road_facing"] = torch.nn.ModuleDict()
    map_modules["road_facing"]["encoder"] = road_facing_encoder
    map_modules["road_facing"]["decoder"] = road_facing_decoder

    output_dims = collections.OrderedDict()
    output_dims["road_facing"] = [output_dim_of_decoder, network_out_width, network_out_height]
    fusion_net = FusionNet(side_channel_modules, map_modules, output_dims, params_dict)

    return fusion_net


class FusionNet(torch.nn.Module):
    """
    FusionNet is the network class that combines the different modules
    (road facing, driver facing, decoder) to form the encoder-decoder backbone
    """

    def __init__(self, side_channel_modules, map_modules, output_dims, params_dict):
        super().__init__()
        self.params_dict = params_dict
        self.side_channel_modules = side_channel_modules
        self.map_modules = map_modules
        self.output_dims = output_dims

        self.dropout_ratio = self.params_dict.get("dropout_ratio", {"driver_facing": 0.5, "optic_flow": 0.0})
        self.dropout_ratio_external_inputs = self.params_dict.get("dropout_ratio_external_inputs", 0.0)

        # If key is in the self.force_input_dropout, use it to override the input's dropout.
        # The key corresonds to the name of the child networks. driver facing, optic flow
        self.force_input_dropout = {}
        import IPython

        IPython.embed(banner1="check in fusion net init")

    def forward(self, side_channel_input, should_drop_indices_dict=None, should_drop_entire_channel_dict=None):
        side_channel_outputs = []  # list containing the outputs of each side_channel networks
