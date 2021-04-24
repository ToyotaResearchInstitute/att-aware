import torch
import collections

from chm.model.EncoderNet import create_encoder
from chm.model.DecoderNet import create_decoder


def create_fusion_net(params_dict, network_out_height, network_out_width):

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

    """

    def __init__(self, side_channel_modules, map_modules, output_dims, params_dict):
        """
        Parameters:
        -----------
        side_channel_modules: torch.nn.ModuleDict()
            Side channel network modules to process different types of side channel input such as gaze, optic flow etc.
        map_modules: torch.nn.Modules()
            Main encoder-decoder backbone used for processing road image input
        output_dims: OrderedDict()
            dict containing the output dimensions (for example, [3, 240, 135]) for each of the encoder-decoder backbone in map_modules
        params_dict : dict
            dict containing args for network structure
        """
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

    def forward(self, fusion_net_input, should_drop_indices_dict=None, should_drop_entire_channel_dict=None):
        """
        Parameters:
        ----------
        fusion_net_input: dict
            Dict containing inputs to the main encoder as the side channel inputs
        should_drop_indices_dict: dict
            Dict with boolean flags indicating which batch items needed to dropped out during training
        should_drop_entire_channel_dict: dict
            Dict with flags indicating if an entire side channel needs to be dropped out

        Returns:
        -------

        fusion_output: torch.Tensor
            Output of the FusionNet (= output from the final decoder layer)
        side_channel_output: dict
            Containing outputs of the side channel modules. Used as input to the decoder layers
        should_drop_indices_dict: dict
            Dict with boolean flags indicating which batch items were dropped out during training
        should_drop_entire_channel_dict: dict
            Dict with flags indicating which entire side channel was dropped out
        """
        side_channel_module_outputs = []  # list containing the outputs of each side_channel networks

        # Process the side channel information.
        for fusion_net_inp_key in fusion_net_input:
            if fusion_net_inp_key in self.side_channel_modules:
                # For fusion_net_inp_key='driver_facing', fusion_net_input[fusion_net_inp_key] is (B, T, L, 4).
                # Last dimension is [x,y,should_train_indicator, droput_indicator].
                # The input to driver facing module comes from the GazeTransform module

                # for fusion_net_inp_key='optic_flow', fusion_net_input[fusion_net_inp_key] is (B, T, 2, H, W)
                sc_input = fusion_net_input[fusion_net_inp_key]
                if fusion_net_inp_key == "driver_facing":
                    sc_out = sc_input
                    for subkey in self.side_channel_modules[fusion_net_inp_key]:
                        sc_net = self.side_channel_modules[fusion_net_inp_key][subkey]
                        if sc_net is not None:
                            if subkey == "linear0":
                                sc_input_reshaped = sc_input.reshape(
                                    sc_input.shape[0], sc_input.shape[1] * sc_input.shape[2], -1
                                )  # (B, TL, 4)
                                sc_out_reshaped = sc_net(sc_input_reshaped)  # (B, TL,4)
                                sc_out = sc_out_reshaped.reshape(
                                    sc_input.shape[0], sc_input.shape[1], sc_input.shape[2], -1
                                )  # (B, T, L, 4)
                elif fusion_net_inp_key == "optic_flow":
                    sc_out = sc_input  # (B,T,C=2,H,W)
                    for subkey in self.side_channel_modules[fusion_net_inp_key]:
                        sc_net = self.side_channel_modules[fusion_net_inp_key][subkey]
                        if sc_net is not None:
                            sc_out = sc_net(sc_out)

                side_channel_module_outputs.append((sc_out, fusion_net_inp_key))

        side_channel_module_outputs_list = [x[0] for x in side_channel_module_outputs]
        # names of the side channel module outputs that produced the intermediate list
        side_channel_module_outputs_list_keys = [x[1] for x in side_channel_module_outputs]
        # For each type of child network output choose to dropout or not some of the data_items in the batch
        for i, (x, x_key) in enumerate(zip(side_channel_module_outputs_list, side_channel_module_outputs_list_keys)):
            # creates a vector of size BATCH_SIZE (x.shape[0]) of bools indicating whether the intermediate result should be dropped or not?
            # x_key is driver_facing, task_network etc. THis is the "dict" passed from the command line
            assert x_key in self.dropout_ratio
            # each side channel input has its own dropout ratio, as specified in the dropout_ratio dict
            # if not should_drop_indices_dict was explicitly passed to the forward function
            if should_drop_indices_dict is None:
                # determines which of the batch items should be dropped out
                should_drop = torch.rand([x.shape[0]]) < self.dropout_ratio[x_key]
                should_drop_indices_dict = collections.OrderedDict()
                should_drop_indices_dict[x_key] = should_drop
            else:
                # could happen because a partial should_drop_indices_dict was passed or it was created in the previous if block
                if should_drop_indices_dict is not None:
                    if x_key not in should_drop_indices_dict:
                        should_drop = torch.rand([x.shape[0]]) < self.dropout_ratio[x_key]
                        should_drop_indices_dict[x_key] = should_drop
                    else:
                        assert (
                            x_key in should_drop_indices_dict
                        ), "the key for the side channel should be present in the indices dict"
                        should_drop = should_drop_indices_dict[x_key]

            # If it is testing override the rand vector created and zero it out
            if not (self.training):
                should_drop = should_drop * 0

            # this should be the same keys as the side channel networks, such as driver_facing, optic_flow etc.
            # Also doesn't make sense to loop over all keys. Only use the key corresponding to the ith element in the side_channel_module_outputs_list
            if x_key in self.force_input_dropout:
                for b in range(x.shape[0]):
                    should_drop[b] = self.force_input_dropout[x_key]
            for b in range(x.shape[0]):
                if should_drop[b]:
                    x[b, :] *= 0  # is mutable. Therefore elements in intermediate list will be changed

            # side channel output (after processed through side channel modules) and dropout.
            # From the decoder's perspective, it will be referred to as side_channel_input in DecoderNet class
            side_channel_output = {key: sc_out for sc_out, key in side_channel_module_outputs}

        for key in side_channel_output:
            # if any of the intermediate outputs are nans STOP!
            if torch.isnan(side_channel_output[key]).sum():
                import IPython

                print("Intermediate output", side_channel_output)
                print("Input", input)
                IPython.embed(
                    header="Fusion::forward - side_channel_output (concatenated outputs of the child networks) has nans"
                )

        fusion_output = collections.OrderedDict()
        for fusion_net_inp_key in fusion_net_input:
            # Process the main enc-dec backbone inputs
            if fusion_net_inp_key in self.map_modules:
                # The intermediate output from the side_channel networks are fed alongside the output of the encoder
                # as the input to decoder for EACH enc-dec module separately.
                enc = self.map_modules[fusion_net_inp_key]["encoder"]
                dec = self.map_modules[fusion_net_inp_key]["decoder"]

                enc_input_shape = fusion_net_input[fusion_net_inp_key].shape

                # output of the encoder network as a dictionary containing the intermediates results of layer1,2,3,4
                encoder_output_dict = enc(fusion_net_input[fusion_net_inp_key])
                side_channel_output_tmp = side_channel_output
                if self.training:
                    # to determine if the whole child network output should be dropped off or not.
                    for side_key in side_channel_output_tmp:
                        if should_drop_entire_channel_dict is None:
                            should_drop_entire_channel = torch.rand([1]) < self.dropout_ratio_external_inputs
                            should_drop_entire_channel_dict = collections.OrderedDict()
                            should_drop_entire_channel_dict[side_key] = should_drop_entire_channel
                        else:
                            if should_drop_entire_channel_dict is not None:
                                if side_key not in should_drop_entire_channel_dict:
                                    should_drop_entire_channel = torch.rand([1]) < self.dropout_ratio_external_inputs
                                    should_drop_entire_channel_dict[side_key] = should_drop_entire_channel
                                else:
                                    assert (
                                        side_key in should_drop_entire_channel_dict
                                    ), "side channel key should be in the should_drop_entire_channel_dict"
                                    should_drop_entire_channel = should_drop_entire_channel_dict[side_key]

                        if should_drop_entire_channel:
                            side_channel_output_tmp[side_key] = side_channel_output[side_key] * 0

                fusion_output[fusion_net_inp_key] = dec(
                    encoder_output_dict, side_channel_output_tmp, enc_input_shape=enc_input_shape
                )

        # returns the final output (after going through encoder, decoder and all that) and also the combined output of the side_channel networks
        return fusion_output, side_channel_output, (should_drop_indices_dict, should_drop_entire_channel_dict)
