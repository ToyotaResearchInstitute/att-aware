import torch

from chm.model.DecoderUnit import DecoderUnit


def create_decoder(decoder_net_params=None):
    assert decoder_net_params is not None

    use_s3d = decoder_net_params["use_s3d"]
    decoder_layer_features = decoder_net_params["decoder_layer_features"]
    output_dim_of_decoder = decoder_net_params["output_dim_of_decoder"]
    side_channel_input_dim = decoder_net_params["side_channel_input_dim"]
    skip_layers = decoder_net_params["skip_layers"]

    decoder_layers = torch.nn.ModuleDict()

    # note that the layer names go backwards. Because the decoder layers mirror the encoder layers
    if not use_s3d:
        decoder_layers["layer4"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-1],  # 128
            output_dim=decoder_layer_features[-2],  # 64
            skip_dim=skip_layers[-1],  # 0
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )  # innermost DU. skip_layers[-1] = 0
        decoder_layers["layer3"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-2],  # 64
            output_dim=decoder_layer_features[-3],  # 32
            skip_dim=skip_layers[-2],  # 256
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
        decoder_layers["layer2"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-3],  # 32
            output_dim=decoder_layer_features[-4],  # 16
            skip_dim=skip_layers[-3],  # 32
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
        decoder_layers["layer1"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-4],  # 16
            output_dim=output_dim_of_decoder,  # 16
            skip_dim=skip_layers[-4],  # 16
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
    else:
        decoder_layers["s3d_net_3"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-1],  # 256 #
            output_dim=decoder_layer_features[-2],  # 128
            skip_dim=skip_layers[-1],  # 0, no skip connections for S3D
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
        decoder_layers["s3d_net_2"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-2],  # 128
            output_dim=decoder_layer_features[-3],  # 64
            skip_dim=skip_layers[-2],  # 0
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
        decoder_layers["s3d_net_1"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-3],  # 64
            output_dim=decoder_layer_features[-4],  # 32
            skip_dim=skip_layers[-3],  # 0
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
        decoder_layers["layer2"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-4],  # 32
            output_dim=decoder_layer_features[-5],  # 16
            skip_dim=skip_layers[-4],  # 32
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )
        decoder_layers["layer1"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-5],  # 16
            output_dim=output_dim_of_decoder,  # 16
            skip_dim=skip_layers[-5],  # 16
            side_channel_input_dim=side_channel_input_dim,
            use_s3d=use_s3d,
        )

    # keys indicating the source of skip connections from the encoder.
    if not use_s3d:
        skip_layers_keys = ["layer3", "layer2", "layer1"]
    else:
        skip_layers_keys = ["layer2", "layer1"]

    return DecoderNet(
        decoder_layers=decoder_layers,
        skip_layers_keys=skip_layers_keys,
    )


class DecoderNet(torch.nn.Module):
    def __init__(self, decoder_layers, skip_layers_keys):
        super().__init__()
        self.decoder_layers = decoder_layers  # these are the DecoderUnits.
        self.skip_layers_keys = skip_layers_keys
        self.upsm = torch.nn.Upsample(mode="bilinear")
        import IPython

        IPython.embed(banner1="check in Decoder Net init")

    def forward(self, input, side_channel_input, enc_input_shape):
        pass