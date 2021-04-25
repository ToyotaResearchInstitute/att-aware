import torch
import collections

from chm.model.gaze_transform import GazeTransform
from chm.model.FusionNet import create_fusion_net
from chm.model.CHMPredictorNet import create_chm_predictor


def create_identity_gaze_transform(scale_factor=1.0):
    """
    create_identity_gaze_transform

    Parameters:
    ---------
    scale_factor: float
        Affine scaling used by the transform

    Returns:
    -------
    gaze_transform: torch.nn.Module
        gaze_transform torch module used for transforming input gaze in CognitiveHeatNeat
    """
    gaze_transform = GazeTransform(scale_factor=scale_factor)
    return gaze_transform


class CognitiveHeatNet(torch.nn.Module):
    def __init__(self, params_dict):
        """
        Model class encapsulating Fusion Net (encoder, decoder, side-channel) and MapPredictor networks for CHM

        Parameters
        ----------
        params_dict : dict
            dict containing args for network structure
        """
        super().__init__()
        self.params_dict = params_dict

        self.add_optic_flow = self.params_dict.get("add_optic_flow", False)
        self.num_latent_layers = self.params_dict.get("num_latent_layers", 6)
        self.aspect_ratio_reduction_factor = self.params_dict.get("aspect_ration_reduction_factor", 8.0)

        self.ORIG_ROAD_IMG_DIMS = self.params_dict.get("orig_road_image_dims")
        self.ORIG_ROAD_IMAGE_HEIGHT = self.ORIG_ROAD_IMG_DIMS[1]
        self.ORIG_ROAD_IMAGE_WIDTH = self.ORIG_ROAD_IMG_DIMS[2]
        self.NETWORK_OUT_SIZE = [
            self.num_latent_layers,
            int(round(self.ORIG_ROAD_IMAGE_HEIGHT / self.aspect_ratio_reduction_factor)),
            int(round(self.ORIG_ROAD_IMAGE_WIDTH / self.aspect_ratio_reduction_factor)),
        ]

        self.NETWORK_OUT_HEIGHT = self.NETWORK_OUT_SIZE[1]
        self.NETWORK_OUT_WIDTH = self.NETWORK_OUT_SIZE[2]
        # create an identity gaze transform module.
        self.gaze_transform = create_identity_gaze_transform()

        self.fusion_net = create_fusion_net(
            params_dict=self.params_dict,
            network_out_height=self.NETWORK_OUT_HEIGHT,
            network_out_width=self.NETWORK_OUT_WIDTH,
        )

        # this is the dict containing the output dimensions of all outputs from the decoder.
        chm_predictor_input_dim_dict = self.fusion_net.output_dims

        # create image cognitive predictor (heatmap), predictor_input_dim is the output size (64, 224, 224) of the decoder
        # output_img_size is the size of the heatmap. 1 channel heatmap
        self.chm_predictor = create_chm_predictor(
            chm_predictor_input_dim_dict=chm_predictor_input_dim_dict, predictor_output_size=self.NETWORK_OUT_SIZE
        )

    def get_modules(self):
        """
        Get individual modules in the network
        """
        road_facing_network = self.fusion_net.map_modules["road_facing"]
        fusion_net_network = self.fusion_net
        gaze_transform_prior_loss = self.gaze_transform.prior_loss

        return fusion_net_network, road_facing_network, gaze_transform_prior_loss

    def forward(self, batch_input, should_drop_indices_dict=None, should_drop_entire_channel_dict=None):
        """
        Parameters:
        ----------
        batch_input: dict
            dict containing various input tensors.
        should_drop_indices_dict: dict or None
            if not None, dict containing booleans flag for which batch item is dropped in the side channel
        should_drop_entire_channel_dict: dict or None
            if not None, dict with flags indicating which entire side channel was dropped out

        Returns:
        --------

        """
        road_image = batch_input["road_image"] / 255.0  # normalized road image
        normalized_input_gaze = batch_input["normalized_input_gaze"].clone().detach()  # (B, T, L, 2)
        if "no_detach_gaze" in batch_input:
            if batch_input["no_detach_gaze"]:
                # don't detach gaze. Used during calibration optimization training
                normalized_input_gaze = batch_input["normalized_input_gaze"]  # (B, T, L, 2)

        should_train_input_gaze = batch_input["should_train_input_gaze"].clone().detach()  # (B, T, L, 1)

        try:
            # (B, T, L, 4)
            normalized_transformed_gaze = self.gaze_transform(normalized_input_gaze, should_train_input_gaze)
        except:
            import IPython

            IPython.embed(header="normalized input gaze invalid")

        # prepare fusion net input
        fusion_net_input = collections.OrderedDict()

        # main encoder input
        fusion_net_input["road_facing"] = road_image

        # side channel input
        fusion_net_input["driver_facing"] = normalized_transformed_gaze
        if self.add_optic_flow:
            fusion_net_input["optic_flow"] = batch_input["optic_flow_image"].clone().detach()

        # check for nans in side channel. can happen if side channel module blows up during training
        if torch.isnan(fusion_net_input["driver_facing"]).sum():
            import IPython

            print(fusion_net_input["driver_facing"])
            IPython.embed(banner1="CHM::Forward, fusion_net_input[driver_facing] has nans")

        (decoder_output, side_channel_output, should_drop_dicts) = self.fusion_net(
            fusion_net_input, should_drop_indices_dict, should_drop_entire_channel_dict
        )

        # predictor input
        chm_predictor_input = decoder_output["road_facing"]
        if (torch.isnan(chm_predictor_input)).sum():  # check if there is any nans in the encoder-decoder pass
            import IPython

            print(chm_predictor_input)
            IPython.embed(header="CHM:forward. Output of the decoder decoder_output[0][road_facing] has nans")

        # gaze and awareness density maps
        chm_output = self.chm_predictor(chm_predictor_input)

        return chm_output, decoder_output, side_channel_output, should_drop_dicts
