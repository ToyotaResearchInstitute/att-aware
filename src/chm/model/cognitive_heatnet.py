import torch

from chm.model.gaze_transform import GazeTransform
from chm.model.FusionNet import create_enc_dec_backbone


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
        Model class encapsulating encoder, decoder, side-channel and predictor networks for CHM

        Parameters
        ----------
        depth_net : dict
            dict containing args for network structure
        """
        super().__init__()
        self.params_dict = params_dict

        self.add_optic_flow = self.params_dict.get("add_optic_flow", False)
        self.num_latent_layers = self.params_dict.get("num_latent_layers", 6)
        self.aspect_ratio_reduction_factor = self.params_dict.get("aspect_ration_reduction_factor", 8.0)

        self.ORIG_ROAD_IMG_DIMS = self.params_dict.get("orig_road_img_dims")
        self.ORIG_ROAD_IMAGE_HEIGHT = self.ORIG_ROAD_IMG_DIMS[1]
        self.ORIG_ROAD_IMAGE_WIDTH = self.ORIG_ROAD_IMG_DIMS[2]
        self.NETWORK_OUT_SIZE = [
            num_latent_layers,
            int(round(self.ORIG_ROAD_IMAGE_HEIGHT / aspect_ratio_reduction_factor)),
            int(round(self.ORIG_ROAD_IMAGE_WIDTH / aspect_ratio_reduction_factor)),
        ]

        self.NETWORK_OUT_HEIGHT = self.NETWORK_OUT_SIZE[1]
        self.NETWORK_OUT_WIDTH = self.NETWORK_OUT_SIZE[2]
        # create an identity gaze transform module.
        self.gaze_transform = create_identity_gaze_transform()

        self.enc_dec_backbone = create_enc_dec_backbone(
            params_dict=self.params_dict
            network_out_height=self.NETWORK_OUT_HEIGHT,
            network_out_weight=self.NETWORK_OUT_WIDTH,
        )

        # this is the dict containing the output dimensions of all outputs from the decoder.
        chm_predictor_input_dim_dict = self.enc_dec_backbone.output_dims

        # create image cognitive predictor (heatmap), predictor_input_dim is the output size (64, 224, 224) of the decoderoutput_img_size is the size of the heatmap. 1 channel heatmap
        self.chm_predictor = create_predictor_fn(chm_predictor_input_dim_dict, self.NETWORK_OUT_SIZE)
