# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch
import numpy as np

from abc import ABC, abstractmethod


class AwarenessEstimator(ABC):
    def __init__(self, road_images=None, gaze_inputs=None):
        """
        Base class for different types of awareness estimation. Examples of different implementations:
        - using Gaussian filtering, smear the gaze inputs in space/time
        - using Gaussian filtering, smear the gaze inputs in space/time leveraging optic flow information

        Parameters:
        ----------
        road_images: torch.Tensor
            Tensor containing the road images used as input

        gaze_inputs: torch.Tensor
            Tensor containing the gaze inputs to the network
        """
        self.road_images = road_images
        self.gaze_inputs = gaze_inputs

    @abstractmethod
    def estimate_awareness(self, frame, coordinate, additional_info=None):
        """
        Given the inputs, estimates awareness for a given coordinate set at a specific frame
        Derived class contains the implementation
        Parameters:
        ----------
        frame: int
            Frame index at which the awareness estimation need to be performed

        coordinate: tuple
            (x, y) coordinate in frame at which the awareness estimation need to be performed

        additional_info=None or valid data type
            Any additional info to be used for estimating awareness. Derived class implementation ensure checks for correctness
        """
        return None


class SpatioTemporalGaussianWithOpticFlowAwarenessEstimator(AwarenessEstimator):
    """
    Strawman baseline for awareness estimation using optic flow based spatio temporal gaussian filter.
    """

    def __init__(
        self,
        gaze_inputs,
        should_train_bit_sequence,
        optic_flow_sequence,
        sigma_kernel=4,
        spatial_scale=0.005,
        temporal_scale=0.01,
        temporal_decay=0.8,
        temporal_filter_type="exponential",
    ):
        """
        Parameters:
        -----------
        gaze_inputs: torch.Tensor
            Tensor containing the gaze inputs to the network
        should_train_bit_sequence: torch.Tensor
            Tensor containing validity bits for the gaze inputs
        optic_flow_sequence: torch.Tensor
            Tensor containing the optic flow images for the sequence
        sigma_kernel: int
            Kernel size for the gaussian filter in pixel
        spatial_scale: float
            Scale factor for the spatial component of the spatiotemporal gaussian
        temporal_scale: float
            Scale factor for the temporal component of the spatiotemporal gaussian
        temporal_filter_type: str
            String indicating the type of temporal filtering to be done. Either 'exponential' or 'geometric'
        """
        super().__init__(None, gaze_inputs)
        self.should_train_bit_sequence = should_train_bit_sequence
        self.optic_flow_sequence = optic_flow_sequence
        self.temporal_scale = temporal_scale
        self.temporal_decay = temporal_decay
        self.spatial_scale = spatial_scale
        self.sigma_kernel = sigma_kernel
        self.temporal_filter_type = temporal_filter_type

    def compute_spatio_temporal_filter_withOF_on_sequence(self):
        """
        Uses optic flow to compute the gaussian filtered spatio temporal volume for the given sequence

        Returns:
        --------
        st_filtered_awareness_sequence_overall: torch.Tensor (T, H, W)
            Tensor containing the spatio temporally smoothed gaussian filter based awareness estimate
        """
        # Grab the temporal and spatial dimensions of the OF map
        T = self.optic_flow_sequence.shape[0]
        H = self.optic_flow_sequence.shape[-2]
        W = self.optic_flow_sequence.shape[-1]

        # empty tensor to accumulate all the spatio temporal tubes. 'st' in variable names refers to SpatioTemporal
        st_filtered_awareness_sequence_overall = torch.zeros(T, H, W)

        N = int(self.spatial_scale)  # sigma of gaussian in pixels
        sig = self.sigma_kernel
        for outer_t in range(T):  # [0, 1, .....,T-1]
            # if invalid gaze point continue
            if self.should_train_bit_sequence[outer_t] == 0.0:
                continue

            # (T, H, W) #the first 0....outer_t wil remain zero
            st_filtered_awareness_sequence = torch.zeros(T, H, W)
            x = int(self.gaze_inputs[outer_t, 0])  # x value of gaze
            y = int(self.gaze_inputs[outer_t, 1])  # y value of gaze

            # create the gausian kernel, taking into account the edge artifacts
            shift_x = 0
            shift_y = 0
            shift_x = max(0, int(-(x - N)))
            min_x = x - max(0, x - N)
            max_x = 2 * N - min_x + 1
            # if not near left edge, check if close to right edge
            if shift_x == 0:
                shift_x = min(0, -(int(x + N) - (W - 1)))  # negative value
                min_x = x - max(0, x - N) - shift_x
                max_x = 2 * N - min_x + 1

            # check if gaze is close to top edge
            shift_y = max(0, int(-(y - N)))
            min_y = y - max(0, y - N)
            max_y = 2 * N - min_y + 1
            # if not near top edge, check for near bottom edge
            if shift_y == 0:
                shift_y = min(0, -(int(y + N) - (H - 1)))  # negative value
                min_y = y - max(0, y - N) - shift_y
                max_y = 2 * N - min_y + 1

            kernel_x, kernel_y = np.meshgrid(range(-N + shift_x, N + 1 + shift_x), range(-N + shift_y, N + 1 + shift_y))

            # create the shifted gaussian kernel
            kernel = np.exp(-(kernel_x ** 2 + kernel_y ** 2) / (sig ** 2))

            kernel = self.optic_flow_sequence.new_tensor(kernel / np.sum(kernel))
            min_x = int(min_x)
            max_x = int(max_x)
            min_y = int(min_y)
            max_y = int(max_y)

            # awareness of frame outer_t
            # (same size as kernel)
            st_filtered_awareness_sequence[outer_t, y - min_y : y + max_y, x - min_x : x + max_x] = kernel / torch.max(
                kernel
            )

            # coordinate grid. xv is the row indices, yv is the column indices
            xv, yv = torch.meshgrid(
                [
                    self.optic_flow_sequence.new_tensor(torch.arange(0, H).clone().detach().numpy()),
                    self.optic_flow_sequence.new_tensor(torch.arange(0, W).clone().detach().numpy()),
                ]
            )

            # (H, W) init the t-1 heatmap
            inner_tm1_heatmap = st_filtered_awareness_sequence[outer_t, :, :]

            # deal with the temporal propagation according to optic flow
            # inner loop controls ALL those frames that will be affected by the gaze at outer_t
            for inner_t in range(outer_t + 1, T):
                # using the optic flow information figure out where the heat is going to be propagated
                # since optic flow is computed backward. That is,the optic flow at outer_t+1 can be used to compute where the flow originated in the previous frame.
                # (2, H, W)
                optic_flow_at_outer_tp1_ux_uy = self.optic_flow_sequence[inner_t, :, :, :]
                # displaced row indices
                xv_new_p = torch.floor(xv + torch.floor(optic_flow_at_outer_tp1_ux_uy[1, :, :]))
                xv_new_p[xv_new_p < 0] = 0.0  # limit the index to stay within bounds
                xv_new_p[xv_new_p >= H] = H - 1

                # displaced column indices
                yv_new_p = torch.floor(yv + torch.floor(optic_flow_at_outer_tp1_ux_uy[0, :, :]))
                yv_new_p[yv_new_p < 0] = 0.0
                yv_new_p[yv_new_p >= W] = W - 1

                # (grab pixels from tm1 according to the OF) (H, W)
                inner_t_heatmap = inner_tm1_heatmap[xv_new_p.flatten().long(), yv_new_p.flatten().long()].reshape(
                    inner_tm1_heatmap.shape[0], inner_tm1_heatmap.shape[1]
                )

                # apply temporal decay
                # this is in terms of frames
                if self.temporal_filter_type == "exponential":
                    temporal_weight = np.exp(-((inner_t - outer_t) ** 2) / self.temporal_scale ** 2)
                elif self.temporal_filter_type == "geometric":
                    temporal_weight = (self.temporal_decay) ** (inner_t - outer_t)

                inner_t_heatmap = temporal_weight * inner_t_heatmap  # (H, W)

                # update the block
                st_filtered_awareness_sequence[inner_t, :, :] = inner_t_heatmap
                inner_tm1_heatmap = inner_t_heatmap  # recursively do this.

            st_filtered_awareness_sequence_overall += st_filtered_awareness_sequence

        return st_filtered_awareness_sequence_overall  # (T, H, W)

    def estimate_awareness(self, frame, coordinate, additional_info=None):
        """
        Parameters:
        ----------
        frame: int
            Frame index at which the awareness estimation need to be performed
        coordinate: tuple
            (x, y) coordinate in frame at which the awareness estimation need to be performed
        additional_info=None or valid data type
            Any additional info to be used for estimating awareness. Derived class implementation ensure checks for correctness

        Returns:
        --------
        st_filtered_awareness_sequence_overall: torch.Tensor
            (T, H, W). SpatioTemporally filtered awareness maps
        estimate: float
            Awareness estimate at frame, coordinate
        """
        st_filtered_awareness_sequence_overall = self.compute_spatio_temporal_filter_withOF_on_sequence()  # (T, H, W)
        assert frame < st_filtered_awareness_sequence_overall.shape[0]
        estimate = (
            st_filtered_awareness_sequence_overall[frame, coordinate[1].long(), coordinate[0].long()].cpu().numpy()
        )
        estimate = np.minimum(1, np.maximum(estimate, 0))
        return st_filtered_awareness_sequence_overall, estimate
