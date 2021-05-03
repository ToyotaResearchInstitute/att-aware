import torch
import numpy as np
from chm.model.gaze_transform import GazeTransform


class GazeCorruption:
    def __init__(
        self, bias_std, noise_std, transforms=None, x_weight_factor=1.0, y_weight_factor=1.3, is_spatially_varying=False
    ):
        """
        Parameters:
        ----------
        bias_std: float
            The amount of bias in the Gaussian noise
        noise_std: float
            The standard deviation of the Gaussian noise
        transforms: list or None
            List of GazeTransforms to be applied before adding noise
        x_weight_factor: float
            Weight factor for x dimension when using spatially varying noise
        y_weight_factor: float
            Weight factor for y dimension when using spatially varying noise
        is_spatially_varying: bool
            Bool indicating whether the noise adding is spatially varying

        """
        self.bias_std = bias_std
        self.noise_std = noise_std
        self.is_spatially_varying = is_spatially_varying
        self.x_weight_factor = x_weight_factor
        self.y_weight_factor = y_weight_factor
        self.added_noise = np.random.normal(0, self.bias_std, [1, 2])  # fixed zero mean noise added
        if transforms is not None and type(transforms) is not list:
            transforms = [transforms]
        self.transforms = transforms

    def corrupt_gaze(self, gaze):
        """
        Corrupts the gaze tensor according the noise parameters.

        Parameters
        ----------
        gaze: torch.Tensor
            Gaze tensor to be corrupted (B, T, L, 2)

        Returns
        -------
        corrupted_gaze: torch.Tensor
            Corrupted gaze (B, T, L, 2)
        """
        if self.transforms is not None:
            for transform in self.transforms:
                # :2 so that the dropout indicator is not incorporated
                gaze = transform(gaze)[:, :, :, :2]  # (B, T, L, 2)

        if not self.is_spatially_varying:
            corrupted_gaze = (
                gaze
                + gaze.clone().normal_() * self.noise_std
                + gaze.new_tensor(
                    np.tile(
                        np.expand_dims(np.expand_dims(self.added_noise, 0), 0),
                        [gaze.shape[0], gaze.shape[1], gaze.shape[2], 1],
                    )
                )
            )
        else:
            weight_factor = torch.ones_like(gaze)  # (B, T, L, 2)
            weight_factor[:, :, :, 0] = weight_factor[:, :, :, 0] * self.x_weight_factor
            weight_factor[:, :, :, 1] = weight_factor[:, :, :, 1] * self.y_weight_factor
            # (B, T, L, 2) gaze is already normalized gaze. Therefore subtract 0.5 from each dimension computes the offset from the center
            diff_from_center = gaze - 0.5 * torch.ones_like(gaze)
            weighted_diff_from_center = diff_from_center * weight_factor  # (B, T, L, 2)
            weighted_dist_from_center = weighted_diff_from_center ** 2  # (B, T, L, 2) dx^2, dy^2

            # (B, T, L, 2)
            corrupted_gaze = (
                gaze
                + gaze.clone().normal_() * self.noise_std
                + gaze.clone().normal_() * weighted_dist_from_center
                + gaze.new_tensor(
                    np.tile(
                        np.expand_dims(np.expand_dims(self.added_noise, 0), 0),
                        [gaze.shape[0], gaze.shape[1], gaze.shape[2], 1],
                    )
                )
            )
        return corrupted_gaze
