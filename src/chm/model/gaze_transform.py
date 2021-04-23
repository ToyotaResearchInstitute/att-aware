import torch
import numpy as np


def compute_inverted_affine_tform(linear_mtx, trans_vec):
    # for  y = Ax + b ---> x = A^-1( y - b) --> A' = A^-1 and b' = -(A^-1)*b
    result_mtx = np.linalg.pinv(linear_mtx)
    result_vec = -np.dot(result_mtx, trans_vec)
    return (result_mtx, result_vec)


class GazeTransform(torch.nn.Module):
    def __init__(self, scale_factor=1.0, linear_transform=None, translation=None, pad_gaze_vector=True):
        """
        A module to capture the transformation of the gaze. This should be used either to correct the transformation,
        or to corrupt it.
        Parameters
        ----------
        scale_factor: float

        linear_transform:

        translation:

        pad_gaze_vector: bool
            Bool indicating whether the should_train_bit and the dropout bit should be added to the gaze tensor

        """
        super().__init__()
        # the weight matrix associated with network is a 2 by 2 matrix since gaze points are
        #  always in 2D. - set the weight to be scale_factor on the diagonal +noise for the whole weight matrix / bias
        self.lin_trans = torch.nn.Linear(2, 2)
        self.lin_trans.weight.data.normal_(mean=0, std=0.00001)
        self.lin_trans.bias.data.normal_(mean=0, std=0.00001)
        self.scale_factor = scale_factor
        self.linear_transform = linear_transform
        self.translation = translation
        assert not (
            (linear_transform is None) ^ (translation is None)
        ), "linear_transform and the translation vector should be either both None or both instantiated"

        if not (linear_transform is None):
            assert tuple(self.lin_trans.weight.shape) == linear_transform.shape
            assert tuple(self.lin_trans.bias.shape) == translation.shape
            self.lin_trans.weight = torch.nn.Parameter(self.lin_trans.weight.new_tensor(linear_transform))
            self.lin_trans.bias = torch.nn.Parameter(self.lin_trans.bias.new_tensor(translation))
        else:
            for i in range(2):
                self.lin_trans.weight.data[i][i] += scale_factor

        self.pad_gaze_vector = pad_gaze_vector

    def forward(self, input, should_train_input_gaze=False):
        """
        input dimension = (B, T, L, 2). L is the number of gaze points per frame
        output dimension = (B, T, L, 4 or 2), affine transformed (representing miscalibration) gaze points.
        """

        # Change from (B, T, L, 2) ---> (B, TL, 2)
        input_reshaped = input.reshape(input.shape[0], input.shape[1] * input.shape[2], -1)

        output_reshaped = self.lin_trans(input_reshaped)  # (B, TL, 2)
        output = output_reshaped.reshape(input.shape[0], input.shape[1], input.shape[2], -1)  # (B, T, L, 2)
        # adds a ones-all element to the last dim, to state the gaze input is there
        if self.pad_gaze_vector:
            # return (B, T, L, 4)
            return torch.cat([output, should_train_input_gaze, output.new_ones(output[:, :, :, 0:1].shape)], dim=3)
        else:
            # return (B, T, L, 2)
            return output

    def prior_loss(self):
        gaze_transform_loss = 0.0
        if not (self.linear_transform is None):
            for i in range(2):
                gaze_transform_loss += (self.lin_trans.bias.data[i] - self.translation[i]) ** 2
                for j in range(2):
                    mean = self.linear_transform[i][j]
                    gaze_transform_loss += (self.lin_trans.weight.data[i][j] - mean) ** 2

        else:
            for i in range(2):
                gaze_transform_loss += (self.lin_trans.bias.data[i]) ** 2
                for j in range(2):
                    mean = self.scale_factor if i == j else 0
                    gaze_transform_loss += (self.lin_trans.weight.data[i][j] - mean) ** 2
        return gaze_transform_loss
