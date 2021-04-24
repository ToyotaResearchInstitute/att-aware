import collections

import numpy as np
import torch
import torchvision.models

from cognitive_heatmap.regularizations import EPSpatialRegularization, EPTemporalRegularization, SmoothnessOperator
from cognitive_heatmap.awareness_label_loss import AwarenessPointwiseLabelLoss, TYPE_HUBER_LOSS, TYPE_SQUARED_ERROR
from cognitive_heatmap.chm_eyelink_dreyeve_dataset import EYELINK_DREYEVE_TASK_TYPE_ENUMS
from cognitive_heatmap.gaze_transform import GazeTransform
from model_zoo.intent.SimpleNet import SimpleNet


class GazeLSTM(torch.nn.Module):
    def __init__(self, hidden_dim, gaze_list_length, lstm_input_dim, lstm_fc_output_dim):
        """
        hidden_dim = output size of the LSTMCell. The hidden state of the last (depth-wise) LSTM Cell is taken as the output of the fully connected layer
        gaze_list_length = number of gaze points in the gaze list
        lstm_input_dim = input dimension of the LSTM cell
        lstm_fc_output_dim = output dimension of the fully connected layer
        """
        super(GazeLSTM, self).__init__()

        # init params
        self.hidden_dim = hidden_dim
        self.gaze_list_length = gaze_list_length
        self.lstm_input_dim = lstm_input_dim
        self.lstm_fc_output_dim = lstm_fc_output_dim

        self.gaze_lstm1 = torch.nn.LSTMCell(input_size=self.lstm_input_dim, hidden_size=self.hidden_dim)
        self.fc = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.lstm_fc_output_dim)

    def forward(self, input):
        """
        input-shape = B x T x L x 2. Where B is the batch, T is the sequence_length, L is the length of the gaze list for each timestep, and 2 is the dimension of the gaze (x, y)
        returns the output seq containing the lstm output from each gaze point in the gaze_list
        """
        original_batch_size = input.shape[0]  # B
        time_slice_length = input.shape[1]  # T
        gaze_list_length = input.shape[2]  # L

        input = input.view(original_batch_size * time_slice_length, gaze_list_length, -1)  # (B, T, L, 2) --> (BT, L, 2)
        gaze_batch_size = input.shape[0]
        # tensor to hold the intermediate outputs
        output_seq = input.new_tensor(torch.empty((gaze_list_length, gaze_batch_size, self.lstm_fc_output_dim)))
        h = input.new_tensor(torch.zeros(gaze_batch_size, self.hidden_dim))  # hidden and cell state initialization
        c = input.new_tensor(torch.zeros(gaze_batch_size, self.hidden_dim))
        for l in range(gaze_list_length):  # step through each gaze point and feed it into LSTM
            gaze_l = input[:, l, :]  # grab the l^th gaze point from gaze list. (BT, 2)
            h, c = self.gaze_lstm1(gaze_l, (h, c))
            output_seq[l] = self.fc(h)

        output_seq = output_seq.permute(1, 0, 2)  # (L, BT, 2) --> (BT, L, 2)
        # (BT, L, 2) --> (B, T, L, 2)
        output_seq = output_seq.reshape(original_batch_size, time_slice_length, gaze_list_length, -1)
        return output_seq

    def init_hidden(self):
        return (torch.zeros(self.batch_size, self.hidden_dim), torch.zeros(self.batch_size, self.hidden_dim))


def create_gaze_list_lstm(hidden_dim, gaze_list_length, lstm_input_dim, lstm_fc_output_dim):
    res = GazeLSTM(hidden_dim, gaze_list_length, lstm_input_dim, lstm_fc_output_dim)
    return res


def create_identity_gaze_transform(scale_factor=1.0):
    """
    create_identity_gaze_transform
    :param scale_factor: the expected scaling factor.
    :return: returns the gaze transform function and a prior on it
    """
    res_tfm = GazeTransform(scale_factor=scale_factor)
    return res_tfm


def run_over_images(input, net, axis):
    # todo: replace with efficient convolutions without concats
    res = []
    for i in range(input.shape[axis]):  # looping over time dimension?
        # (16 4 3 h w) --> (16 1 3 h w) --> (16 3 h w) --> (16 64 h w)[processed] --> 16 1 3 h/scaled w/scaled [unsqueezed]
        # for the entire bacth grab the ith image by using index_select, remove the time dimension using squeeze, pass it through a network that expects a batch of images, and then finally add back the removed dimension
        tmp = net(torch.index_select(input, dim=axis, index=input.new_tensor([i]).long()).squeeze(axis)).unsqueeze(axis)
        res.append(tmp)
    # import IPython;IPython.embed(header='run_over_images')
    # concatenate the results along wthe seuqnece dimension. #so its back to B * T * C * H * W
    res = torch.cat(res, axis)
    return res


class EncoderNet(torch.nn.Module):
    def __init__(
        self, preproc, layered_outputs, s3d_layers=None, post_layer=None, post_layer_bn=None, post_layer_nonlin=None
    ):
        super().__init__()
        self.preproc = preproc
        self.layered_outputs = layered_outputs
        self.s3d_layers = s3d_layers
        self.post_layer = post_layer
        self.post_layer_bn = post_layer_bn
        self.post_layer_nonlin = post_layer_nonlin

    def forward(self, input):
        """
        Encode an input (or a sequence of input) image(s) via a set of convolutions and subsamples, save the intermediate outputs for shortcut
        connections.
        :param input: input (or sequence of) images
        :return: a dictionary containing all the intermediate results of applying fwd() on the image or batch of images
        """
        # TODO: replace with better balanced image coloring, for better matching to pretrained Resnet assumptions.
        intermediate = run_over_images(input, self.preproc, axis=1)
        res = collections.OrderedDict()

        for key in self.layered_outputs:  # keys are layer1, layer2, layer3, layer4
            res[key] = run_over_images(intermediate, self.layered_outputs[key], axis=1)  # .detach()
            intermediate = res[key]

        if self.s3d_layers is not None:
            for key in self.s3d_layers:
                # B, T, C, H, W --> B, C, T, H, W
                intermediate = intermediate.permute(0, 2, 1, 3, 4)
                out = self.s3d_layers[key](intermediate)  # (B, C, T, H, W)
                res[key] = out.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
                intermediate = res[key]

        if self.post_layer is not None:
            # from [B,T,C,H,W] to [B,C,T,H,W] for 3D conv, and back.
            intermediate_reduced = self.post_layer_nonlin(
                self.post_layer_bn(self.post_layer(res[key].transpose(1, 2)))
            ).transpose(1, 2)
            res[key] = intermediate_reduced
        return res


def create_encoder(reduced_layer_size=None, use_relu=True, use_s3d_encoder=False):

    orig_net = torchvision.models.resnet18(pretrained=True)  # use resnet
    # initial layers of resnet for preprocessing
    preproc_net = torch.nn.Sequential()
    preproc_net.add_module("conv1", orig_net.conv1)
    preproc_net.add_module("bn1", orig_net.bn1)
    preproc_net.add_module("relu1", orig_net.relu)
    preproc_net.add_module("maxpool", orig_net.maxpool)

    # encoder layers from resnet for 2d convs. Present for both nonS3D and S3D architectures
    layered_net = torch.nn.ModuleDict()
    layered_net["layer1"] = orig_net.layer1
    layered_net["layer2"] = orig_net.layer2
    if not use_s3d_encoder:
        layered_net["layer3"] = orig_net.layer3
        layered_net["layer4"] = orig_net.layer4
        s3d_net = None
    else:
        s3d_net = torch.nn.ModuleDict()
        # the input dim for s3d net should match the output of layer2 of resnet.
        s3d_net["s3d_net_1"] = STConv3d(
            in_channels=list(orig_net.layer2.children())[-1].bn2.num_features,
            # currenly hard coding the sizes of the layers. #TODO (deepak.gopinath) make it an args
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            use_relu=use_relu,
        )
        s3d_net["s3d_net_2"] = STConv3d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, use_relu=use_relu
        )
        s3d_net["s3d_net_3"] = STConv3d(
            in_channels=512,
            out_channels=512,  # potentially change this to 1024?
            kernel_size=3,
            stride=1,
            padding=1,
            use_relu=use_relu,
        )

    post_processing = None
    post_processing_bn = None
    post_processing_nonlin = None
    if reduced_layer_size is not None:
        if not use_s3d_encoder:
            orig_latent_dim = list(orig_net.layer4.children())[-1].conv2.out_channels
        else:
            orig_latent_dim = 512  # out_channels in s3d_net['s3d_net_3']

        new_latent_dim = reduced_layer_size
        post_processing = torch.nn.Conv3d(orig_latent_dim, new_latent_dim, 1)
        post_processing_bn = torch.nn.InstanceNorm3d(new_latent_dim)
        post_processing_nonlin = torch.nn.ReLU()
        # post_processing_nonlin = torch.nn.Tanh()  # maybe Sigmoid so that the output is between 0 and 1.0?

    return EncoderNet(
        preproc=preproc_net,
        layered_outputs=layered_net,
        s3d_layers=s3d_net,
        post_layer=post_processing,
        post_layer_bn=post_processing_bn,
        post_layer_nonlin=post_processing_nonlin,
    )


class DecoderNet(torch.nn.Module):
    def __init__(
        self,
        postproc,
        layered_outputs,
        skip_layers_keys=["layer3", "layer2", "layer1"],
        full_scale=[1.0, 1.0],
        optic_flow_downsample_mode="max",
    ):
        super().__init__()
        self.postproc = postproc
        self.layered_outputs = layered_outputs  # these are the DecoderUnits.
        self.skip_layers_keys = skip_layers_keys
        self.upsm = torch.nn.Upsample(mode="bilinear")
        self.full_scale = full_scale
        self.optic_flow_downsample_mode = optic_flow_downsample_mode

    def forward(self, input, side_channel_input, enc_input_shape):
        """
        Decode an output image via a deconvolutional + side information model
        :param input: the image input to the decoder network. includes the shortcut connections information
        :param side_channel_input: the side information available to the network,
        :return:
        """
        all_keys = list(self.layered_outputs.keys())
        upsm_target_size_list = [[input[x].shape[3], input[x].shape[4]] for x in all_keys]
        upsm_target_size_list.append([int(round(enc_input_shape[3] / 2)), int(round(enc_input_shape[4] / 2))])
        # layer4, layer3, layer2, layer1 in that order. so [0] is layer4
        key = list(self.layered_outputs)[0]
        # tensor. input is a dictionary that contains all the intermediate calculations from the encoder net indexed as layer 1 2 3 4. with key=layer4 we grab the endmost outuput of the encoder and put it in current
        last_out = input[key]
        intermediates = [last_out]  #
        # keys of the decoder units go from layer4, layer3, layer2, layer1? or s3d_net_2, s3d_net_1, layer2, layer1
        for i, key in enumerate(self.layered_outputs):
            int_input = None
            int_inputs = []
            # (H,W) pairs are for aspect aspect_ratio_reduction_factor of 8
            # layer4 = (5, 8), layer3 =
            H = input[key].shape[-2]
            W = input[key].shape[-1]
            if side_channel_input is not None:
                for side_key in side_channel_input:
                    if side_key == "driver_facing":
                        # side_channel_input is a tensor of sahpe (B, T, L, 4) when the driver_facing module only has linear0 module.
                        # (B, T, L, 1)
                        dropout_indicator = side_channel_input[side_key][:, :, :, 3:]
                        # (B, T, L, 1)
                        should_train_input_gaze_indicator = side_channel_input[side_key][:, :, :, 2:3]

                        # (B, T, 1) Since the dropout is applied for each item in the batch, the value at 0th position is the same
                        dropout_indicator = dropout_indicator[:, :, 0, :]
                        # (B,T, L, 2). Grab only the gaze points and ignore the dropout indicator
                        gaze_points = side_channel_input[side_key][:, :, :, 0:2]

                        # Create meshgrid containing normalized coordinates for computing dx and dy at each pixel to the nearest gaze point
                        yv, xv = torch.meshgrid(
                            [
                                torch.linspace(
                                    start=0.0, end=1.0, steps=input[key].shape[3], device=gaze_points.device
                                ),
                                torch.linspace(
                                    start=0.0, end=1.0, steps=input[key].shape[4], device=gaze_points.device
                                ),
                            ]
                        )  # normalized coordinates along each dimension
                        yv = yv.unsqueeze(dim=2)  # ( H, W, 1)
                        xv = xv.unsqueeze(dim=2)  # (H, W, 1)
                        coord_grid = torch.cat([xv, yv], dim=2)  # (H, W, 2)

                        # Create a tensor to hold the dxdy to nearest gaze point side channel information. Consists of two channels. one for dx and one for dy.
                        # gaze_side_channel_layer = torch.zeros(gaze_points.shape[0], gaze_points.shape[1], coord_grid.shape[0], coord_grid.shape[1], 2, device=gaze_points.device) #(B,T,H,W,2), C=2 To be permuted later into B, T, C, H, W where C = 2. First channel contains dx information and second channel contains dy information
                        gp_list_grid = (
                            gaze_points.unsqueeze(2)
                            .unsqueeze(2)
                            .repeat(1, 1, coord_grid.shape[0], coord_grid.shape[1], 1, 1)
                        )  # (B,T,H,W,L,2)
                        coord_grid_list = (
                            coord_grid.unsqueeze(2)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .repeat(gaze_points.shape[0], gaze_points.shape[1], 1, 1, gaze_points.shape[2], 1)
                        )  # (B, T, H, W, L 2)
                        dx_dy_grid_gaze_list = coord_grid_list - gp_list_grid  # (B, T, H, W, L, 2)
                        gp_list_distance_grid = torch.norm(
                            dx_dy_grid_gaze_list, dim=-1
                        )  # (B, T, H, W, L) #compute distances to all the L gaze points.
                        # (B, T, H, W) #compute the index of gaze point which is of shortest distance to a particular coordinate point.

                        # since the "invalid" gaze points were kept at -10,-10, these points will never be the "smallest" distance gaze point, if there are other valid gaze points in the list.In the case all L points are invalid,
                        # then the multiplication (later) on with the train bit will take care of zeroing out the entire frame
                        (min_distances, min_indices) = torch.min(gp_list_distance_grid, dim=-1)
                        # (B, T, H, W, 1) #this is the index corresponding to which one of the L'th gaze point is closest to a particular coordinate
                        min_indices = min_indices.unsqueeze(-1)
                        # (B, T, H, W, L). Initialize the one-hot tensor with zeros
                        min_indices_one_hot = torch.zeros_like(
                            dx_dy_grid_gaze_list[:, :, :, :, :, 0]
                        )  # the one hot vector for each coordinate position will be L dimensional. This is picking the proper gaze index for dx dy coordinate corresponding to the shortest distance gaze point
                        # (B, T, H, W, L) #scatter 1 at the specified indices along the L dimension of min_indices_one_hot
                        min_indices_one_hot.scatter_(4, min_indices, 1)
                        # (B, T, H, W, L, 2). Make one hot tensor same size as dx_dy_grid_gaze_list
                        min_indices_one_hot = min_indices_one_hot.unsqueeze(-1).repeat(
                            1, 1, 1, 1, 1, dx_dy_grid_gaze_list.shape[-1]
                        )  # so that both dx and dy can be selected using the same index
                        # (B, T, H, W, L, 2) Note that only one of L gaze points will be nonzero. Therefore, go ahead and sum up along L dimension
                        gaze_side_channel_layer = dx_dy_grid_gaze_list * min_indices_one_hot

                        gaze_side_channel_layer = torch.sum(
                            gaze_side_channel_layer, 4
                        )  # (B, T, H, W, 2) #dx,dy values to the nearest gaze point.
                        int_input = gaze_side_channel_layer.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W) C=2

                        dropout_indicator = (
                            dropout_indicator.unsqueeze(-1)
                            .unsqueeze(-1)
                            .repeat(1, 1, 1, int_input.shape[3], int_input.shape[4])
                        )  # (B, T, 1, H, W) #add dimension at the end for H and W.

                        # (B, T, H, W, L, 1)
                        min_indices_one_hot_single_dim = min_indices_one_hot[:, :, :, :, :, 1:]
                        # (B, T, H, W, L, 1)
                        should_train_input_gaze_indicator = (
                            should_train_input_gaze_indicator.unsqueeze(2)
                            .unsqueeze(2)
                            .repeat(
                                1,
                                1,
                                min_indices_one_hot_single_dim.shape[2],
                                min_indices_one_hot_single_dim.shape[3],
                                1,
                                1,
                            )
                        )

                        should_train_input_gaze_indicator = torch.sum(
                            should_train_input_gaze_indicator * min_indices_one_hot_single_dim, 4
                        )  # (B, T, H, W, 1)

                        should_train_input_gaze_indicator = should_train_input_gaze_indicator.permute(
                            0, 1, 4, 2, 3
                        )  # (B, T, 1, H, W)

                        st_mult = should_train_input_gaze_indicator.repeat(
                            1, 1, int_input.shape[2], 1, 1
                        )  # (B,T,2,H,W)

                        # should_train_input_gaze_indicator = should_train_input_gaze_indicator * \
                        #     dropout_indicator  # (B, T, 1, H, W)
                        # (B, T, C, H, W) with C=2, same as int_input's channel dimension
                        di_mult = dropout_indicator.repeat(1, 1, int_input.shape[2], 1, 1)
                        # this multiplication with the di_mult will make sure that the dxdy layers that are dropped out are all zero'ed out.

                        # (B, T, C, H, W) C=2 #only pass through when should train and dropout indicator are true.
                        int_input = int_input * di_mult * st_mult
                        min_distances = min_distances.unsqueeze(2)  # (B, T, C=1, H, W)
                        # (B, T,1,H,W)
                        min_dx_layer_sq = ((gaze_side_channel_layer[:, :, :, :, 0:1]) ** 2).permute(0, 1, 4, 2, 3)
                        min_dx_layer_sq = min_dx_layer_sq * dropout_indicator * should_train_input_gaze_indicator
                        # (B, T, 1,H, W)
                        min_dy_layer_sq = ((gaze_side_channel_layer[:, :, :, :, 1:]) ** 2).permute(0, 1, 4, 2, 3)
                        min_dy_layer_sq = min_dy_layer_sq * dropout_indicator * should_train_input_gaze_indicator
                        # (B, T, 1,H, W)
                        min_dx_dy_layer = (
                            gaze_side_channel_layer[:, :, :, :, 0:1] * gaze_side_channel_layer[:, :, :, :, 1:]
                        ).permute(0, 1, 4, 2, 3)
                        min_dx_dy_layer = min_dx_dy_layer * dropout_indicator * should_train_input_gaze_indicator
                        int_input = torch.cat(
                            [
                                int_input,
                                min_dx_layer_sq,
                                min_dy_layer_sq,
                                min_dx_dy_layer,
                                min_distances,
                                should_train_input_gaze_indicator,
                                dropout_indicator,
                            ],
                            dim=2,
                        )  # (B, T, 2+1+1+1+1+1+1, H, W) C=8
                        int_inputs.append(int_input)

                    elif side_key == "task_network":
                        # if side_key is task C = 5. Because tasks are represented as one hot vectors of dim 5.
                        side_inp = side_channel_input[side_key]  # (B, T, C=5)
                        # (B, T, C, H, W)
                        # add last two dimensions and expand it to match the height and width
                        int_input = side_inp.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, -1, H, W])
                        int_inputs.append(int_input)

                    elif side_key == "optic_flow":
                        side_inp = side_channel_input[side_key]  # (B, T, C=2, H, W)
                        if self.optic_flow_downsample_mode == "max":  # TODO change this to an enum
                            # (B, T, H, W) #magnitude of optic flow tensor
                            mag_tensor = torch.norm(side_inp, dim=2)
                            # (B, T, 1, H, W) #add channel dimension
                            mag_tensor = mag_tensor.unsqueeze(2)
                            # (BT, C=1, H, W)
                            mag_tensor = mag_tensor.view(
                                -1, mag_tensor.shape[2], mag_tensor.shape[3], mag_tensor.shape[4]
                            )  # magnitude tensor, combine B and T for AdaptiveMaxPool2d
                            amp = torch.nn.AdaptiveMaxPool2d((H, W), return_indices=True)
                            # todo apply padding before max pooling. however unsure why it would matter when using AdapativeMaxPool.
                            _, max_inds = amp(mag_tensor)  # max_inds (BT, 1, h, w)
                            max_inds = max_inds.view(
                                max_inds.shape[0], max_inds.shape[1], -1
                            )  # (BT, 1, hw), indices corresponding to max magnitude locations
                            # (BT, 2, hw) #repeat along channel dimensions. To be used to extarct the ux and uy at the max_ind locations
                            max_inds = max_inds.repeat([1, 2, 1])

                            # (BT, C=2, HW) (ux, uy)
                            side_inp = side_inp.view(-1, side_inp.shape[2], side_inp.shape[3] * side_inp.shape[4])
                            # (BT, C=2, hw)
                            # grab the ux, uy vals corresponding to the max pool on magnitude tensor
                            low_res_side_inp = torch.gather(side_inp, 2, max_inds)
                            # reshape low_res_side_inp
                            low_res_side_inp = low_res_side_inp.reshape(
                                side_channel_input[side_key].shape[0], side_channel_input[side_key].shape[1], -1, H, W
                            )  # (B, T, C=2, h, w)

                        elif self.optic_flow_downsample_mode == "avg":
                            # (BT, C=2, H, W)
                            side_inp = side_inp.view(-1, side_inp.shape[2], side_inp.shape[3], side_inp.shape[4])

                            aap = torch.nn.AdaptiveAvgPool2d((H, W))
                            # todo add padding before
                            low_res_side_inp = aap(side_inp)  # (BT, C=2, h, w)
                            low_res_side_inp = low_res_side_inp.reshape(
                                side_channel_input[side_key].shape[0], side_channel_input[side_key].shape[1], -1, H, W
                            )  # (B, T, C=2, h, w)

                        elif self.optic_flow_downsample_mode == "median":
                            # zero pad to nearest multiple of the target h and w before applying the median filter
                            import IPython

                            IPython.embed(banner1="to be implemented")
                            # directly feed the downsampled optic flow images
                        int_inputs.append(low_res_side_inp)

                    # print(int_input.shape[2])

            # import IPython;IPython.embed(header='side_inp')
            int_inputs = torch.cat(int_inputs, dim=2)
            # int inputs - side channel gaze input layers
            # last_out - output of the previous DecoderUnit. For the first DecoderUnit this is same the encoderoutput for layer4.
            # input[key] - skip connection
            if (
                not key in self.skip_layers_keys
            ):  # skip the skip connection for layers that are not specified for skip connection
                new_out = self.layered_outputs[key](
                    last_out, None, int_inputs, upsm_size=upsm_target_size_list[i + 1]
                )  # Forward of decoder unit
            else:
                # input[key] IS the skip connection
                new_out = self.layered_outputs[key](
                    last_out, input[key], int_inputs, upsm_size=upsm_target_size_list[i + 1]
                )
            intermediates.append(new_out)
            last_out = new_out

        # at this stage last_out is the output of decoder_unit named layer1.
        self.upsm.size = [enc_input_shape[3], enc_input_shape[4]]
        last_out2 = self.upsm(
            last_out.reshape(
                [last_out.shape[0] * last_out.shape[1], last_out.shape[2], last_out.shape[3], last_out.shape[4]]
            )
        )  # upsampling is done on tensor in which B and T are combined.
        res = self.postproc(last_out2)  # nothing happens now as the postproc container is empty
        # decoupled B and T dimensions using reshape.
        res2 = res.reshape(last_out.shape[0], last_out.shape[1], last_out.shape[2], res.shape[2], res.shape[3])
        return res2


class STConv3d(torch.nn.Module):
    """
    Adapted from https://github.com/qijiezhao/s3d.pytorch/blob/master/S3DG_Pytorch.py

    S3D structure used in encoder and decoder. Consists of separate spatial and temporal Conv3D with (1, k, k) and (k, 1,1) kernels.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, use_relu=False):
        super(STConv3d, self).__init__()
        self.replication_pad1 = torch.nn.ReplicationPad3d(
            (0, 0, padding, padding, padding, padding)
        )  # for spatial conv only pad spatial dim
        self.conv1 = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(0, 0, 0),
        )  # spatial conv (1, k, k)
        self.replication_pad2 = torch.nn.ReplicationPad3d(
            (padding, padding, 0, 0, 0, 0)
        )  # for temp dim only pad temporal dim
        self.conv2 = torch.nn.Conv3d(
            out_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1), padding=(0, 0, 0)
        )  # temporal conv (k, 1, 1)
        self.in1 = torch.nn.InstanceNorm3d(out_channels)
        self.in2 = torch.nn.InstanceNorm3d(out_channels)
        if not use_relu:
            self.nonlin1 = torch.nn.Tanh()
            self.nonlin2 = torch.nn.Tanh()
        else:
            self.nonlin1 = torch.nn.ReLU()
            self.nonlin2 = torch.nn.ReLU()

    def forward(self, x):
        # assumes x is (B, C, T, H, W)
        x = self.nonlin1(self.in1(self.conv1(self.replication_pad1(x))))
        x = self.nonlin2(self.in2(self.conv2(self.replication_pad2(x))))
        return x


class DecoderUnit(torch.nn.Module):
    def __init__(self, last_out_dim, skip_dim, output_dim, side_channel_input_dim, use_relu=False, use_separable=False):

        super().__init__()
        self.use_separable = use_separable
        self.replication_pad = torch.nn.ReplicationPad3d(1)
        if not self.use_separable:
            self.net1 = torch.nn.Conv3d(
                in_channels=last_out_dim + skip_dim + side_channel_input_dim,
                out_channels=output_dim,
                kernel_size=(3, 3, 3),
                padding=(0, 0, 0),
                stride=(1, 1, 1),
            )
            self.bn = torch.nn.InstanceNorm3d(output_dim)
            if not use_relu:
                self.nonlin = torch.nn.Tanh()
            else:
                self.nonlin = torch.nn.ReLU()
        else:
            self.sep_conv3d = STConv3d(
                in_channels=last_out_dim + skip_dim + side_channel_input_dim,
                out_channels=output_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                use_relu=use_relu,
            )

        self.upsm = torch.nn.Upsample(mode="bilinear")
        # conv3d module to process the skip connection if there are present so that the time dimension is properly convoluted
        if skip_dim != 0:
            if not self.use_separable:
                self.skip_net = torch.nn.Conv3d(
                    in_channels=skip_dim,
                    out_channels=skip_dim,
                    kernel_size=(3, 3, 3),
                    padding=(0, 0, 0),
                    stride=(1, 1, 1),
                )
                self.skip_bn = torch.nn.InstanceNorm3d(skip_dim)
                if not use_relu:
                    self.skip_nonlin = torch.nn.Tanh()
                else:
                    self.skip_nonlin = torch.nn.ReLU()
            else:
                self.skip_net_sep_conv3d = STConv3d(
                    in_channels=skip_dim, out_channels=skip_dim, kernel_size=3, stride=1, padding=1, use_relu=use_relu
                )

        # self.nonlin = torch.nn.ReLU()

    def forward(self, last_out, skip_input, side_channel_input, upsm_size):
        # skip dim = 0. This is only for the first decoder unit
        # (the one which directly takes input from the LAST encoder layer.
        # Skip dim is kept to zero because skip connection and the last
        # encoder output are the same and we don't want duplication)
        # import IPython
        # IPython.embed(banner1="check side_channel_input to DU")
        if skip_input is None:
            joint_input = torch.cat(
                # side_channel_input is of size (B, T, 4 + 5 + 1 +1+1, H, W) - Voronoi maps + dropout +train_indicator+ task + dx2 , dy2, dxdy, optic flow
                [last_out, side_channel_input],
                2,
            )
        else:
            # process the temporal aspect of skip connection first.
            skip_input = skip_input.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            # replication pad so that after conv3d the size remains the same.
            if not self.use_separable:
                skip_input = self.replication_pad(skip_input)
                skip_input = self.skip_nonlin(self.skip_bn(self.skip_net(skip_input)))
            else:
                skip_input = self.skip_net_sep_conv3d(skip_input)

            skip_input = skip_input.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
            joint_input = torch.cat(
                [last_out, skip_input, side_channel_input], 2
            )  # concatenate along the channel dimension. (B, T, C, H, W)
        joint_input = joint_input.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        if not self.use_separable:
            # (B, C, T+2, H+2, W+2), increased T, H, W dimensions due to replication padding. on both sides.
            joint_input = self.replication_pad(joint_input)
            # (B, C', T, H, W) C' is the output dim of the conv3d
            res = self.nonlin(self.bn(self.net1(joint_input)))
        else:
            # (B, C', T, H, W)
            res = self.sep_conv3d(joint_input)

        # (BC', T, H, W) for applying upsm on H and W, the tensor has to be 4D
        res = res.view(res.shape[0] * res.shape[1], res.shape[2], res.shape[3], -1)
        self.upsm.size = upsm_size  # set the upsm size
        res = self.upsm(res)  # (BC', T, H', W')
        # (B, C, T, H', W'). Use B from joint_input and infer C dimension
        res = res.reshape(joint_input.shape[0], -1, res.shape[1], res.shape[2], res.shape[3])
        # (B, T, C', H', W') So that the skip connection from encoder can be added properly along C, H', W' dimensions for subsequent decoder layer
        res = res.permute(0, 2, 1, 3, 4)
        return res


# #The following is legacy code. With 2d Convolutions.
# class DecoderUnit(torch.nn.Module):
#     def __init__(self, input_dim, skip_dim, output_dim,
#                  intermediate_input_dim):
#         super().__init__()

#         self.normalization = torch.nn.BatchNorm2d(input_dim + skip_dim + intermediate_input_dim)
#         self.net1 = torch.nn.Conv2d(
#             in_channels=input_dim + skip_dim + intermediate_input_dim,
#             out_channels=output_dim,
#             kernel_size=[3, 3],
#             stride=1,
#             padding=1)
#         self.upsm = torch.nn.Upsample(mode='bilinear')
#         self.nonlin = torch.nn.Tanh()
#         # self.nonlin=torch.nn.ReLU()

#     def forward(self, last_out, skip_input, side_channel_input, upsm_size):
#         '''
#         :param last_out: Bx
#         :param skip_input:
#         :return:
#         '''
#         if (side_channel_input is not None):
#             joint_input = torch.cat(
#                 [last_out, skip_input, side_channel_input],
#                 2)  #concatenate along the channel dimension.
#         else:
#             joint_input = torch.cat([last_out, skip_input], 2)

#         # import IPython; IPython.embed(banner1='check intermediate medium. encoder + gaze info ')

#         #coalesce batch and time dimension. no more temporal information retained when joint_input.shape[0] * joint_input.shape[1]
#         res = self.nonlin(self.net1(
#                 self.normalization(joint_input.view(
#                         joint_input.shape[0] * joint_input.shape[1],
#                         joint_input.shape[2], joint_input.shape[3],
#                         joint_input.shape[4])))
#         )  #maybe the order of batch norm and net1 is flipped? first apply net1 on input then batch norm before applying the nonlinearity?
#         self.upsm.size = upsm_size
#         res = self.upsm(res)
#         res = res.reshape(joint_input.shape[0], joint_input.shape[1],
#                           res.shape[1], res.shape[2],
#                           res.shape[3])  #back to B * T * C * H * W
#         return res


def create_decoder(
    num_units,
    output_dim_of_decoder,
    decoder_layer_features,
    skip_layers,
    intermediate_input_dim,
    image_size,
    optic_flow_downsample_mode,
    use_relu=False,
    use_separable=False,
    use_s3d_encoder=False,
):
    postproc = torch.nn.Sequential()  # this is just a container.
    layered_outputs = torch.nn.ModuleDict()

    # decoder_layer_features = [16, 32, 64, 128]
    # skip lyaers =  [64, 128, 256, 0]
    # intermediate_input_dim =  sum of all channels of side information
    # output_dim_of_decoder = 16
    if not use_s3d_encoder:
        layered_outputs["layer4"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-1],  # 128
            output_dim=decoder_layer_features[-2],  # 64
            skip_dim=skip_layers[-1],  # 0
            side_channel_input_dim=intermediate_input_dim,
            use_relu=use_relu,
            use_separable=use_separable,
        )  # innermost DU. skip_layers[-1] = 0
        layered_outputs["layer3"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-2],  # 64
            output_dim=decoder_layer_features[-3],  # 32
            skip_dim=skip_layers[-2],  # 256
            side_channel_input_dim=intermediate_input_dim,
            use_relu=use_relu,
            use_separable=use_separable,
        )
        layered_outputs["layer2"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-3],  # 32
            output_dim=decoder_layer_features[-4],  # 16
            skip_dim=skip_layers[-3],  # 32
            side_channel_input_dim=intermediate_input_dim,
            use_relu=use_relu,
            use_separable=use_separable,
        )
        layered_outputs["layer1"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-4],  # 16
            output_dim=output_dim_of_decoder,  # 16
            skip_dim=skip_layers[-4],  # 16
            side_channel_input_dim=intermediate_input_dim,
            use_relu=use_relu,
            use_separable=use_separable,
        )
    else:

        layered_outputs["s3d_net_3"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-1],  # 128 #
            output_dim=decoder_layer_features[-2],  # 64
            skip_dim=skip_layers[-1],  # 0
            side_channel_input_dim=intermediate_input_dim,
            use_relu=use_relu,
            use_separable=use_separable,
        )  # innermost DU. skip_layers[-1] = 0

        layered_outputs["s3d_net_2"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-2],  # 128
            output_dim=decoder_layer_features[-3],  # 64
            skip_dim=skip_layers[-2],  # 0
            side_channel_input_dim=intermediate_input_dim,
            use_relu=use_relu,
            use_separable=use_separable,
        )

        layered_outputs["s3d_net_1"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-3],  # 64
            output_dim=decoder_layer_features[-4],  # 32
            skip_dim=skip_layers[-3],  # 0
            side_channel_input_dim=intermediate_input_dim,
            use_relu=use_relu,
            use_separable=use_separable,
        )

        layered_outputs["layer2"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-4],  # 32
            output_dim=decoder_layer_features[-5],  # 16
            skip_dim=skip_layers[-4],  # 32
            side_channel_input_dim=intermediate_input_dim,
            use_relu=use_relu,
            use_separable=use_separable,
        )

        layered_outputs["layer1"] = DecoderUnit(
            last_out_dim=decoder_layer_features[-5],  # 16
            output_dim=output_dim_of_decoder,  # 16
            skip_dim=skip_layers[-5],  # 16
            side_channel_input_dim=intermediate_input_dim,
            use_relu=use_relu,
            use_separable=use_separable,
        )

    if not use_s3d_encoder:
        skip_layers_keys = ["layer3", "layer2", "layer1"]
    else:
        skip_layers_keys = ["layer2", "layer1"]

    return DecoderNet(
        postproc=postproc,
        layered_outputs=layered_outputs,
        skip_layers_keys=skip_layers_keys,
        optic_flow_downsample_mode=optic_flow_downsample_mode,
    )  # since input gaze is normalized, the full_scale parameters is set to 1.0


class FusionNet(torch.nn.Module):
    """
    FusionNet is a class that combines the different network modules (road facing, driver facing, decoder)
    """

    def __init__(
        self,
        children_networks,
        domain_dict,
        output_dims,
        add_task_net,
        add_optic_flow,
        dropout_ratio,
        dropout_ratio_external_inputs,
    ):
        """
        children_networks: side channel networks providing aux information to the decoder units (driver_facing, task_network etc)
        domain_dict: Encoder-Decoder structure for road facing camera.
        output_dims: output dims of the road facing decoder (C, H, W) of the latent layer before the gaze and awareness heatmap
        add_task_net: flag indicating if the fusion net has to include the task side channel network or not.
        dropout_ratio: dictionary containing dropout factors for each side channel child network. The factor is used randomly zero out certain elements in a batch
        dropout_ratio_external_inputs: scalar determining if ALL of the children_networks should be dropped out during a forward pass. Applied after dropout of elements in batch is applied.
        """
        super().__init__()
        self.children_networks = children_networks
        self.domain_dict = domain_dict
        self.output_dims = output_dims
        self.dropout_ratio = dropout_ratio
        # dropout ratio for gaze and other context inputs
        self.add_task_net = add_task_net
        self.add_optic_flow = add_optic_flow
        self.dropout_ratio_external_inputs = dropout_ratio_external_inputs
        # TODO: add a different parameter for different inputs, such as gaze
        # if key is in the input, use it to override the input's dropout. The key corresonds to the name of the child networks. driver facing, task etc
        self.force_input_dropout = {}

    def forward(self, input, should_drop_indices_dict=None, should_drop_entire_channel_dict=None):
        intermediate_products = []  # list containing the outputs of each child networks
        has_lstm = False

        # Process side channel information
        for key in input:
            # such as 'driver_facing'. so far it is only 'driver_facing' (input gaze for the time being)
            if key in self.children_networks:
                # for driver_facing, input[key] is (B,T,L,4) #last dimension is [x,y,should_train_indicator, droput_indicator] The input to driver facing module comes from the GazeTransform module
                # for task_network, input[key] is (B,T,1,num_tasks) or (B, T, num_tasks, 1)
                # for optic flow input[key] is (B,T, 2, H, W)
                inp = input[key]
                # print('child activation: '+str(key))
                if key == "driver_facing":
                    out = inp
                    for subkey in self.children_networks[key]:  # linear and lstm
                        child_net = self.children_networks[key][subkey]
                        if child_net is not None:
                            if subkey == "linear0":
                                inp_reshaped = inp.reshape(inp.shape[0], inp.shape[1] * inp.shape[2], -1)  # (B, TL, 4)
                                out_reshaped = child_net(inp_reshaped)  # (B, TL,4)
                                out = out_reshaped.reshape(inp.shape[0], inp.shape[1], inp.shape[2], -1)  # (B, T, L, 4)
                            # this is not used now as we are avoiding LSTMs to process side channel gaze input.
                            elif subkey == "lstm0":
                                has_lstm = True
                                # This is the GazeLSTM module. Input is output of linear0 layer.(B, T, L, 4) -- > (B, T, L, f), where f (=2 for the time being) is the output dimensionality of the LSTM
                                out_seq = child_net(out)
                                # (B, T, f) Grab the last output of the lstm output sequence
                                out = out_seq[:, :, -1, :]
                elif key == "task_network":
                    # import IPython;IPython.embed(header='recheck dimensionalities')
                    # (B,T,num_tasks,1)->#(B,T,1,num_tasks), not sure why is there 1 to begin with
                    out = inp.transpose(2, 3)  # (B, T, 1, NUMTASKS)
                    for subkey in self.children_networks[key]:  # linear and lstm
                        child_net = self.children_networks[key][subkey]
                        if child_net is not None:
                            out = child_net(out)  # for any other types of child network modules

                    # (B,T,output_dim of the task embedding network) if task embedding not None, else (B, T, NUM TASKS)
                    out = out[:, :, -1, :]
                    # out = out.transpose(2, 3)
                # append the (output, name of childnetwork)
                # collect all the side channel outputs. Corresponding to driver_facing, task_network
                elif key == "optic_flow":
                    out = inp  # (B,T,C,H,W)
                    for subkey in self.children_networks[key]:
                        child_net = self.children_networks[key][subkey]
                        if child_net is not None:
                            out = child_net(out)

                intermediate_products.append((out, key))
        if has_lstm:
            intermediate_list = [x[0].clone().view(x[0].shape[0], x[0].shape[1], -1) for x in intermediate_products]
        else:
            intermediate_list = [x[0] for x in intermediate_products]

        # names of the child networks that produced the intermediate list
        intermediate_list_keys = [x[1] for x in intermediate_products]
        # For each type of child network output choose to dropout or not some of the data_items in the batch
        for i, (x, x_key) in enumerate(zip(intermediate_list, intermediate_list_keys)):
            # creates a vector of size BATCH_SIZE (x.shape[0]) of bools indicating whether the intermediate result should be dropped or not?
            # x_key is driver_facing, task_network etc. THis is the "dict" passed from the command line
            assert x_key in self.dropout_ratio
            # each side channel input has its own dropout ratio, as specified in the dropout_ratio dict
            if (
                should_drop_indices_dict is None
            ):  # if not should_drop_indices_dict was explicitly passed to the forward function
                # determines which of the batch items should be dropped out
                should_drop = torch.rand([x.shape[0]]) < self.dropout_ratio[x_key]
                should_drop_indices_dict = collections.OrderedDict()
                should_drop_indices_dict[x_key] = should_drop
            else:
                if (
                    should_drop_indices_dict is not None
                ):  # could happen because a partial should_drop_indices_dict was passed or it was created in the previous if block
                    if x_key not in should_drop_indices_dict:
                        should_drop = torch.rand([x.shape[0]]) < self.dropout_ratio[x_key]
                        should_drop_indices_dict[x_key] = should_drop
                    else:
                        assert (
                            x_key in should_drop_indices_dict
                        ), "the key for the side channel should be present in the indices dict"
                        should_drop = should_drop_indices_dict[x_key]

            if not (self.training):  # If it is testing override the rand vector created and 0 it out
                should_drop = should_drop * 0

            if (
                x_key in self.force_input_dropout
            ):  # this should be the same keys as the child networks, such as driever_facing, task_network etc. Also doesn't make sense to loop over all keys. Only use the key corresponding to the ith element in the intermediate_list
                for b in range(x.shape[0]):
                    should_drop[b] = self.force_input_dropout[x_key]
            # import IPython;IPython.embed(header='force input, self.training='+str(self.training)+', self.force_input_dropout = '+str(self.force_input_dropout))
            for b in range(x.shape[0]):  # possible vectorized way to do this?
                if should_drop[b]:
                    x[b, :] *= 0  # is mutable. Therefore elements in intermediate list will be changed

            # why should we be concatanating outputs from different children networks? that too along time dimension
            # import IPython;IPython.embed(header='concat')
            # potential issue here? Has the out in intermediate products been properly changed after applying dropout?
            side_channel_output = {key: out for out, key in intermediate_products}
            # print(side_channel_output, side_channel_output.shape)
            # TODO (deepak.gopinath). Might need some sort of feature adapter for concatenating combining outputs of different children networks
            # I think the concatenation will not work as it is iif there are multiple types of children output. For example
            # Child_output_0 - B x T x 2 (for gaze input)
            # Child_output_1 - B x T x 3 x 224 x 224 ( for driver facing camera image)
            # These can't be concatenated as above.
        # else:  #if in testing mode.
        #     side_channel_output = torch.cat
        #         [
        #             x.view(x.shape[0], x.shape[1], -1)
        #             for x in intermediate_products
        #         ],
        #         dim=1)  # same concatenation issue here.
        for key in side_channel_output:
            # if any of the intermediate outputs are nans STOP!
            if torch.isnan(side_channel_output[key]).sum():
                import IPython

                print("Intermediate output", side_channel_output)
                print("Input", input)
                IPython.embed(
                    header="Fusion::forward - side_channel_output (concatenated outputs of the child networks) has nans"
                )

        res = collections.OrderedDict()
        # Now process the image based input using the encoder-decoder structure.
        for key in input:
            # such as 'road facing' #every domain module is supposed to have an encode-decoder structure.
            # TODO(deepak.gopinath) rename self.domain_dict to something more meaningful.
            if key in self.domain_dict:
                # The intermediate output from the child networks are fed alongside the output of the encoder as the input to decoder for EACH domain module separately.
                enc = self.domain_dict[key]["encoder"]
                dec = self.domain_dict[key]["decoder"]

                input_shape = input[key].shape  # needed for dimension matching for skip connections
                # output of the encoder network as a dictionary containing the intermediates results of layer1,2,3,4
                encoder_output_dict = enc(input[key])
                # output of the child networks [driver facing/gaze] as a dictionary containing outputs of different side channel networks.
                # with some of the samples in the batch dimension already zero'ed out.
                # If force_dropout is true then all of it is zeroed out
                side_channel_output2 = side_channel_output
                if self.training:
                    # to determine if the whole child network output should be dropped off or not.
                    for side_key in side_channel_output2:
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
                            side_channel_output2[side_key] = side_channel_output[side_key] * 0
                    # else:

                # import IPython;IPython.embed(header='side_channel_output2 = side_channel_output')
                # encoder_outpout_dict is a dictionary containing all of the intermediate results from passing the input through the encoder indexed by the layers in the encoder. side_channel_output2 is the output (as a dict) from the child networks with dropout already applied (if dropout_ratio > 0.0)
                res[key] = dec(encoder_output_dict, side_channel_output2, enc_input_shape=input_shape)

        # returns the final output (after going through encoder, decoder and all that) and also the combined output of the child networks
        return res, side_channel_output, (should_drop_indices_dict, should_drop_entire_channel_dict)


def create_simple_cognitive_map(dropout_ratio=0.1, intermediate_reduced=128):
    # create driver-facing module - currently just gaze
    driver_facing = torch.nn.Linear(2, 2)
    # import IPython;
    # IPython.embed(header='create_identity_gaze_transform')
    driver_facing.weight.data.normal_(mean=0, std=0.01)
    driver_facing.bias.data.normal_(mean=0, std=0.01)
    for i in range(2):
        driver_facing.weight.data[i][i] += 1.0

    # Create road-facing module. For now, assume image-to-image
    # road_facing=torchvision.models.resnet18(pretrained=True) #unused.
    road_facing_encoder = create_encoder(reduced_layer_size=intermediate_reduced)
    output_dim_of_decoder = 64
    decoder_layer_features = [64, 128, 256, 512]

    skip_layers = [64, 128, 256, 0]
    # skip_layers = [64, 128, 0, 0]  # skip connections only for layer 1 and layer 2
    decoder_layer_features[-1] = intermediate_reduced

    road_facing_decoder = create_decoder(
        num_units=4,
        output_dim_of_decoder=output_dim_of_decoder,
        decoder_layer_features=decoder_layer_features,
        skip_layers=skip_layers,
        intermediate_input_dim=2,
    )
    child_modules = torch.nn.ModuleDict()
    child_modules["driver_facing"] = driver_facing

    map_modules = torch.nn.ModuleDict()
    map_modules["road_facing"] = torch.nn.ModuleDict()
    map_modules["road_facing"]["encoder"] = road_facing_encoder
    map_modules["road_facing"]["decoder"] = road_facing_decoder
    output_dims = collections.OrderedDict()
    output_dims["road_facing"] = [64, 224, 224]
    res = FusionNet(child_modules, map_modules, output_dims, dropout_ratio=dropout_ratio)
    return res  # seems like this is exactly the same as create_image_cognitive_map()


def create_image_cognitive_map(
    add_task_net=False,
    add_optic_flow=False,
    optic_flow_downsample_mode="max",
    gaze_lstm_hidden_dim=2,
    gaze_list_length=5,
    gaze_transform_output_dim=4,
    gaze_lstm_fc_output_dim=2,
    dropout_ratio_external_inputs=0.0,
    dropout_ratio={"driver_facing": 0.01, "task_network": 0.0},
    decoder_layer_features=[64, 128, 256, 512],
    road_image_height=1080,
    road_image_width=1920,
    task_output_dim=5,
    optic_flow_output_dim=2,
    intermediate_reduced=128,
    use_relu=False,
    use_separable=False,
    use_s3d_encoder=False,
):
    # create driver-facing module - currently just gaze. COULD HAVE IMAGE LATER From a driver facing camera etc
    # looks like the driver_facing is currently an identity transform.
    # input to the linear layer of driever facing module is the output from GazeTransform which is (B, T, L, 3)
    # num_tasks = len(EYELINK_DREYEVE_TASK_TYPE_ENUMS.keys())
    if not use_s3d_encoder:
        assert len(decoder_layer_features) == 4
    else:
        assert len(decoder_layer_features) == 5
    # this has to match the output channel size of layer4 from the ResNet.
    # assert(decoder_layer_features[-1] == 512)
    # driver_facing_linear = torch.nn.Linear(gaze_transform_output_dim, gaze_transform_output_dim)
    # driver_facing_linear.weight.data.normal_(mean=0, std=0.000001)
    # driver_facing_linear.bias.data.normal_(mean=0, std=0.000001)
    ROAD_IMAGE_WIDTH = road_image_width
    ROAD_IMAGE_HEIGHT = road_image_height
    # for i in range(gaze_transform_output_dim):
    #     driver_facing_linear.weight.data[i][i] += 1.0

    # LSTM module not used for the time being. Replaced with voronoi maps that pass dx and dy info as side channels to the decoder units.
    # driver_facing_lstm = create_gaze_list_lstm(hidden_dim=gaze_lstm_hidden_dim, gaze_list_length=gaze_list_length,
    #                                            lstm_input_dim=gaze_transform_output_dim, lstm_fc_output_dim=gaze_lstm_fc_output_dim)

    # Create road-facing module. For now, assume image-to-image
    # road_facing=torchvision.models.resnet18(pretrained=True) #not used here?
    road_facing_encoder = create_encoder(reduced_layer_size=intermediate_reduced, use_s3d_encoder=use_s3d_encoder)
    # input to the cognitive predictor (more like the output size of the decoder that should match the input size of the cognitive predictor)
    output_dim_of_decoder = decoder_layer_features[0]
    # num_of_features in the decoder units. [Last...last-1,..,first]
    # decoder_layer_features = [64, 128, 256, 512]
    # last skip dim is 0 or else the layer4 of encoder output will be passed twice to the DecoderUnit4
    # this comes from the encoder channel sizes. Fixed by resnet layer1, layer2, layer3, layer4 architecture
    # skip_layers = [64, 128, 256, 0]
    if not use_s3d_encoder:
        skip_layers = [64, 128, 256, 0]
    else:
        # skip connections only for layer 1 and layer 2, s3d_net_1, s3d_net_2, s3d_net_3 don't have skip connections
        skip_layers = [64, 128, 0, 0, 0]
    # intermediate_reduced is the output_num_channels of the 3d conv after the last the encoder layer and before the first decoder
    decoder_layer_features[-1] = intermediate_reduced
    dx_sq_layer_dim = 1
    dy_sq_layer_dim = 1
    dx_times_dy_layer_dim = 1
    dxdy_dist_layer_dim = 1
    if use_s3d_encoder:
        num_decoder_units = 5
    else:
        num_decoder_units = 4
    if add_task_net:
        if add_optic_flow:
            road_facing_decoder = create_decoder(
                num_units=num_decoder_units,
                output_dim_of_decoder=output_dim_of_decoder,
                decoder_layer_features=decoder_layer_features,
                skip_layers=skip_layers,
                intermediate_input_dim=gaze_transform_output_dim
                + task_output_dim
                + dx_sq_layer_dim
                + dy_sq_layer_dim
                + dx_times_dy_layer_dim
                + dxdy_dist_layer_dim
                + optic_flow_output_dim,
                image_size=[ROAD_IMAGE_WIDTH, ROAD_IMAGE_HEIGHT],
                optic_flow_downsample_mode=optic_flow_downsample_mode,
                use_relu=use_relu,
                use_separable=use_separable,
                use_s3d_encoder=use_s3d_encoder,
            )  # channels for (
            # dx, dy, dropout indicator) + task_output_dim + dx^2 layer, dy^2  layer, dx*dy layer, optic flow output dim
        else:
            road_facing_decoder = create_decoder(
                num_units=num_decoder_units,
                output_dim_of_decoder=output_dim_of_decoder,
                decoder_layer_features=decoder_layer_features,
                skip_layers=skip_layers,
                intermediate_input_dim=gaze_transform_output_dim
                + task_output_dim
                + dx_sq_layer_dim
                + dy_sq_layer_dim
                + dx_times_dy_layer_dim
                + dxdy_dist_layer_dim,
                image_size=[ROAD_IMAGE_WIDTH, ROAD_IMAGE_HEIGHT],
                optic_flow_downsample_mode=optic_flow_downsample_mode,
                use_relu=use_relu,
                use_separable=use_separable,
                use_s3d_encoder=use_s3d_encoder,
            )  # channels for (
            # dx, dy, dropout indicator) + task_output_dim + dx^2 layer, dy^2  layer, dx*dy layer

    else:
        if add_optic_flow:
            road_facing_decoder = create_decoder(
                num_units=num_decoder_units,
                output_dim_of_decoder=output_dim_of_decoder,
                decoder_layer_features=decoder_layer_features,
                skip_layers=skip_layers,
                intermediate_input_dim=gaze_transform_output_dim
                + dx_sq_layer_dim
                + dy_sq_layer_dim
                + dx_times_dy_layer_dim
                + dxdy_dist_layer_dim
                + optic_flow_output_dim,
                # channels for (dx, dy, dropout indicator)
                image_size=[ROAD_IMAGE_WIDTH, ROAD_IMAGE_HEIGHT],
                optic_flow_downsample_mode=optic_flow_downsample_mode,
                use_relu=use_relu,
                use_separable=use_separable,
                use_s3d_encoder=use_s3d_encoder,
            )
        else:
            road_facing_decoder = create_decoder(
                num_units=num_decoder_units,
                output_dim_of_decoder=output_dim_of_decoder,
                decoder_layer_features=decoder_layer_features,
                skip_layers=skip_layers,
                intermediate_input_dim=gaze_transform_output_dim
                + dx_sq_layer_dim
                + dy_sq_layer_dim
                + dx_times_dy_layer_dim
                + dxdy_dist_layer_dim,
                # channels for (dx, dy, dropout indicator)
                image_size=[ROAD_IMAGE_WIDTH, ROAD_IMAGE_HEIGHT],
                optic_flow_downsample_mode=optic_flow_downsample_mode,
                use_relu=use_relu,
                use_separable=use_separable,
                use_s3d_encoder=use_s3d_encoder,
            )

    child_modules = torch.nn.ModuleDict()
    child_modules["driver_facing"] = torch.nn.ModuleDict()
    # child_modules['driver_facing']['linear0'] = driver_facing_linear
    child_modules["driver_facing"]["linear0"] = None
    # child_modules['driver_facing']['lstm0'] = driver_facing_lstm

    if add_task_net:
        # task_network_linear = torch.nn.Linear(num_tasks, task_output_dim)
        # task_network_linear.weight.data.normal_(mean=0, std=0.000001)
        # task_network_linear.bias.data.normal_(mean=0, std=0.000001)
        # for i in range(task_output_dim):
        #     task_network_linear.weight.data[i][i] += 1.0
        # import IPython;IPython.embed(header='task_network_linear')
        child_modules["task_network"] = torch.nn.ModuleDict()
        # child_modules['task_network']['linear0'] = task_network_linear
        child_modules["task_network"]["linear0"] = None

    if add_optic_flow:
        print(add_optic_flow)
        child_modules["optic_flow"] = torch.nn.ModuleDict()
        child_modules["optic_flow"]["linear0"] = None

    map_modules = torch.nn.ModuleDict()
    map_modules["road_facing"] = torch.nn.ModuleDict()
    map_modules["road_facing"]["encoder"] = road_facing_encoder
    map_modules["road_facing"]["decoder"] = road_facing_decoder

    output_dims = collections.OrderedDict()
    output_dims["road_facing"] = [output_dim_of_decoder, ROAD_IMAGE_WIDTH, ROAD_IMAGE_HEIGHT]
    res = FusionNet(
        child_modules,
        map_modules,
        output_dims,
        add_task_net=add_task_net,
        add_optic_flow=add_optic_flow,
        dropout_ratio=dropout_ratio,
        dropout_ratio_external_inputs=dropout_ratio_external_inputs,
    )
    return res


class GaussianPredictorNet(torch.nn.Module):
    def __init__(self, predictor_input_dim_dict):
        super().__init__()
        predictor_input_dim = np.sum(list(predictor_input_dim_dict.values()))
        self.predictor_input_dim = predictor_input_dim
        predictor_output_dim = 2 * 2
        self.common_predictor = torch.nn.Linear(predictor_input_dim, predictor_output_dim)
        self.output_mean = torch.nn.Linear(predictor_output_dim, 2)
        self.output_lstd = torch.nn.Linear(predictor_output_dim, 2)
        # import IPython;
        # IPython.embed(header='PredictorNet:init')
        self.output_mean.weight.data.normal_(0, 0.001)
        self.output_mean.bias.data.normal_(0, 0.0001)
        self.output_lstd.weight.data.normal_(0, 0.01)
        self.output_lstd.bias.data.normal_(-1, 0.001)

    def forward(self, input):
        # import IPython;IPython.embed(header='prd:forward')
        common_output = self.common_predictor(input)
        means = self.output_mean(common_output)
        lstds = self.output_lstd(common_output)
        stds = lstds.exp()
        res = collections.OrderedDict()
        res["means"] = means
        res["stds"] = stds
        res["lstds"] = lstds
        if torch.isnan(lstds).sum():
            import IPython

            IPython.embed(header="predictor:forward")

        return res


def create_simple_cognitive_predictor(input_dim):
    """
    Create a single gaussian variational predictor
    :return: an nn.Module that supports both creation of parameters, and the loss function of negative log-probability.
    """
    # import IPython;
    # IPython.embed(header='create_simple_cognitive_predictor')

    return GaussianPredictorNet(input_dim)


class MapPredictorNet(torch.nn.Module):
    def __init__(
        self,
        predictor_input_dim_dict,
        predictor_output_n_features,
        output_image_size,  # [num_latent_layers, reduced_H, reduced_W]
        add_task_output_head,
        task_output_fc_layers,
        task_output_use_bn,
        task_output_use_dropout,
        task_output_dim=5,
        task_mp_kernel_size=2,
        kernel_size=5,
    ):
        super().__init__()
        predictor_input_n_features = np.sum(
            [x[0] for x in predictor_input_dim_dict.values()]
        )  # sum up the number of channels for each of the outputs from the decoder net. In the current state, there is only Enc-Dec pipeline

        self.output_image_size = output_image_size
        self.predictor_input_dim_dict = predictor_input_dim_dict
        self.predictor_input_dim = predictor_input_n_features  # input channels #16
        self.predictor_output_dim = predictor_output_n_features  # num_latent_layers #2
        self.add_task_output_head = add_task_output_head
        self.task_output_fc_layers = task_output_fc_layers
        self.task_output_use_bn = task_output_use_bn
        self.task_output_use_dropout = task_output_use_dropout
        # import IPython;
        # IPython.embed(header='PredictorNet:init')
        # intermediate_input_dim=np.prod(predictor_output_n_features)
        stencil_size = 5  # kernel size
        # Just one conv2d? directly from 64 channels to 1 channel?
        self.common_predictor = torch.nn.Conv2d(
            predictor_input_n_features,
            predictor_output_n_features,  # = num_latent_layers argument
            stencil_size,
            padding=stencil_size // 2,
        )
        self.gaze_predictor = torch.nn.Conv1d(
            predictor_output_n_features, 1, [1], padding=0  # gaze_predictot_output_feature = 1
        )
        self.awareness_predictor = torch.nn.Conv1d(
            predictor_output_n_features, 1, [1], padding=0  # awareness_predictor_output_features = 1
        )
        self.softmax = torch.nn.Softmax(
            dim=3
        )  # dim=3 works because before applying the softmax we flatten the 2d heatmap. The softmax makes sure the probability sums to one over the entire heatmap

        # task output network
        if self.add_task_output_head:
            self.task_predictor = torch.nn.Conv1d(predictor_output_n_features, 1, [1], padding=0)
            self.task_maxpool = torch.nn.MaxPool2d(kernel_size=task_mp_kernel_size)
            task_fc_in_features = (
                1 * int(output_image_size[1] / task_mp_kernel_size) * int(output_image_size[2] / task_mp_kernel_size)
            )
            self.task_fc = SimpleNet(
                in_features=task_fc_in_features,
                out_features=task_output_dim,
                structure=tuple(self.task_output_fc_layers),
                use_bn=self.task_output_use_bn,
                use_dropout=self.task_output_use_dropout,
            )  # task_fc_in_features ---> task_output_fc_layers[0]...[n] --> task_output_dim

            self.task_softmax = torch.nn.Softmax(dim=2)

    def forward(self, input):
        # initialize a zero tensor to hold the output of the common predictor
        common_output = input.new_zeros(
            input.shape[0],  # batch
            input.shape[1],  # time
            # only change the number of channels to match output dimensions.
            self.predictor_output_dim,
            input.shape[3],  # height
            input.shape[4],
        )  # width

        for t in range(input.shape[1]):
            common_output[:, t, :, :, :] = self.common_predictor(
                input[:, t, :, :, :]
            )  # run predictor over each image in sequence

        # common output is of shape (B, T, C=num_latent_layers, H, W)
        # reshaped_common ---> gaze_estimate ---> softmax =
        # reshape_common ---> awareness_estimate
        reshaped_common = common_output.permute(0, 2, 1, 3, 4)  # (B, C=num_latent_layers, T, H, W)
        reshaped_common2 = reshaped_common.contiguous().view(
            reshaped_common.shape[0], reshaped_common.shape[1], -1
        )  # (B, C=num_latent_layers, T*H*W)

        # reshape in order to allow softmax to run (only used on the gaze estimate for now)
        # gaze_estimate = self.gaze_predictor(
        #     reshaped_common2).view_as(reshaped_common).transpose(2, 4)
        # self.gaze_predictor(reshaped_common2) #(B, 1, T*H*W)
        # self.gaze_predictor(reshaped_common2).view(reshaped_common.shape[0], 1, reshaped_common.shape[2], reshaped_common.shape[3], -1) #(B, C=1, T, H, W)
        gaze_estimate = (
            self.gaze_predictor(reshaped_common2)
            .view(reshaped_common.shape[0], 1, reshaped_common.shape[2], reshaped_common.shape[3], -1)
            .permute(0, 2, 1, 3, 4)
        )  # (B, T, C=1, H, W)
        # awareness_estimate = torch.nn.Sigmoid()(self.awareness_predictor(
        #     reshaped_common2).view_as(reshaped_common).transpose(2, 4))
        awareness_estimate = torch.nn.Sigmoid()(
            self.awareness_predictor(reshaped_common2)
            .view(reshaped_common.shape[0], 1, reshaped_common.shape[2], reshaped_common.shape[3], -1)
            .permute(0, 2, 1, 3, 4)
        )  # (B, T, C=1, H, W)

        # common_output=self.common_predictor(input)
        # TODO: incorporate discrete out-of-image locations for in-cabin
        # import IPython;IPython.embed(header='prd:forward')
        gaze_image_output = self.softmax(
            gaze_estimate.contiguous().view(gaze_estimate.shape[0], gaze_estimate.shape[1], 1, -1)
        ).view_as(
            gaze_estimate
        )  # (B, T, C=1, H, W)

        # task estimation
        if self.add_task_output_head:
            task_estimate = (
                self.task_predictor(reshaped_common2)
                .view(
                    reshaped_common.shape[0],
                    -1,
                    reshaped_common.shape[2],
                    reshaped_common.shape[3],
                    reshaped_common.shape[4],
                )
                .permute(0, 2, 1, 3, 4)
            )  # (B, T, C=1, H, W
            task_estimate = task_estimate.view(
                task_estimate.shape[0] * task_estimate.shape[1], 1, task_estimate.shape[3], -1
            )  # (BT, C=1, H, W))
            task_estimate = self.task_maxpool(task_estimate)  # (BT, C=1, H/k, W/k)
            task_estimate = task_estimate.view(task_estimate.shape[0], -1)  # (BT, C=1 *H/k *W/k)
            task_estimate = self.task_fc(task_estimate)  # (BT, 5)
            task_estimate = task_estimate.view(reshaped_common.shape[0], reshaped_common.shape[2], -1)  # (B, T, 5)
            # (B, T, 5) # proper probability vector along the feature dimension
            task_estimate = self.task_softmax(task_estimate)

        res = collections.OrderedDict()
        if self.add_task_output_head:
            res["task_output_probability"] = task_estimate
        else:
            res["task_output_probability"] = None

        res["gaze_density_map"] = gaze_image_output
        res["log_gaze_density_map"] = torch.log(gaze_image_output)
        res["awareness_map"] = awareness_estimate
        res["unnormalized_gaze"] = gaze_estimate  # sq(mean). higher coeff.
        res["common_predictor_map"] = reshaped_common  # mean(sq) - L2 reg
        if torch.isnan(gaze_image_output).sum():
            import IPython

            print(gaze_image_output)
            IPython.embed(header="predictor:forward. gaze image output (gaze_density_map) has nans")
        return res


def create_image_cognitive_predictor(
    predictor_input_dim_dict,
    img_size,
    add_task_output_head,
    task_output_fc_layers,
    task_output_use_bn,
    task_output_use_dropout,
    task_output_dim,
):  # duplicate function in this file.
    """
    Create a single image predictor
    :return: an nn.Module that supports both creation of parameters, and the loss function of negative log-probability.
    """
    # import IPython;
    # IPython.embed(header='create_image_cognitive_predictor')

    predictor_output_n_features = img_size[0]
    return MapPredictorNet(
        predictor_input_dim_dict,
        predictor_output_n_features,
        img_size,
        add_task_output_head,
        task_output_fc_layers,
        task_output_use_bn,
        task_output_use_dropout,
        task_output_dim,
    )
    # first argument - predictor_input_dim_dict is the output of all the decoders. Currently only one Encoder-Decoder sturcture is used
    # second argument is the number of channels in the output map of the predictor = num_latent_layers


class CognitiveHeatNet(torch.nn.Module):
    """"""

    def __init__(
        self,
        create_gaze_transform_fn=create_identity_gaze_transform,
        create_cognitive_map_fn=create_image_cognitive_map,
        create_predictor_fn=create_simple_cognitive_predictor,
        orig_input_img_size=[3, 1080, 1920],  # channels, height, width
        aspect_ratio_reduction_factor=6,
        num_latent_layers=2,
        add_task_output_head=False,
        task_output_dim=5,
        task_output_fc_layers=[10, 5],
        task_output_use_bn=False,
        task_output_use_dropout=False,
        # gaze_spatial_regularization_coeff=1.0,
        # awareness_spatial_regularization_coeff=1.0,
        # gaze_temporal_regularization_coeff=1.0,
        # awareness_temporal_regularization_coeff=1.0,
        add_task_net=False,
        add_optic_flow=False,
    ):
        # gaze_coeff=1.0):
        super().__init__()
        # main network structure
        # gaze_transform - a network that takes a gaze and outputs a tranformed gaze (identity for now)
        self.add_task_net = add_task_net
        self.add_task_output_head = add_task_output_head
        self.task_output_dim = task_output_dim
        self.add_optic_flow = add_optic_flow
        self.task_output_fc_layers = task_output_fc_layers
        self.task_output_use_bn = task_output_use_bn
        self.task_output_use_dropout = task_output_use_dropout
        self.output_img_size = [
            num_latent_layers,
            int(round(orig_input_img_size[1] / aspect_ratio_reduction_factor)),
            int(round(orig_input_img_size[2] / aspect_ratio_reduction_factor)),
        ]
        self.ROAD_IMAGE_HEIGHT = self.output_img_size[1]
        self.ROAD_IMAGE_WIDTH = self.output_img_size[2]
        self.gaze_transform = create_gaze_transform_fn()  #
        self.cognitive_map = create_cognitive_map_fn(
            road_image_height=self.ROAD_IMAGE_HEIGHT,
            task_output_dim=self.task_output_dim,
            road_image_width=self.ROAD_IMAGE_WIDTH,
        )  # create image cognitive map ()
        # this is the dict containing the output dimensions of all outputs form the decoder.
        predictor_input_dim_dict = self.cognitive_map.output_dims
        # self.gaze_spatial_regularization_coeff = gaze_spatial_regularization_coeff
        # self.awareness_spatial_regularization_coeff = awareness_spatial_regularization_coeff
        # self.gaze_temporal_regularization_coeff = gaze_temporal_regularization_coeff
        # self.awareness_temporal_regularization_coeff = gaze_temporal_regularization_coeff
        # self.gaze_data_coeff = gaze_coeff
        # create image cognitive predictor (heatmap), predictor_input_dim is the output size (64, 224, 224) of the decoderoutput_img_size is the size of the heatmap. 1 channel heatmap
        self.cognitive_predictor = create_predictor_fn(
            predictor_input_dim_dict,
            self.output_img_size,
            self.add_task_output_head,
            self.task_output_fc_layers,
            self.task_output_use_bn,
            self.task_output_use_dropout,
            self.task_output_dim,
        )
        # self.spatial_regularization = EPSpatialRegularization()
        # self.temporal_regularization = EPTemporalRegularization()

    def get_modules(self):
        # TODO(guy.rosman) is this the right test? you could have distributed w/ 1 GPU
        if torch.cuda.device_count() > 1:
            road_facing_network = self.cognitive_map.domain_dict["road_facing"]
            cognitive_map_network = self.cognitive_map
            gaze_transform_prior_loss = self.gaze_transform.prior_loss
        else:
            road_facing_network = self.cognitive_map.domain_dict["road_facing"]
            cognitive_map_network = self.cognitive_map
            gaze_transform_prior_loss = self.gaze_transform.prior_loss

        return cognitive_map_network, road_facing_network, gaze_transform_prior_loss

    def forward(self, data, should_drop_indices_dict=None, should_drop_entire_channel_dict=None):
        """
        :param data: data instance, dictionary of inputs, of size BxTx(other dimensions) each: for now, BxTxLx2 for gaze, BxTxCxHxW, where:
        B - batch size, T - time span for an example, C - num of channels, H,W - height&width
        :return: distribution parameters of gaze
        """
        # tracker_img = data['tracker_image']  #image from tracker camera
        # image from garmin camera normalized. #TODO(deepak.gopinath) possibly move normalization into the dataset  (B, T, C, H, W)
        road_img = data["road_image"] / 255.0

        # input gaze in road frame. possibly corrupted as well. (B, T, L, 2)
        # input_gaze = data['input_gaze']``
        if "no_detach_gaze" in data:
            if data["no_detach_gaze"]:
                normalized_input_gaze = data["normalized_input_gaze"]  # (B, T, L, 2)
            else:
                normalized_input_gaze = data["normalized_input_gaze"].clone().detach()  # (B, T, L, 2)
        else:
            normalized_input_gaze = data["normalized_input_gaze"].clone().detach()  # (B, T, L, 2)
        should_train_input_gaze = data["should_train_input_gaze"].clone().detach()  # (B, T, L, 1)
        # [tracker_img, road_img, input_gaze, transform, transform_confidence, flags]=data
        # get gaze coordinate in etg
        # transform to road-facing
        try:
            # import IPython
            # IPython.embed(banner1='check nan gaze')
            # transformed_gaze = self.gaze_transform(input_gaze)  #this transform is the 'correction' that gets learned to a corrupted gaze input?
            normalized_transformed_gaze = self.gaze_transform(
                normalized_input_gaze, should_train_input_gaze
            )  # (B, T, L, 4)
        except:
            import IPython

            IPython.embed(header="input_gaze")

        chm_input = collections.OrderedDict()
        # chm_input['driver_facing']=transformed_gaze #
        chm_input["driver_facing"] = normalized_transformed_gaze
        chm_input["road_facing"] = road_img
        if self.add_task_net:
            # (B, T, NUM_TASKS, 1)  #need to clone and detach so that when dropout happens it doesn't modify the dict item
            chm_input["task_network"] = data["task_type"].clone().detach()

        if self.add_optic_flow:
            chm_input["optic_flow"] = data["optic_flow_image"].clone().detach()

        if torch.isnan(chm_input["driver_facing"]).sum():
            import IPython

            print(chm_input["driver_facing"])
            IPython.embed(banner1="CHM::Forward, chm_input[driver_facing] has nans")

        (decoder_output, child_networks_output, should_drop_dicts) = self.cognitive_map(
            chm_input, should_drop_indices_dict, should_drop_entire_channel_dict
        )
        # if there are multiple Enc-Decoder pipelines, then all the outputs needs to be collated along the channel dimension
        # to create the predictor input. predictor_input = (B, T, C1+C2+...., H, W)
        # B, T, C, H, W, where C is output_dim_of_decoder
        predictor_input = decoder_output["road_facing"]
        # predictor_input=predictor_input.view(predictor_input.shape[0],predictor_input.shape[1],-1)
        if (torch.isnan(predictor_input)).sum():  # check if there is any nans in the encoder-decoder pass
            import IPython

            print(predictor_input)
            IPython.embed(header="CHM:forward. Output of the decoder fused[0][road_facing] has nans")
        # compute image-facing part
        # feed into temporal estimate
        output = self.cognitive_predictor(predictor_input)
        return output, decoder_output, child_networks_output, should_drop_dicts
