import torch
import numpy as np


class AwarenessPointwiseLabelLoss:
    def __init__(self, loss_type="huber_loss", patch_half_size=4, annotation_image_size=[3, 1080, 1920]):
        """
        Parameters:
        ----------
        loss_type: str
            string indicating the type of loss to be computed. ['huber_loss', 'squared_loss']
        patch_half_size: int
            half size of the patch around the label to be considered for computing loss
        """
        assert loss_type == "huber_loss" or loss_type == "squared_loss"
        self.loss_type = loss_type
        self.patch_half_size = patch_half_size
        self.annotation_image_size = annotation_image_size

    def loss(self, awareness_output, awareness_batch_annotation_data):
        """
        Parameters:
        ----------
        awareness_output: dict
            dict containing the predicted gaze and awareness maps from the awareness_dataset
        awareness_batch_annotation_data: dict
            dict containing the annotation data. (query_x, query_y, annotation_target)

        Returns:
        -------
        annotation_loss: torch.Tensor
            huber or mse loss computed on annotation labels
        stats: dict
            dict containing auxiliary information regarding the loss computation
        """

        stats = {}
        stats["awareness_mse"] = 0
        stats["awareness_img_max"] = 0
        stats["awareness_l1_loss"] = 0
        stats["awareness_label_mean"] = 0
        stats["awareness_predicted_label_mean"] = 0
        stats["awareness_per_pixel_l1_loss"] = 0

        if awareness_batch_annotation_data is None and awareness_output is None:
            return 0.0, stats
        else:
            awareness_map = awareness_output["awareness_map"]
            annotation_loss = awareness_map.new_zeros([1, 1])
            squared_error = awareness_map.new_zeros([1, 1])
            batch_img_max = awareness_map.new_zeros([1, 1])
            batch_l1_error = awareness_map.new_zeros([1, 1])
            per_pixel_l1_error = awareness_map.new_zeros([1, 1])
            err_threshold = 8
            err_threshold_sq = err_threshold ** 2
            batch_predicted_label_list = []

            for b in range(awareness_map.shape[0]):  # batch dimension
                # grab the last frame in the time slice. Because the imerit annotation IS for the last time frame.
                t = awareness_map.shape[1] - 1
                # grab the 2d awareness heatmap for the last time frame for the bth time slice in the batch
                img = awareness_map[b, t, 0, :, :]  # 0 because it is a single channel, img is 2D
                # scale annotation to the proper network output size. query_x and query_y are in full video resolution
                x = (
                    awareness_batch_annotation_data["query_x"][b]
                    / self.annotation_image_size[1]
                    * awareness_map.shape[-1]
                ).int()
                y = (
                    awareness_batch_annotation_data["query_y"][b]
                    / self.annotation_image_size[0]
                    * awareness_map.shape[-2]
                ).int()

                # get an patch around the label and compute loss for all pixels in the patch.
                # under the assumption that the annotated awareness is the same in every pixel in the patch
                minx = max(x.new_tensor(0), x - self.patch_half_size)
                maxx = min(x.new_tensor(img.shape[1] - 1), x + self.patch_half_size)
                miny = max(y.new_tensor(0), y - self.patch_half_size)
                maxy = min(y.new_tensor(img.shape[0] - 1), y + self.patch_half_size)
                img_patch = img[miny:maxy, minx:maxx]  # tensor
                num_pixels_in_patch = img_patch.numel()
                target_patch = (
                    awareness_batch_annotation_data["annotation_target"][b]
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(img_patch.shape[0], img_patch.shape[1])
                )
                d_awareness = torch.sum((img_patch - target_patch).abs())

                batch_predicted_label_list.append(img[y, x].detach().cpu().numpy())
                d_awareness_sq = d_awareness ** 2

                # compute Huber loss
                if self.loss_type == "huber_loss":
                    if d_awareness_sq < err_threshold_sq:
                        loss = d_awareness_sq * 0.5
                    else:
                        loss = err_threshold * d_awareness - 0.5 * err_threshold_sq
                elif self.loss_type == "squared_loss":
                    loss = d_awareness_sq

                img_max = torch.max(img)

                annotation_loss += loss
                squared_error += d_awareness ** 2
                batch_l1_error += d_awareness
                per_pixel_l1_error += d_awareness / num_pixels_in_patch
                batch_img_max += img_max  # accumulate the max pixel value for each predicted frame.

            # since already divided by the batch_size the error is the average error PER sequence.
            stats["awareness_mse"] = squared_error / awareness_map.shape[0]  # denominator is the batch_size
            # this l1_loss is for the ENTIRE patch considered. For per pixel average error, divide this by the total size of the patch
            stats["awareness_l1_loss"] = batch_l1_error / awareness_map.shape[0]
            stats["awareness_per_pixel_l1_loss"] = per_pixel_l1_error / awareness_map.shape[0]
            stats["awareness_img_max"] = batch_img_max / awareness_map.shape[0]
            stats["awareness_label_mean"] = awareness_batch_annotation_data["annotation_target"][:].float().mean()
            stats["awareness_predicted_label_mean"] = loss.new_tensor((np.mean(batch_predicted_label_list)))
            return annotation_loss, stats
