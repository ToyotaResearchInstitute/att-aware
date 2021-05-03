# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

plt.switch_backend("agg")


def visualize_overlaid_images(
    predicted_output,
    batch_input,
    batch_target,
    global_step,
    num_visualization_examples=5,
    logger=None,
    is_gaze=True,
    normalize_scale=255,
    color_range=None,
    alpha=0.6,
    dl_key="test",
    force_value_str=None,
):
    """
    Log heatmaps on tensorboard. Supports both cumulative as well as single frame heatmap logging
    """
    # needs a logger to log visualized heatmaps
    assert logger is not None
    for instance_idx in range(num_visualization_examples):
        # if the instance idx is greater than the batch size break this loop
        if instance_idx >= batch_target.shape[0]:
            break

        for cumulative in [True, False]:
            postfix_str = "" if cumulative is True else "_single"

            fg = plt.figure()
            ax = fg.subplots()

            # BGR and now between 0 and 1.0
            road_img = batch_input["road_image"][instance_idx, -1, :, :, :].cpu().detach().numpy() / normalize_scale
            road_img_t = road_img.transpose([1, 2, 0])  # BGR #(H, W, 3)

            num_images_fused = 1

            if is_gaze:
                # accumulate gaze heatmap
                img = predicted_output["gaze_density_map"][instance_idx, 0, 0, :, :].cpu().detach().numpy()
                seq_len = predicted_output["gaze_density_map"].shape[1]
                if cumulative:
                    for i in range(1, seq_len):  # accumulate the heatmaps for each timestep in the timeslice.
                        img += predicted_output["gaze_density_map"][instance_idx, i, 0, :, :].cpu().detach().numpy()
                    num_images_fused = seq_len
            else:
                # accumulate awareness heatmap
                img = predicted_output["awareness_map"][instance_idx, 0, 0, :, :].cpu().detach().numpy()
                seq_len = predicted_output["awareness_map"].shape[1]
                if cumulative:
                    for i in range(1, seq_len):  # accumulate the heatmaps for each timestep in the timeslice.
                        img += predicted_output["awareness_map"][instance_idx, i, 0, :, :].cpu().detach().numpy()
                    num_images_fused = seq_len

            img /= num_images_fused
            if color_range is None:
                img_qs = np.percentile(img, [1, 99])
                # remove outliers
                img[img <= img_qs[0]] = img_qs[0]
                img[img > img_qs[1]] = img_qs[1]
            else:
                img_qs = color_range

            # after this 0.0 is mapped to img_qs[0] and 1.0 is ampped to img_qs[1]
            img = (img - img_qs[0]) / (img_qs[1] - img_qs[0])
            img_cv_bgr = cv2.applyColorMap(np.uint8(img * 255), cv2.COLORMAP_JET)  # (H, W, 3) #BGR
            img_cv_bgr = np.float32(img_cv_bgr) / 255  # Normalize to 0 to 1

            # Create blended image
            img_cv_rgb = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) #RGB
            road_img_t = cv2.cvtColor(road_img_t, cv2.COLOR_BGR2RGB)  # (H, W, 3, RGB
            # to ensure that the the heatmap and the road image can be properly blended
            assert road_img_t.shape == img_cv_rgb.shape
            overlaid_img = alpha * road_img_t + (1 - alpha) * img_cv_rgb  # (H, W, 3, RGB)

            plt.imshow(overlaid_img, cmap="jet")  # RGB overlaid image
            cb = plt.colorbar()
            num_ticks = np.size(cb.get_ticks())  # number of tick marks on the color bar
            # make the tick mark labels to be the true probabilities and not the normalized probabilities
            cb.ax.set_yticklabels(np.around(np.linspace(img_qs[0], img_qs[1], num=num_ticks), decimals=7))

            # plot gaze points
            if cumulative:
                target = batch_target[instance_idx, :, :, :].cpu().detach()  # (T, L, 2)
            else:
                target = batch_target[instance_idx, 0:1, :, :].cpu().detach()  # (1, L, 2)

            # (TL, 2) accumulate all gaze points from all frames in this time slice
            target = target.reshape(target.shape[0] * target.shape[1], -1)

            # batch_input['normalized_input_gaze'] is of shape (B,T,L,2)
            if cumulative:
                # (T, L, 2)
                noisy_gaze = batch_input["normalized_input_gaze"][instance_idx, :, :, :].cpu().detach().numpy()
            else:
                # (1, L, 2)
                noisy_gaze = batch_input["normalized_input_gaze"][instance_idx, 0:1, :, :].cpu().detach().numpy()

            gaze_list_length = noisy_gaze.shape[1]
            seq_len = noisy_gaze.shape[0]

            # (TL, 2) All gaze points are within [0, 1]
            noisy_gaze = noisy_gaze.reshape(noisy_gaze.shape[0] * noisy_gaze.shape[1], -1)
            # scale noisy gaze to network image dimensions
            road_img_height, road_img_width = road_img_t.shape[:2]
            road_img_dim = np.array((road_img_width, road_img_height))
            # (TL, 2) of (width, height) Tile it so that each of the gaze points in noisy gaze can be scaled to the correct network dimensions
            road_img_dim = np.tile(road_img_dim, reps=(noisy_gaze.shape[0], 1))
            # (TL, 2) scale the normalized noisy gaze to the network image dimensions
            noisy_gaze = np.multiply(noisy_gaze, road_img_dim)

            target_np = target.numpy()
            # both should have the same number of data points and should be in correspondence.
            assert target_np.shape[0] == noisy_gaze.shape[0]
            t = 0
            for i in range(target_np.shape[0]):
                if i % gaze_list_length == 0:
                    t += 1
                    mrkrsize = (t / seq_len) * 13 + 0.5
                    if (
                        target_np[i, 0] < 0.0
                        or target_np[i, 0] > road_img_width
                        or target_np[i, 1] < 0.0
                        or target_np[i, 1] > road_img_height
                    ):
                        pass
                    else:
                        plt.plot(
                            target_np[i, 0],
                            target_np[i, 1],
                            marker="+",
                            color="lime",
                            markersize=mrkrsize,
                            linewidth=mrkrsize / 4,
                        )
                # skip invalid points
                if (
                    noisy_gaze[i, 0] < 0.0
                    or noisy_gaze[i, 0] > road_img_width
                    or noisy_gaze[i, 1] < 0.0
                    or noisy_gaze[i, 1] > road_img_height
                ):
                    continue
                else:
                    # white is noisy, green is true
                    plt.plot(
                        noisy_gaze[i, 0],
                        noisy_gaze[i, 1],
                        marker="+",
                        color="w",
                        markersize=mrkrsize,
                        linewidth=mrkrsize / 4,
                    )
                    # draw connecting line between true and noisy gaze.
                    if not cumulative:
                        # plotting lines when cumulative is just too messy.
                        plt.plot([target_np[i, 0], noisy_gaze[i, 0]], [target_np[i, 1], noisy_gaze[i, 1]], "y")

            # log the visualized heatmaps on tensorboard
            if is_gaze:
                logger.add_figure(
                    tag=dl_key + "/gaze_heatmap" + postfix_str + str(instance_idx) + "_" + force_value_str,
                    figure=fg,
                    global_step=global_step,
                )
            else:
                logger.add_figure(
                    tag=dl_key + "/awareness_heatmap" + postfix_str + str(instance_idx) + "_" + force_value_str,
                    figure=fg,
                    global_step=global_step,
                )


def visualize_awareness_labels(
    predicted_awareness_output,
    awareness_batch_input,
    awareness_batch_target,
    awareness_batch_annotation_data,
    global_step,
    normalize_scale=255,
    alpha=0.6,
    num_visualization_examples=5,
    logger=None,
    annotation_image_size=[3, 1080, 1920],
    dl_key="awareness_test",
    force_value_str=None,
):
    """
    Logs awareness heatmaps with labels visualizes on tensorboard.
    """
    assert logger is not None
    for instance_idx in range(num_visualization_examples):
        # if the instance idx is greater than the batch size break this loop
        if instance_idx >= awareness_batch_target.shape[0]:
            break

        fg = plt.figure()
        ax = fg.subplots()
        road_img = (
            awareness_batch_input["road_image"][instance_idx, -1, :, :, :].cpu().detach().numpy() / normalize_scale
        )
        road_img_t = road_img.transpose([1, 2, 0])  # {H, W, C}
        img = predicted_awareness_output["awareness_map"][instance_idx, -1, 0, :, :].cpu().detach().numpy()

        img_cv_bgr = cv2.applyColorMap(np.uint8(img * 255), cv2.COLORMAP_JET)
        img_cv_bgr = np.float32(img_cv_bgr) / 255  # Normalize to 0 to 1
        img_cv_rgb = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2RGB)
        overlaid_img = alpha * road_img_t + (1 - alpha) * img_cv_rgb

        plt.imshow(overlaid_img, cmap="jet")
        plt.colorbar()

        x = (awareness_batch_annotation_data["query_x"][instance_idx] / annotation_image_size[2] * img.shape[-1]).int()
        y = (awareness_batch_annotation_data["query_y"][instance_idx] / annotation_image_size[1] * img.shape[-2]).int()

        true_label = float(awareness_batch_annotation_data["annotation_target"][instance_idx])
        predicted_label = float(img[y, x])
        title = "PredictedL = " + "{0:.3f}".format(predicted_label) + ", TrueL = " + str(true_label)
        ax.set_title(title)  # presented the predicted and true label as heading.
        ax.scatter(x, y, s=140, color="r", marker="x", linewidth=6)  # plot the annotation cross hair

        noisy_gaze = (
            awareness_batch_input["normalized_input_gaze"][instance_idx, :, :, :].cpu().detach().numpy()
        )  # (T, L, 2)
        # (TL, 2) All gaze points are within [0, 1]
        noisy_gaze = noisy_gaze.reshape(noisy_gaze.shape[0] * noisy_gaze.shape[1], -1)
        # scale noisy gaze to network image dimensions
        road_img_height, road_img_width = road_img_t.shape[:2]
        road_img_dim = np.array((road_img_width, road_img_height))
        # (TL, 2) of (width, height) Tile it so that each of the gaze points in noisy gaze can be scaled to the correct network dimensions
        road_img_dim = np.tile(road_img_dim, reps=(noisy_gaze.shape[0], 1))
        # scale noisy gaze to proper network dimensions
        noisy_gaze = np.multiply(noisy_gaze, road_img_dim)
        # awareness_batch_target is of dim, B x T x L x 2
        # (T, L=3, 2) #get gaze points from the frames. To get a sense of what type of gaze patterns preceded the cross hair in annotation
        target = awareness_batch_target[instance_idx, :, :, :].cpu().detach()
        gaze_list_length = target.shape[1]
        seq_len = target.shape[0]
        # (TL, 2) accumulate all gaze points from the last two frames in this time slice (L=2)
        target = target.reshape(target.shape[0] * target.shape[1], -1)
        target_np = target.numpy()
        assert target_np.shape[0] == noisy_gaze.shape[0]

        t = 0
        for i in range(target_np.shape[0]):
            if i % gaze_list_length == 0:
                t += 1
                mrkrsize = (t / seq_len) * 13 + 0.5
            if (
                noisy_gaze[i, 0] < 0.0
                or noisy_gaze[i, 0] > road_img_width
                or noisy_gaze[i, 1] < 0.0
                or noisy_gaze[i, 1] > road_img_height
            ):
                continue
            else:
                plt.plot(
                    target_np[i, 0],
                    target_np[i, 1],
                    marker="+",
                    color="lime",
                    markersize=mrkrsize,
                    linewidth=mrkrsize / 4,
                )
                # white is noisy, lime is true
                plt.plot(
                    noisy_gaze[i, 0],
                    noisy_gaze[i, 1],
                    marker="+",
                    color="w",
                    markersize=mrkrsize,
                    linewidth=mrkrsize / 4,
                )
        logger.add_figure(
            tag=dl_key + "/overlaid_awareness_label_map" + str(instance_idx) + "_force_value_" + force_value_str,
            figure=fg,
            global_step=global_step,
        )
