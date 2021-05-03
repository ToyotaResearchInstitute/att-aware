# Copyright 2020 Toyota Research Institute.  All rights reserved.
import numpy as np


def seek_mode(img, start_x, start_y, sigma):
    """
    Mean shift algorithm (searches for nearest mode) on img starting from (start_x, start_y)

    Parameters:
    -----------
    img: np.array
        Image on which meanshift algorithm needs to be performed
    start_x: int
        x coordinate of starting point
    start_y: int
        y coordinate of starting point
    sigma: int
        Sigma for the gaussian centered around current x, y during meanshift

    Returns:
    --------
    final_xy: tuple
        Tuple containing the final point of the meanshift search. (x_coord, y_coord)
    stats: dict
        Dict containing the sequence of x and y points. Typically useful for visualization.

    """
    x = start_x  # in pixels
    y = start_y  # in pixels
    pos_x = range(img.shape[1])  # width of the image
    pos_y = range(img.shape[0])  # height of the image.
    mesh_x, mesh_y = np.meshgrid(pos_x, pos_y)  # meshgrid for the pixel coordinate
    sequence_x = []
    sequence_y = []
    for i in range(20):
        sequence_x.append(x)
        sequence_y.append(y)
        gaussian = np.exp((-((mesh_x - x) ** 2) - (mesh_y - y) ** 2) / sigma ** 2)  # gaussian centered around x, y
        modulated_image = img * gaussian
        EPS_OLD_LOC = 1e-5
        new_x = (np.sum(modulated_image * mesh_x) + EPS_OLD_LOC * x) / (np.sum(modulated_image) + EPS_OLD_LOC)
        new_y = (np.sum(modulated_image * mesh_y) + EPS_OLD_LOC * y) / (np.sum(modulated_image) + EPS_OLD_LOC)
        squared_norm = (new_x - x) ** 2 + (new_y - y) ** 2
        x = new_x
        y = new_y
        if squared_norm < (0.5) ** 2:  # this threshold should be a param, maybe
            break

    sequence_x.append(x)
    sequence_y.append(y)
    stats = {}
    stats["sequence_x"] = sequence_x
    stats["sequence_y"] = sequence_y
    final_xy = (x, y)
    return final_xy, stats