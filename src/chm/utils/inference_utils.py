import numpy as np


def seek_mode(img, start_x, start_y, sigma):
    """
    Do mean-shift seeking
    :param img: an intensity image to converge on
    :param start_x: initial guess on location
    :param start_y: initial guess on location
    :param sigma: the standard deviation / scale of the gaussian filter to use
    :return:
    """
    x = start_x  # in pixels
    y = start_y  # in pixels
    pos_x = range(img.shape[1])  # width of the image
    pos_y = range(img.shape[0])  # height of the image.
    mesh_x, mesh_y = np.meshgrid(pos_x, pos_y)  # mesh rid for the pixel coordinate
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
    return (x, y), stats