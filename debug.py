"""
data aug methods
"""
import math
import cv2
import random
import numpy as np
from skimage import transform as sktsf
import torch


def random_flip(img, bbox, y_random=False, x_random=False,
                return_param=False):
    """Randomly flip an image and bbox in vertical or horizontal direction.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            HWC format.
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            R is the number of bbox, (y_{min}, x_{min}, y_{max}, x_{max})
        y_random (bool): Randomly flip in vertical direction.
        x_random (bool): Randomly flip in horizontal direction.
        return_param (bool): Returns information of flip.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):
        img, bbox, *param{y_flip (bool), x_flip(bool)}
    """

    H, W = img.shape[:2]
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[::-1, :, :]
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        img = img[:, ::-1, :]
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max

    if return_param:
        return img, bbox, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img, bbox


if __name__ == "__main__":
    pass