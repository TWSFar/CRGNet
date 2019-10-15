from scipy.ndimage.filters import gaussian_filter
import numpy as np
import cv2
from skimage import transform as sktsf
import math
from PIL import Image


def NN_interpolation(input, out_scale):
    scrH, scrW = input.shape
    dstH, dstW = out_scale

    output = np.zeros((dstH, dstW))
    for i in range(dstH):
        for j in range(dstW):
            scrx = round((i+1) * (scrH/dstH))
            scry = round((j+1) * (scrW/dstW))
            output[i, j] = input[scrx-1, scry-1]
    return output


def BiLinear_interpolation(input, out_scale):
    scrH, scrW = input.shape
    dstH, dstW = out_scale
    ratio_x = (scrH / dstH)
    ratio_y = (scrW / dstW)
    input = np.pad(input, ((0, 1), (0, 1)), 'constant')
    output = np.zeros(out_scale)
    for i in range(dstH):
        for j in range(dstW):
            scrx = (i+1) * ratio_x - 1
            scry = (j+1) * ratio_y - 1
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx-x
            v = scry-y
            output[i, j] = (1-u) * (1-v) * input[x, y] +\
                           u * (1-v) * input[x+1, y] +\
                           (1-u) * v * input[x, y+1] +\
                           u * v * input[x+1, y+1]
    return output

img = np.random.rand(10, 10) * 7
sum1 = img.sum()

img2 = sktsf.resize(img, (20, 20))
sum2 = img2.sum()



img4 = BiLinear_interpolation(img, (20, 20))
sum4 = img4.sum()

pass