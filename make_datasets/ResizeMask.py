import math
import numpy as np


def pooling(input, out_scale=(30, 40), method='sum'):
    in_h, in_w = input.shape
    out_h, out_w = out_scale
    ratio_x, ratio_y = in_h / out_h, in_w / out_w
    output_density = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            x1 = int(i * ratio_x)
            x2 = int((i + 1) * ratio_x)
            y1 = int(j * ratio_y)
            y2 = int((j + 1) * ratio_y)
            if method == 'sum':
                output_density[i, j] = input[x1:x2, y1:y2].sum()
            elif method == 'max':
                output_density[i, j] = input[x1:x2, y1:y2].max()
            elif method == 'mean':
                output_density[i, j] = input[x1:x2, y1:y2].mean()

    return output_density


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


def BiBubic(x):
    x = abs(x)
    if x <= 1:
        return 1 - 2 * (x**2) + (x**3)
    elif x < 2:
        return 4 - 8 * x + 5 * (x**2) - (x**3)
    else:
        return 0


def BiCubic_interpolation(input, out_scale=(30, 40)):
    scrH, scrW = input.shape
    dstH, dstW = out_scale
    output = np.zeros(out_scale)
    ratio_x = (scrH / dstH)
    ratio_y = (scrW / dstW)
    for i in range(dstH):
        for j in range(dstW):
            scrx = i * ratio_x
            scry = j * ratio_y
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx - x
            v = scry - y
            tmp = 0
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if x+ii < 0 or y+jj < 0 or x+ii >= scrH or y+jj >= scrW:
                        continue
                    tmp += input[x+ii, y+jj] * BiBubic(ii-u) * BiBubic(jj-v)
            output[i, j] = tmp

    return output
