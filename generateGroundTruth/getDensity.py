import os
import cv2
import h5py
import scipy
import scipy.spatial
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from getGTBox import getGTBox
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from scipy.ndimage.filters import gaussian_filter 
try:
    from mypath import Path
except:
    import sys
    sys.append('G:/CV/Reading/ClusterRegionGenerationNetwork')
    from mypath import path


hyp = {'lbt': 100,  # Multiple of the density map numerical magnification
       'output_scale': (30, 40)}


def show_image(img, labels=None):
    if type(labels) is not np.ndarray:
        labels = np.array(labels)
    # plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    if labels is not None:
        plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-')
    # plt.savefig('test_0.jpg')
    plt.show()


def show_density(density):
    plt.imshow(density, cmap=CM.jet)
    plt.show()


def double_linear(input_density, out_scale=(30, 40)):
    in_h, in_w = input_density.shape
    out_h, out_w = out_scale
    ratio_x, retio_y = float(in_h) / out_h, float(in_w) / out_w

    # result of resize after double linear
    output_density = np.zeros(out_scale)

    for i in range(out_scale[0]):
        for j in range(out_scale[1]):
            src_x = (i + 0.5) * ratio_x - 0.5
            src_y = (j + 0.5) * retio_y - 0.5

            # find the coordinates of the points which will be used to compute the interpolation
            src_x0 = int(np.floor(src_x))
            src_x1 = min(src_x0 + 1, in_h - 1)
            src_y0 = int(np.floor(src_y))
            src_y1 = min(src_y0 + 1, in_w - 1)

            # calculate the interpolation
            temp0 = (src_x1 - src_x) * input_density[src_x0, src_y0] +\
                    (src_x - src_x0) * input_density[src_x1, src_y0]
            temp1 = (src_x1 - src_x) * input_density[src_x0, src_y1] +\
                    (src_x - src_x0) * input_density[src_x1, src_y1]
            output_density[i, j] = (src_y1 - src_y) * temp0 +\
                                   (src_y - src_y0) * temp1

    return output_density


def gaussian_filter_density(img, bboxes):
    if type(bboxes) is not np.ndarray:
        bboxes = np.array(bboxes)

    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img.shape[1])
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img.shape[0])

    gt = np.zeros(img.shape[:2])

    for bbox in bboxes:
        c_x = int((bbox[0] + bbox[2]) / 2)
        c_y = int((bbox[1] + bbox[3]) / 2)
        gt[c_y][c_x] = 1

    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[0], np.nonzero(gt)[1])))
    leafsize = 2048

    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[0], pt[1]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


def getDensity(root_path=Path.db_root_dir('visdrone'), dataset='visdrone'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--type', default='txt', type=str,
                        choices=['xml', 'json', 'txt'], help='annotation type')
    opt = parser.parse_args()

    imgs_path = osp.join(root_path, 'JPEGImages')
    img_list = os.listdir(imgs_path)
    mask_path = osp.join(root_path, 'Mask')
    if not osp.exists(mask_path):
        os.mkdir(mask_path)

    for img_id in tqdm(img_list):
        img_path = osp.join(imgs_path, img_id)
        img = cv2.imread(img_path)

        if opt.type == 'json':
            anno_path = ""
            bboxes = getGTBox(anno_path, img_id)
        else:
            anno_path = img_path.replace('JPEGImages', 'Annotations').replace('jpg', opt.type)
            bboxes = getGTBox(anno_path)

        density = gaussian_filter_density(img, bboxes)
        density = density * 100
        final_density = double_linear(density, (30, 40))

        show_density(density)
        show_image(img, bboxes)
        show_density(final_density)


if __name__ == '__main__':
    getDensity()
