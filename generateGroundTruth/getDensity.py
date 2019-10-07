import os
import cv2
import h5py
import scipy
import scipy.spatial
import argparse
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from scipy.ndimage.filters import gaussian_filter 
from ResizeMask import BiLinear_interpolation, BiCubic_interpolation
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from mypath import Path
from dataloaders.datasets import Datasets


hyp = {'visdrone': 0.1,  # Multiple of the density map numerical magnification
       'hkb': 1,
       'interpolation_scale': (30, 40),
       'stand_scale': (90, 120)}


def show_image(img, labels=None):
    if type(labels) is not np.ndarray:
        labels = np.array(labels)
    # plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    if labels is not None:
        plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-')
    # plt.savefig('test_0.jpg')
    plt.show()


def show_density(density_mask):
    plt.imshow(density_mask, cmap=CM.jet)
    plt.show()


def gaussian_filter_density(img_scale, bboxes, idx):
    k = 3  # k neighbor
    if type(bboxes) is not np.ndarray:
        bboxes = np.array(bboxes)

    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img_scale[1])
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img_scale[0])

    gt = np.zeros(hyp['stand_scale'])
    ratio_x = float(hyp['stand_scale'][0]) / img_scale[0]
    ratio_y = float(hyp['stand_scale'][1]) / img_scale[1]

    for bbox in bboxes:
        c_y = int((bbox[0] + bbox[2]) / 2.0 * ratio_y)
        c_x = int((bbox[1] + bbox[3]) / 2.0 * ratio_x)
        gt[c_x][c_y] += 1

    density_mask = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density_mask

    pts = np.array(list(zip(np.nonzero(gt)[0], np.nonzero(gt)[1])))
    leafsize = 2048

    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=k+1)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[0], pt[1]] = gt[pt[0], pt[1]]
        nb = 0  # neighbor number
        if gt_count > 1:
            dist = 0
            for j in range(1, k+1):
                if distances[i][j] != float('inf'):
                    dist += distances[i][j]
                    nb += 1
            sigma = dist * nb * 0.04
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
        density_mask += gaussian_filter(pt2d, sigma, mode='constant')
        sys.stdout.write('\rcomplete: {:d}/{:d}, {:d}'
                         .format(i + 1, gt_count, idx + 1))
        sys.stdout.flush()
    return density_mask


def getDensity(dataset_name='visdrone'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    opt = parser.parse_args()
    opt.data_dir = Path.db_root_dir(dataset_name)
    opt.input_size = None
    opt.dataset = dataset_name

    dataset = Datasets(opt)

    mask_path = osp.join(dataset.data_dir, 'DensityMask')
    if not osp.exists(mask_path):
        os.mkdir(mask_path)

    for i, sample in enumerate(dataset.samples):
        img_scale = (sample['height'], sample['width'])
        density_mask = gaussian_filter_density(img_scale, sample['bboxes'], i)
        density_mask = density_mask * hyp[dataset_name]
        dst_density_mask = BiCubic_interpolation(density_mask, 
                                                 hyp['interpolation_scale'])

        dst_density_mask = dst_density_mask * 9  # 9 is downsample
        maskname = osp.join(mask_path, osp.basename(sample['image']).
                            replace(dataset.img_type, '.h5'))
        with h5py.File(maskname, 'w') as hf:
            hf['density'] = dst_density_mask

        if opt.show:
            img = cv2.imread(sample['image'])
            show_image(img, sample['bboxes'])
            show_density(density_mask)
            show_density(dst_density_mask)


if __name__ == '__main__':
    getDensity('hkb')
