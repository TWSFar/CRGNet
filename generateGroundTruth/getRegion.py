import os
import cv2
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from mypath import Path
from dataloader.datasets import Datasets


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


def _myaround_up(value):
    """0.05 * stride = 0.8"""
    tmp = np.floor(value).astype(np.int32)
    return tmp + 1 if value - tmp > 0.05 else tmp


def _myaround_down(value):
    """0.05 * stride = 0.8"""
    tmp = np.ceil(value).astype(np.int32)
    return max(0, tmp - 1 if tmp - value > 0.05 else tmp)


def Region(bboxes, img_scale, mask_scale=(30, 40)):
    try:
        height, width = img_scale

        # Chip mask 40 * 30, model input size 640x480
        mask_h, mask_w = mask_scale
        region_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

        for box in bboxes:
            xmin = _myaround_down(1.0 * box[0] / width * mask_w)
            ymin = _myaround_down(1.0 * box[1] / height * mask_h)
            xmax = _myaround_up(1.0 * box[2] / width * mask_w)
            ymax = _myaround_up(1.0 * box[3] / height * mask_h)
            region_mask[ymin:ymax, xmin:xmax] = 1

        return region_mask

    except Exception as e:
        print(e)
        return None


def getRegion(dataset_name='visdrone'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    opt = parser.parse_args()

    dataset = Datasets(dataset_name)

    mask_path = osp.join(dataset.root_path, 'RegionMask')
    if not osp.exists(mask_path):
        os.mkdir(mask_path)

    for i, sample in enumerate(tqdm(dataset.samples)):
        img_scale = (sample['height'], sample['width'])
        region_mask = Region(sample['bboxes'], img_scale)
        maskname = osp.join(mask_path, osp.basename(sample['image']).
                            replace(dataset.img_type, '.png'))
        cv2.imwrite(maskname, region_mask)

        if opt.show:
            img = cv2.imread(sample['image'])
            show_image(img, sample['bboxes'])
            show_image(img, sample['bboxes'])


if __name__ == '__main__':
    getRegion()
