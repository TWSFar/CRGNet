import os
import cv2
import argparse
import numpy as np
import os.path as osp
from getGTBox import getGTBox


def _myaround_up(value):
    """0.05 * stride = 0.8"""
    tmp = np.floor(value).astype(np.int32)
    return tmp + 1 if value - tmp > 0.05 else tmp


def _myaround_down(value):
    """0.05 * stride = 0.8"""
    tmp = np.ceil(value).astype(np.int32)
    return max(0, tmp - 1 if tmp - value > 0.05 else tmp)


def Mask(img_path, mask_path):
    try:
        img_name = osp.basename(img_path)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # Chip mask 40 * 30, model input size 640x480
        mask_w, mask_h = 40, 30
        region_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        dense_mask = np.zeros((mask_h, mask_w), dtype=np.double)
        boxes = getGTBox(img_path)
        for box in boxes:
            xmin = _myaround_down(1.0 * box[0] / width * mask_w)
            ymin = _myaround_down(1.0 * box[1] / height * mask_h)
            xmax = _myaround_up(1.0 * box[2] / width * mask_w)
            ymax = _myaround_up(1.0 * box[3] / height * mask_h)
            region_mask[ymin:ymax, xmin:xmax] = 1
            dense_mask[ymin:ymax, xmin:xmax] += 1
        dense_mask = cv2.GaussianBlur(dense_mask, (5, 5), 0)
        region_maskname = osp.join(mask_path, img_name[:-4] + '_region.png')
        dense_maskname = osp.join(mask_path, img_name[:-4] + '_dense.png')
        cv2.imwrite(region_maskname, region_mask)
        cv2.imwrite(dense_maskname, dense_mask)
        if  opt.show:
            dense_mask = dense_mask / dense_mask.max() * 255
            dense_mask = dense_mask.repeat(3, 1)
            dense_mask = dense_mask.reshape(mask_h, mask_w, 3)
            dense_mask[..., :2] = 0 
            mask_show = cv2.resize(dense_mask, (400, 300)).astype(np.uint8)
            cv2.imshow(osp.basename(dense_maskname), mask_show)
            cv2.waitKey(0)

    except Exception as e:
        print(e)
        print(img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    opt = parser.parse_args()

    root_path = 'G:/CV/Reading/ClusterRegionGenerationNetwork/data/hkb'
    img_path = osp.join(root_path, 'JPEGImages')
    img_list = os.listdir(img_path)
    mask_path = osp.join(root_path, 'Mask')
    if not osp.exists(mask_path):
        os.mkdir(mask_path)

    for img_id in img_list:
        img = osp.join(img_path, img_id)
        Mask(img, mask_path)
