import os
import cv2
import argparse
import numpy as np
import os.path as osp
from getGTBox import getGTBox
from tqdm import tqdm

def _myaround_up(value):
    """0.05 * stride = 0.8"""
    tmp = np.floor(value).astype(np.int32)
    return tmp + 1 if value - tmp > 0.05 else tmp


def _myaround_down(value):
    """0.05 * stride = 0.8"""
    tmp = np.ceil(value).astype(np.int32)
    return max(0, tmp - 1 if tmp - value > 0.05 else tmp)


def Mask(img, bboxes, mask_scale=(30, 40)):
    try:
        height, width = img.shape[:2]

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
        print(img_path)
        return None


def getMask():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--type', default='xml', type=str,
                        choices=['xml', 'json', 'txt'], help='annotation type')
    opt = parser.parse_args()

    root_path = 'G:/CV/Reading/ClusterRegionGenerationNetwork/data/hkb'
    imgs_path = osp.join(root_path, 'JPEGImages')
    imgs_list = os.listdir(imgs_path)
    mask_path = osp.join(root_path, 'Mask')

    if not osp.exists(mask_path):
        os.mkdir(mask_path)

    for img_id in tqdm(imgs_list):
        img_path = osp.join(imgs_path, img_id)
        img = cv2.imread(img_path)

        if opt.type == 'json':
            anno_path = ""
            bboxes = getGTBox(anno_path, img_id)
        else:
            anno_path = img_path.replace('JPEGImages', 'Annotations').replace('jpg', opt.type)
            bboxes = getGTBox(anno_path)

        region_mask = Mask(img, bboxes)

        region_maskname = osp.join(mask_path, img_path[:-4] + '_region.png')
        cv2.imwrite(region_maskname, region_mask)

        if opt.show:
            cv2.imshow(region_mask)


if __name__ == '__main__':
    getMask()
