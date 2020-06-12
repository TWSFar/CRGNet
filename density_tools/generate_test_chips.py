"""
use rondom array replace objce witch was neglected
"""
import os
import cv2
import sys
import h5py
import json
import joblib
import shutil
import argparse
import numpy as np
import os.path as osp

import utils
user_dir = osp.expanduser('~')


def parse_args():
    parser = argparse.ArgumentParser(description="convert to chip dataset")
    parser.add_argument('--dataset', type=str, default='Visdrone',
                        choices=['DOTA', 'Visdrone'], help='dataset name')
    parser.add_argument('--test_dir', type=str,
                        default=user_dir+"/data/Visdrone/challenge")
                        # default="E:\\CV\\data\\Underwater\\test")
    parser.add_argument('--aim', type=int, default=100,
                        help='gt aim scale in chip')
    parser.add_argument('--show', type=bool, default=False,
                        help="show image and chip box")
    args = parser.parse_args()
    return args


args = parse_args()
print(args)


class MakeDataset(object):
    def __init__(self):
        self.img_dir = osp.join(args.test_dir, "images")
        self.mask_dir = osp.join(args.test_dir, "density_mask")
        self.chip_dir = osp.join(args.test_dir, "density_chip")
        self.loc_dir = osp.join(args.test_dir, "density_loc")
        self.gbm = joblib.load('/home/twsf/work/CRGNet/density_tools/gbm_{}_{}_2.pkl'.format(args.dataset.lower(), args.aim))
        self._init_path()

    def _init_path(self):
        if osp.exists(self.chip_dir):
            shutil.rmtree(self.chip_dir)
        os.makedirs(self.chip_dir)
        if not osp.exists(self.loc_dir):
            os.makedirs(self.loc_dir)

    def __call__(self):
        print("make test detect dataset...")
        chip_ids = []
        chip_loc = dict()
        img_list = os.listdir(self.img_dir)
        assert len(img_list) > 0
        for i, img_name in enumerate(img_list):
            img_id = osp.splitext(osp.basename(img_name))[0]
            sys.stdout.write('\rcomplete: {:d}/{:d} {:s}'
                             .format(i + 1, len(img_list), img_id))
            sys.stdout.flush()

            loc = self.make_chip(img_name)
            for i in range(len(loc)):
                chip_ids.append('{}_{}'.format(img_id, i))
            chip_loc.update(loc)

        # wirte chip loc json
        with open(osp.join(self.loc_dir, 'test_chip.json'), 'w') as f:
            json.dump(chip_loc, f)
            print('write loc json')

    def make_chip(self, img_name):
        image = cv2.imread(osp.join(self.img_dir, img_name))
        height, width = image.shape[:2]
        img_id = osp.splitext(osp.basename(img_name))[0]
        # mask_path = ""
        mask_path = osp.join(self.mask_dir, '{}.hdf5'.format(img_id))
        with h5py.File(mask_path, 'r') as hf:
            mask = np.array(hf['label'])
        mask_h, mask_w = mask.shape[:2]

        # make chip
        region_box, contours = utils.generate_box_from_mask(mask)
        # utils.show_image(mask, np.array(region_box))
        region_box = utils.generate_crop_region(region_box, mask, (mask_w, mask_h), (width, height), self.gbm)
        # utils.show_image(mask, np.array(region_box))
        region_box = utils.resize_box(region_box, (mask_w, mask_h), (width, height))

        if len(region_box) == 0:
            region_box = np.array([[0, 0, width, height]])
        if args.show:
            utils.show_image(image, region_box)

        chip_loc = self.write_chip_and_anno(image, img_id, region_box)

        return chip_loc

    def write_chip_and_anno(self, image, img_id, chip_list):
        """write chips of one image to disk and make xml annotations
        """
        chip_loc = dict()
        for i, chip in enumerate(chip_list):
            img_name = '{}_{}.jpg'.format(img_id, i)
            chip_loc[img_name] = [int(x) for x in chip]
            chip_img = image[chip[1]:chip[3], chip[0]:chip[2], :].copy()
            assert len(chip_img.shape) == 3

            cv2.imwrite(osp.join(self.chip_dir, img_name), chip_img)

        return chip_loc


if __name__ == "__main__":
    makedataset = MakeDataset()
    makedataset()
