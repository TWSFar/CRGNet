import os
import cv2
import sys
import argparse
import numpy as np
import os.path as osp

import utils
from datasets import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--dataset', type=str, default='Visdrone',
                        choices=['VisDrone'], help='dataset name')
    parser.add_argument('--db_root', type=str, default="/home/twsf/data/Visdrone",
                        help="dataset's root path")
    parser.add_argument('--imgsets', type=list, default=['train', 'val'],
                        choices=['train', 'val', 'test'], help='for train or test')
    args = parser.parse_args()
    return args

args = parse_args()


class MakeDataset(object):
    def __init__(self):
        self.dataset = get_dataset(args.dataset, args.db_root)

        self.region_dir = dataset.region_voc_dir
        self.segmentation_dir = region_dir + '/SegmentationClass'

        self.dest_datadir = dataset.detect_voc_dir
        self.image_dir = dest_datadir + '/JPEGImages'
        self.anno_dir = dest_datadir + '/Annotations'
        self.list_dir = dest_datadir + '/ImageSets/Main'
        self._init_path()

    def _init_path(self):
        if not osp.exists(self.dest_datadir):
            os.makedirs(self.dest_datadir)
            os.makedirs(self.image_dir)
            os.makedirs(self.anno_dir)
            os.makedirs(self.list_dir)


    def __call__(self):
        for imgset in args.imgsets:
            print("make {} detect dataset...".format(imgset))
            samples = dataset._load_samples(imgset)
            chip_ids= []
            chip_lov = dict()
            for i, sample in enumerate(samples):
                img_id = osp.basename(sample['image'])[:-4]
                sys.stdout.write('\rcomplete: {:d}/{:d} {:s}'
                            .format(i + 1, len(samples), img_id))
                sys.stdout.flush()
    
                chiplen, loc = self.make_chip(sample, imgset)

    def generate_region_gt(self):
        pass

    def make_xml(self):
        pass

    def write_chip_and_anno(self):
        pass

    def generate_imgset(self):
        pass

    def make_chip(self, sample, imgset):
        image = cv2.imread(sample['image'])
        height, width = sample['height'], sample['width']
        img_id = osp.basename(sample['image'])[:-4]

        mask_path = osp.join(self.segmentation_dir, img_id+'_region.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_h, mask_w = mask.shape[:2]

        region_box, contours = utils.generate_box_from_mask(mask)
        region_box = utils.region_postprocess(region_box, contours(mask_w, mask_h))
        region_box = utils.resize_box(region_box, (mask_w, mask_h))
        region_box = utils.generate_crop_region(region_box, (width, height))

        if imgset == 'train':
            region_box = np.vstack((region_box, np.array([0, 0, width-1, height-1])))

