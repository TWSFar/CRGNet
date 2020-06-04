"""
use rondom array replace objce witch was neglected
"""
import os
import cv2
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from regress import utils
from datasets import get_dataset
import matplotlib.pyplot as plt
user_dir = os.path.expanduser('~')


def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--dataset', type=str, default='DOTA',
                        choices=['VisDrone', 'DOTA'], help='dataset name')
    parser.add_argument('--db_root', type=str,
                        # default="G:\\CV\\Dataset\\Detection\\Visdrone",
                        default="/home/twsf/data/DOTA",
                        help="dataset's root path")
    parser.add_argument('--imgsets', type=str, default=['val'],
                        nargs='+', help='for train or val')
    parser.add_argument('--aim', type=int, default=100,
                        help='gt aim scale in chip')
    parser.add_argument('--show', type=bool, default=False,
                        help="show image and chip box")
    args = parser.parse_args()
    return args


args = parse_args()
print(args)
result_dir = "density_tools/statistic_results"
if not osp.exists(result_dir):
    os.mkdir(result_dir)


class ChipStatistics(object):
    def __init__(self):
        self.dataset = get_dataset(args.dataset, args.db_root)
        self.density_dir = self.dataset.density_voc_dir
        self.segmentation_dir = self.density_dir + '/SegmentationClass'

    def __call__(self):
        for imgset in args.imgsets:
            self.box_scale = []
            self.chip_scale = []
            self.chip_percent = []
            self.chip_box_scale = []
            self.info = []
            print("make {} detect dataset...".format(imgset))
            samples = self.dataset._load_samples(imgset)
            for i, sample in enumerate(tqdm(samples)):
                # if i < 5000 and i % 10 != 0: continue
                self.make_chip(sample, imgset)

            # plot
            box_scale_coco = np.round(np.sqrt(np.array(self.box_scale)*640*480))
            scale_distribution = np.bincount(box_scale_coco.astype(np.int) // 5, minlength=20)
            x_axis = [str(i)+'~'+str(i+5) for i in range(0, 100, 5)]
            y_axis = scale_distribution[:20]
            plt.figure(figsize=(14, 8))
            plt.grid()
            # plt.title('object scale distribution')
            plt.xlabel('object scale')
            plt.ylabel('numbers')
            plt.ylim((0, 40000))
            plt.bar(x_axis, y_axis)
            plt.savefig(osp.join(result_dir, "scale_distribution.png"), dpi=600)
            plt.show()
            plt.close()

            # results = np.array(results)
            chip_scale = np.array(self.chip_scale)
            box_scale = np.array(self.box_scale)
            small_obj = (box_scale_coco <= 32).sum()
            median_obj = (box_scale_coco <= 96).sum() - small_obj
            large_obj = (box_scale_coco > 96).sum()
            nor_box_scale = (np.array(self.box_scale) - min(self.box_scale)) / (max(self.box_scale) - min(self.box_scale))
            with open(result_dir+'/gtArea_mean_std.info', 'a') as f:
                f.writelines('chip num: '+str(len(chip_scale)) + '\n')
                f.writelines('chip percent: '+str(chip_scale[:, 1].mean()) + '\n')
                f.writelines('box num: '+str(len(self.box_scale)) + '\n')
                f.writelines('box mean: '+str(box_scale.mean()) + '\n')
                f.writelines('box nom_mean: '+str(np.sqrt(box_scale.mean()*480*640)) + '\n')
                f.writelines('box mae: '+str((abs(box_scale_coco - box_scale_coco.mean()).mean())) + '\n')
                f.writelines('box_std: '+str(np.std(self.box_scale)) + '\n')
                f.writelines('box_std_2: '+str(np.std(box_scale_coco)) + '\n')
                f.writelines('box_nstd: '+str(np.std(nor_box_scale)) + '\n')
                f.writelines('small_obj: '+str(small_obj) + '\n')
                f.writelines('median_obj: '+str(median_obj) + '\n')
                f.writelines('large_obj: '+str(large_obj) + '\n')
                for i in range(0, 20):
                    f.writelines('scale {} sum: {}'.format(x_axis[i], scale_distribution[i:i+1].sum()) + '\n')

            with open(result_dir+'/{}_{}.csv'.format(args.dataset, imgset), 'w') as f:
                for line in self.info:
                    for i, v in enumerate(line):
                        if i > 0:
                            f.writelines(',')
                        f.writelines(str(v))
                    f.writelines('\n')

    def generate_region_gt(self, region_box, gt_bboxes, labels):
        chip_list = []
        for box in region_box:
            chip_list.append(np.array(box))

        # chip gt
        chip_gt_list = []
        chip_label_list = []
        chip_neglect_list = []
        if gt_bboxes is not None:
            for chip in chip_list:
                chip_gt = []
                chip_label = []
                neglect_gt = []
                for i, box in enumerate(gt_bboxes):
                    if utils.overlap(chip, box, 0.75):
                        box = [max(box[0], chip[0]), max(box[1], chip[1]),
                               min(box[2], chip[2]), min(box[3], chip[3])]
                        new_box = [box[0] - chip[0], box[1] - chip[1],
                                   box[2] - chip[0], box[3] - chip[1]]
                        chip_gt.append(np.array(new_box))
                        chip_label.append(labels[i])
                    elif utils.overlap(chip, box, 0.1):
                        box = [max(box[0], chip[0]), max(box[1], chip[1]),
                               min(box[2], chip[2]), min(box[3], chip[3])]
                        new_box = [box[0] - chip[0], box[1] - chip[1],
                                   box[2] - chip[0], box[3] - chip[1]]
                        neglect_gt.append(np.array(new_box, dtype=np.int))

                chip_gt_list.append(chip_gt)
                chip_label_list.append(chip_label)
                chip_neglect_list.append(neglect_gt)

        return chip_gt_list, chip_label_list, chip_neglect_list

    def make_chip(self, sample, imgset, gbm=None):
        image = cv2.imread(sample['image'])
        # utils.show_image(image[..., ::-1])
        height, width = sample['height'], sample['width']
        img_id = osp.splitext(osp.basename(sample['image']))[0]

        mask_path = osp.join(self.segmentation_dir, '{}.hdf5'.format(img_id))
        with h5py.File(mask_path, 'r') as hf:
            mask = np.array(hf['label'])
        mask_h, mask_w = mask.shape[:2]

        # make chip
        region_box = utils.generate_box_from_mask(mask)

        region_box, info = utils.generate_crop_region(region_box, mask, (mask_w, mask_h), (width, height), gbm)

        region_box = utils.resize_box(region_box, (mask_w, mask_h), (width, height))

        # info = []
        # region_box = np.array([[0, 0, width, height]])

        if args.show:
            utils.show_image(image, np.array(region_box))

        gt_bboxes, gt_cls = sample['bboxes'], sample['cls']

        chip_gt_list, chip_label_list, neglect_list = self.generate_region_gt(
            region_box, gt_bboxes, gt_cls)
        chip_loc = self.write_chip_and_anno(info,
            image, img_id, region_box, chip_gt_list, chip_label_list, neglect_list, imgset)

        return chip_loc

    def write_chip_and_anno(self, info, image, img_id,
                            chip_list, chip_gt_list,
                            chip_label_list, neglect_list, imgset):
        """write chips of one image to disk and make xml annotations
        """
        chip_loc = dict()
        chip_num = 0
        width, height = image.shape[:2]
        for i, chip in enumerate(chip_list):
            if len(chip_gt_list[i]) == 0:
                continue
            chip_size = (chip[2] - chip[0], chip[3] - chip[1])  # w, h
            chip_img = image[chip[1]:chip[3], chip[0]:chip[2], :].copy()
            assert len(chip_img.shape) == 3

            bbox = np.array(chip_gt_list[i], dtype=np.int)
            chip_num += 1

            area = chip_size[0] * chip_size[1]
            area_ratio = np.product(1.0*(bbox[:, 2:4]-bbox[:, :2]), axis=1) / area

            self.box_scale.extend(list(area_ratio))
            self.chip_scale.append([area, 1.0*area/(width*height)])
            self.chip_box_scale.append(np.median(area_ratio))
            self.info.append(info[i] + [image.shape[0]*image.shape[1], np.mean(area_ratio)])

        return chip_loc


if __name__ == "__main__":
    cs = ChipStatistics()
    cs()