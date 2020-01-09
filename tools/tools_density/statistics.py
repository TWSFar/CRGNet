"""
use rondom array replace objce witch was neglected
"""
import os
import cv2
import pdb
import sys
import argparse
import numpy as np
import os.path as osp
from scipy import stats
import utils
from datasets import get_dataset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
user_dir = os.path.expanduser('~')


def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--dataset', type=str, default='VisDrone',
                        choices=['VisDrone'], help='dataset name')
    parser.add_argument('--db_root', type=str,
                        default="G:\\CV\\Dataset\\Detection\\Visdrone",
                        help="dataset's root path")
    parser.add_argument('--imgsets', type=str, default=['train',],
                        nargs='+', help='for train or test')
    parser.add_argument('--padding', type=str, default=[],
                        nargs='+', help='random padding neglect box')
    args = parser.parse_args()
    return args


args = parse_args()
print(args)
result_dir = "G:\\CV\\Reading\\CRGNet\\tools\\tools_density\\result"
if not osp.exists(result_dir):
    os.mkdir(result_dir)


class Statistics(object):
    def __init__(self):
        self.dataset = get_dataset(args.dataset, args.db_root)

        self.density_dir = self.dataset.density_voc_dir
        self.segmentation_dir = self.density_dir + '/SegmentationClass'

    def __call__(self):
        for imgset in args.imgsets:
            print("make {} detect dataset...".format(imgset))
            samples = self.dataset._load_samples(imgset)
            chip_obj_density = []
            chip_obj_scale = []
            chip_obj_weight = []
            chip_obj_weight3D = []
            for i, sample in enumerate(samples):
                # if '0000100_03906_d_0000014' not in sample["image"]:
                #     continue
                # if i > 3: break
                img_id = osp.basename(sample['image'])[:-4]

                mask, chips = self.get_ChipAndMask(sample, imgset)
                if mask.sum() == 0:
                    continue

                density, scale, weight, weight3D = self.statistics(mask, chips)
                chip_obj_density.extend(density)
                chip_obj_scale.extend(scale)
                chip_obj_weight.extend(weight)
                chip_obj_weight3D.extend(weight3D)

                sys.stdout.write('\rcomplete: {:d}/{:d} {:s}'
                                 .format(i + 1, len(samples), img_id))
                sys.stdout.flush()

            result_list = dict(
                Scale=np.array(chip_obj_scale),
                Density=np.array(chip_obj_density),
                Weight=np.array(chip_obj_weight),
                ChipNobj=np.array(chip_obj_weight3D)[:, 0],
                ChipArea=np.array(chip_obj_weight3D)[:, 1],
            )

            result_file = osp.join(result_dir, "chip_informations.txt")
            f = open(result_file, 'w')
            for state_name, state_list in result_list.items():
                f.writelines('{}\n'.format(state_name))
                f.writelines('\tmax: {}\n'.format(state_list.max()))
                f.writelines('\tmin: {}\n'.format(state_list.min()))
                f.writelines('\tmean: {}\n'.format(state_list.mean()))

            # plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            chip_obj_weight3D = np.array(chip_obj_weight3D)
            xs = chip_obj_weight3D[:, 0]
            ys = chip_obj_weight3D[:, 1]
            zs = chip_obj_weight3D[:, 2]
            # 去除重复
            max_xs = int(xs.max()+1)
            max_ys = int(ys.max()+1)
            obj_weight3D = np.zeros((max_xs, max_ys))
            for x, y, z in zip(xs, ys, zs):
                obj_weight3D[int(x), int(y)] = z
            xs, ys, zs = [], [], []
            for x, line in enumerate(obj_weight3D):
                for y, z in enumerate(line):
                    if z != 0:
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)

            ax.scatter(xs, ys, zs, c='r')
            ax.set_xlabel('number')
            ax.set_ylabel('area')
            ax.set_zlabel('weight')

            plt.show()

    def get_ChipAndMask(self, sample, imgset):
        img_id = osp.basename(sample['image'])[:-4]

        mask_path = osp.join(self.segmentation_dir, '{}.png'.format(img_id))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_h, mask_w = mask.shape[:2]

        # make chip
        region_box, contours = utils.generate_box_from_mask(mask)
        region_box = utils.region_postprocess(
            region_box, contours, (mask_w, mask_h))
        # region_box = utils.generate_crop_region(
        #     region_box, (mask_w, mask_h))

        return mask, region_box

    def statistics(self, mask, chips):
        scale = []
        density = []
        weight = []
        weight3D = []
        for chip in chips:
            mask_chip = mask[chip[1]:chip[3]+1, chip[0]:chip[2]+1]
            chip_area = np.where(mask_chip > 0, 1, 0).sum()
            chip_nobj = mask_chip.sum()
            if chip_area == 0 or chip_nobj == 0:
                utils.show_image(mask, chip[None, :])
                continue
            # utils.show_image(mask, chips[:])
            # temp = chip_area ** 0.1 * np.exp(chip_area/chip_nobj)
            # temp = np.log(1 + chip_area/chip_nobj) + 1
            temp = np.log(1 + chip_area ** 1.5 / (chip_nobj * 35)) + 1
            weight.append(temp)
            weight3D.append([chip_nobj, chip_area, temp])
            scale.append(chip_area / chip_nobj)
            density.append(chip_nobj / chip_area)

        return density, scale, weight, weight3D


if __name__ == "__main__":
    statistics = Statistics()
    statistics()
