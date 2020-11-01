"""
use rondom array replace objce witch was neglected
"""
import os
import cv2
import sys
import json
import argparse
import numpy as np
import os.path as osp
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

import utils
from datasets import get_dataset
user_dir = osp.expanduser('~')


def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--dataset', type=str, default='Visdrone',
                        choices=['DOTA', 'Visdrone', 'TT100K', 'UAVDT'], help='dataset name')
    parser.add_argument('--imgsets', type=str, default=['train'],
                        nargs='+', help='choose image set')
    parser.add_argument('--neglect', type=str, default=[],
                        nargs='+', help='random padding neglect box')
    parser.add_argument('--show', type=bool, default=False,
                        help="show image and chip box")
    args = parser.parse_args()
    args.mosaic_scales = ((150, 150), (150, 350), (350, 150), (350, 350))
    args.strides = ((100, 100), (100, 200), (200, 100), (200, 200))
    args.limit_num = 2  # 每个图像每个尺度的候选区域的最大数量
    args.limit_obj = 2  # 每个窗口的类别最小数量
    args.db_root = user_dir + f'/data/{args.dataset}/'
    return args


args = parse_args()
print(args)


class MakeDataset(object):
    def __init__(self):
        self.dataset = get_dataset(args.dataset, args.db_root)
        self.density_dir = self.dataset.density_voc_dir

        self.dest_datadir = args.db_root + '/Mosaic'
        self.image_dir = self.dest_datadir + '/JPEGImages'
        self.anno_dir = self.dest_datadir + '/Annotations'
        self.list_dir = self.dest_datadir + '/ImageSets/Main'
        self._init_path()

    def _init_path(self):
        if not osp.exists(self.dest_datadir):
            os.makedirs(self.dest_datadir)
            os.makedirs(self.image_dir)
            os.makedirs(self.anno_dir)
            os.makedirs(self.list_dir)

    def __call__(self):
        self.static_chip = {mscale: 0 for mscale in args.mosaic_scales}
        for imgset in args.imgsets:
            print("make {} detect dataset...".format(imgset))
            samples = self.dataset._load_samples(imgset)
            chip_ids = []
            chip_loc = dict()
            for i, sample in enumerate(samples):
                # if i % 100 != 0: continue
                img_id = osp.splitext(osp.basename(sample['image']))[0]
                sys.stdout.write('\rcomplete: {:d}/{:d} {:s}'
                                 .format(i + 1, len(samples), img_id))
                sys.stdout.flush()

                loc = self.make_chip(sample, imgset)
                for i in range(len(loc)):
                    chip_ids.append('{}_{}'.format(img_id, i))

            self.generate_imgset(chip_ids, imgset)

        print("region number of each scale: {}".format(self.static_chip))

    def add_mosaic(self, img_shape, scales=(100, 100), strides=None):
        """img_shape: w, h
        """
        if strides is not None:
            stride_w, stride_h = strides
        else:
            stride_w, stride_h = scales[0] // 2, scales[1] // 2
        num_w = img_shape[0] // stride_w - 1
        num_h = img_shape[1] // stride_h - 1
        shift_x = np.arange(0, num_w) * stride_w
        shift_y = np.arange(0, num_h) * stride_h
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel()+scales[0], shift_y.ravel()+scales[1]
        )).transpose().astype(np.int)

        return shifts

    def sort_out(self, chip, chip_gt, chip_label):
        gt = np.array(chip_gt)
        gt_area = np.product(gt[:, 2:4] - gt[:, :2], 1)
        # medianArea = np.median(gt_area)
        maxArea = np.max(gt_area)
        judge1 = maxArea < 0.5 * (chip[2] - chip[0]) * (chip[3] - chip[1])

        return judge1

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

                if len(chip_gt) > 0:
                    judge = self.sort_out(chip, chip_gt, chip_label)
                    if not judge:
                        chip_gt = []
                        chip_label = []
                    # chip_gt = np.array(chip_gt)[keep_idx]  # order 2
                    # chip_label = np.array(chip_label)[keep_idx]  # order 2
                chip_gt_list.append(chip_gt)
                chip_label_list.append(chip_label)
                chip_neglect_list.append(neglect_gt)

        return chip_gt_list, chip_label_list, chip_neglect_list

    def generate_imgset(self, img_list, imgset):
        op = 'a'
        if args.imgsets[0] == imgset:
            op = 'w'
        with open(osp.join(self.list_dir, 'all.txt'), op) as f:
            f.writelines([x + '\n' for x in img_list])
        print('\n%d images in all set.' % len(img_list))

    def make_xml(self, chip, bboxes, labels, image_name, chip_size):
        node_root = Element('annotation')

        node_folder = SubElement(node_root, 'folder')
        node_folder.text = args.dataset

        node_filename = SubElement(node_root, 'filename')
        node_filename.text = image_name

        node_object_num = SubElement(node_root, 'object_num')
        node_object_num.text = str(len(bboxes))

        node_location = SubElement(node_root, 'location')
        node_loc_xmin = SubElement(node_location, 'xmin')
        node_loc_xmin.text = str(int(chip[0]) + 1)
        node_loc_ymin = SubElement(node_location, 'ymin')
        node_loc_ymin.text = str(int(chip[1]) + 1)
        node_loc_xmax = SubElement(node_location, 'xmax')
        node_loc_xmax.text = str(int(chip[2]) + 1)
        node_loc_ymax = SubElement(node_location, 'ymax')
        node_loc_ymax.text = str(int(chip[3]) + 1)

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(chip_size[0])
        node_height = SubElement(node_size, 'height')
        node_height.text = str(chip_size[1])
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'

        for i, bbox in enumerate(bboxes):
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = str(labels[i])
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'

            # voc dataset is 1-based
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(int(bbox[0]) + 1)
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(int(bbox[1]) + 1)
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(int(bbox[2] + 1))
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(int(bbox[3] + 1))

        xml = tostring(node_root, encoding='utf-8')
        dom = parseString(xml)
        # print(xml)
        return dom

    def make_chip(self, sample, imgset):
        image = cv2.imread(sample['image'])
        height, width = sample['height'], sample['width']
        img_id = osp.splitext(osp.basename(sample['image']))[0]

        # 生成马赛克候选区域
        region_box = np.zeros((0, 4))
        for i, mscale in enumerate(args.mosaic_scales):
            region_box = np.vstack((
                region_box, self.add_mosaic((width, height), mscale, args.strides[i])))
        region_box = region_box.astype(np.int)

        if args.show:
            utils.show_image(image, np.array(region_box))

        gt_bboxes, gt_cls = sample['bboxes'], sample['cls']

        chip_gt_list, chip_label_list, neglect_list = self.generate_region_gt(
            region_box, gt_bboxes, gt_cls)

        chip_loc = self.write_chip_and_anno(
            image, img_id, region_box,
            chip_gt_list, chip_label_list, neglect_list, imgset)

        return chip_loc

    def write_chip_and_anno(self, image, img_id,
                            chip_list, chip_gt_list,
                            chip_label_list, neglect_list, imgset):
        """write chips of one image to disk and make xml annotations
        """
        chip_num = 0
        chip_loc = dict()
        acount = {mscale: 0 for mscale in args.mosaic_scales}

        # sort
        chip_gt_list = np.array(chip_gt_list)
        chip_label_list = np.array(chip_label_list)
        neglect_list = np.array(neglect_list)
        gt_num = np.array([len(v) for v in chip_label_list])
        sort_idx = (-gt_num).argsort()
        chip_list = chip_list[sort_idx]
        chip_gt_list = chip_gt_list[sort_idx]
        chip_label_list = chip_label_list[sort_idx]
        neglect_list = neglect_list[sort_idx]

        for idx, chip in enumerate(chip_list):
            if len(chip_gt_list[idx]) < args.limit_obj:
                break
            if acount[(chip[2]-chip[0], chip[3]-chip[1])] >= args.limit_num:
                continue
            acount[(chip[2]-chip[0], chip[3]-chip[1])] += 1
            self.static_chip[(chip[2]-chip[0], chip[3]-chip[1])] += 1
            img_name = '{}_{}.jpg'.format(img_id, chip_num)
            xml_name = '{}_{}.xml'.format(img_id, chip_num)
            chip_loc[img_name] = [int(x) for x in chip]
            chip_size = (chip[2] - chip[0], chip[3] - chip[1])  # w, h
            chip_img = image[chip[1]:chip[3], chip[0]:chip[2], :].copy()
            assert len(chip_img.shape) == 3

            if imgset in args.neglect and neglect_list is not None:
                for neg_box in neglect_list[idx]:
                    neg_box = np.array(neg_box, dtype=np.int)
                    neg_w = neg_box[2] - neg_box[0]
                    neg_h = neg_box[3] - neg_box[1]
                    zeros_box = np.zeros((neg_h, neg_w, 3))
                    chip_img[neg_box[1]:neg_box[3], neg_box[0]:neg_box[2], :] = zeros_box

            bbox = np.array(chip_gt_list[idx], dtype=np.int)
            label = np.array(chip_label_list[idx])
            dom = self.make_xml(chip, bbox, label, img_name, chip_size)
            with open(osp.join(self.anno_dir, xml_name), 'w') as f:
                f.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8')) 
            # utils.show_image(chip_img[..., ::-1], bbox)
            cv2.imwrite(osp.join(self.image_dir, img_name), chip_img)
            chip_num += 1

        return chip_loc


if __name__ == "__main__":
    makedataset = MakeDataset()
    makedataset()
