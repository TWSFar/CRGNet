"""
use rondom array replace objce witch was neglected
"""
import os
import cv2
import sys
import json
import h5py
import joblib
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
    # parser.add_argument('--db_root', type=str,
    #                     default=user_dir+"/data/TT100K/",
    #                     # default="E:\\CV\\data\\visdrone",
    #                     help="dataset's root path")
    parser.add_argument('--imgsets', type=str, default=['train', 'test', 'val'],
                        nargs='+', help='for train or val')
    parser.add_argument('--aim', type=int, default=100,
                        help='gt aim scale in chip')
    parser.add_argument('--augment', type=bool, default=True,
                        help='augmentation dataset by pasting class mask')
    parser.add_argument('--padding', type=str, default=[],
                        nargs='+', help='random padding neglect box')
    parser.add_argument('--show', type=bool, default=False,
                        help="show image and chip box")
    args = parser.parse_args()
    args.db_root = user_dir + f'/data/{args.dataset}/'
    return args


args = parse_args()
print(args)
hpy = {
    "kernel_size": (5, 5),  # road的腐蚀核大小
    "fx": 0.1,  # 候选点之前的road下采样倍数x
    "fy": 0.1,  # 候选点之前的road下采样倍数y
    "pasting_maximum": 5,  # 一个chip最多粘贴数量
    "adjustLumin": 2,  # 背景差异除以此数
    "obt": 0.01,  # 粘贴目标允许和其他的目标覆盖阈值
    "ort": 0.8,  # 路径覆盖粘贴目标的阈值
}
i = 1


class MakeDataset(object):
    def __init__(self):
        self.dataset = get_dataset(args.dataset, args.db_root)
        self.debug = 0
        self.density_dir = self.dataset.density_voc_dir
        self.roadMask_dir = osp.join(args.db_root, "road_mask")
        self.segmentation_dir = self.density_dir + '/SegmentationClass'

        self.dest_datadir = self.dataset.detect_voc_dir
        self.image_dir = self.dest_datadir + '/JPEGImages'
        self.anno_dir = self.dest_datadir + '/Annotations'
        self.list_dir = self.dest_datadir + '/ImageSets/Main'
        self.maskPools_dir = args.db_root + '/paster_pool'
        self.loc_dir = self.dest_datadir + '/Locations'
        self.gbm = joblib.load(user_dir + '/work/CRGNet/density_tools/weights/gbm_{}_{}.pkl'.format(args.dataset.lower(), args.aim))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, hpy["kernel_size"])
        if args.augment:
            self.getMaskPools()
        self._init_path()

    def _init_path(self):
        if not osp.exists(self.dest_datadir):
            os.makedirs(self.dest_datadir)
            os.makedirs(self.image_dir)
            os.makedirs(self.anno_dir)
            os.makedirs(self.list_dir)
            os.makedirs(self.loc_dir)

    def __call__(self):
        for imgset in args.imgsets:
            print("make {} detect dataset...".format(imgset))
            samples = self.dataset._load_samples(imgset)
            chip_ids = []
            chip_loc = dict()
            for i, sample in enumerate(samples):
                # if i < 950: continue
                img_id = osp.splitext(osp.basename(sample['image']))[0]
                sys.stdout.write('\rcomplete: {:d}/{:d} {:s}'
                                 .format(i + 1, len(samples), img_id))
                sys.stdout.flush()

                loc = self.make_chip(sample, imgset)
                for i in range(len(loc)):
                    chip_ids.append('{}_{}'.format(img_id, i))
                chip_loc.update(loc)

            self.generate_imgset(chip_ids, imgset)

            # wirte chip loc json
            with open(osp.join(self.loc_dir, imgset+'_chip.json'), 'w') as f:
                json.dump(chip_loc, f)
                print('write loc json')

    def augmentation(self, chip_img, loc, bbox, labels):
        acount = 0
        # height, widht = self.roadMask.shape[:2]
        road_chip = self.roadMask[loc[1]:loc[3], loc[0]:loc[2]].copy()
        # utils.show_image(road_chip)
        cand_chip = cv2.resize(road_chip, (0, 0), fx=hpy["fx"], fy=hpy["fy"], interpolation=cv2.INTER_NEAREST)
        cand_chip = cv2.erode(cand_chip, self.kernel)
        # utils.show_image(cand_chip)
        cand_points = np.argwhere(cand_chip[..., 0] == 255) * (road_chip.shape[0] / cand_chip.shape[0])

        np.random.shuffle(cand_points)
        raresIn = [rare for rare in self.rare_cls if rare in labels]
        for point in cand_points:
            obj_scales = bbox[:, 2:4] - bbox[:, :2]
            obj_center = (bbox[:, 2:4] + bbox[:, :2]) / 2
            self.debug += 1
            # if (self.debug >= 1514):
            adj_obj_idx = np.argmin(np.power(point - obj_center, 2).sum(1))
            adj_scale = obj_scales[adj_obj_idx]
            adj_label = labels[adj_obj_idx]

            adj_rank = self.scale_rank[adj_label]
            if acount < len(raresIn):
                pasting_cls = raresIn[acount]
            else:
                pasting_cls = np.random.choice(self.aid_cls[adj_label])
            scale_map = self.scale_rank[pasting_cls] / adj_rank
            scale = np.array(adj_scale * scale_map)

            # 寻找粘贴对象
            paster_cls_pool = np.array(self.paster_pool[pasting_cls])
            paster_shape = np.array(paster_cls_pool[:, -2:], dtype=np.float)
            idx1 = paster_shape[:, scale.argmax()] > paster_shape[:, scale.argmin()]
            idx2 = paster_shape[:, scale.argmax()] >= scale.max()
            idx = idx1 & idx2
            paster_cls_pool = paster_cls_pool[idx]
            if len(paster_cls_pool) == 0:
                continue
            paster_idx = np.random.choice(np.arange(len(paster_cls_pool)))
            paster_info = paster_cls_pool[paster_idx]
            paster_path = osp.join(self.maskPools_dir, "_".join([v for v in paster_info[:-2]]))
            paster = cv2.imread(paster_path)

            # 获取粘贴位置, 判断位置是否合法
            paster_box = self.getPastingLocal(chip_img, paster, scale, point)
            paster_area = np.product(paster_box[2:] - paster_box[:2])
            paster_road_area = road_chip[paster_box[1]:paster_box[3], paster_box[0]:paster_box[2]].sum()
            if utils.iou_calc1(paster_box, bbox).max() > hpy["obt"] or paster_road_area < paster_area * hpy["ort"]:
                continue

            # 调整明亮度, 粘贴
            paster = self.adjustLumin(chip_img, paster)
            chip_img = self.pasting(chip_img, paster, paster_box)
            # utils.show_image(chip_img, np.array([paster_box]))
            bbox = np.vstack((bbox, paster_box))
            labels = np.hstack((labels, int(pasting_cls)))
            acount += 1
            if acount > hpy["pasting_maximum"]:
                break

        return chip_img, bbox, labels

    def adjustLumin(self, chip, paster):
        # 得到mask中的非黑色索引，黑色的的rgb为0
        arraySum = np.sum(paster, axis=2)
        index1 = (arraySum > 0).nonzero()
        # 计算亮度均值
        avgChip = chip.mean()
        avgPaste = paster[index1[0], index1[1]].mean()
        diff = int(avgChip - avgPaste) // hpy["adjustLumin"]
        paster[index1[0], index1[1]] = np.clip(paster[index1[0], index1[1]]+diff, 0, 255).astype(np.uint8)
        return paster

    def pasting(self, chip_img, paster, local):
        scale = (local[2]-local[0], local[3]-local[1])
        new_paster = cv2.resize(paster, scale, cv2.INTER_NEAREST)
        b1f0 = np.where(new_paster > 0, 0, 1)  # >10是为了消除因为resize而在边缘产生的乱像素
        temp = chip_img[local[1]:local[3], local[0]:local[2], :]
        chip_img[local[1]:local[3], local[0]:local[2], :] = temp * b1f0 + new_paster

        return chip_img

    def getPastingLocal(self, chip_img, paster, scale, point):
        chip_h, chip_w = chip_img.shape[:2]
        paster_scale = (paster.shape[1], paster.shape[0])
        ratio = scale.max() / paster_scale[scale.argmax()]
        scale[scale.argmin()] = paster_scale[scale.argmin()] * ratio
        paster_hw, paster_hh = scale / 2.0
        past_box = [point[0] - paster_hw, point[1] - paster_hh, point[0] + paster_hw, point[1] + paster_hh]
        loc_box = np.array(utils.region_enlarge(past_box, (chip_w, chip_h), 1), dtype=np.int)

        return loc_box

    def getMaskPools(self):
        self.paster_pool = {}
        self.rare_cls = []
        for file in os.listdir(self.maskPools_dir):
            mask_h, mask_w = cv2.imread(osp.join(self.maskPools_dir, file)).shape[:2]
            info = file.split('_')
            if int(info[0]) in self.paster_pool:
                self.paster_pool[int(info[0])].append(info + [str(mask_w), str(mask_h)])
            else:
                self.paster_pool[int(info[0])] = [info + [str(mask_w), str(mask_h)]]
        for rare in self.paster_pool:
            self.rare_cls.append(int(rare))
        self.scale_rank = {0: 1, 1: 1, 2: 1.5, 3: 4, 4: 4, 5: 8, 6: 4, 7: 4, 8: 8, 9: 1.5}
        # 辅助类的rank必须小于粘贴类, 否则会出现resize到极小的情况
        self.aid_cls = {0: [2, 7], 1: [2, 6], 2: [6, 7], 3: [6, 7, 8], 4: [6, 7, 8], 5: [8], 6: [7], 7: [6], 8: [5], 9: [2]}

    def getRoadMask(self, img_id):
        self.roadMask = cv2.imread(osp.join(self.roadMask_dir, img_id+'.jpg'))

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

    def generate_imgset(self, img_list, imgset):
        with open(osp.join(self.list_dir, imgset+'.txt'), 'w') as f:
            f.writelines([x + '\n' for x in img_list])
        print('\n%d images in %s set.' % (len(img_list), imgset))
        if imgset.lower() != 'test':
            op = 'a'
            if args.imgsets[0] == imgset:
                op = 'w'
            with open(osp.join(self.list_dir, 'trainval.txt'), op) as f:
                f.writelines([x + '\n' for x in img_list])
            print('\n%d images in trainval set.' % len(img_list))

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

        mask_path = osp.join(self.segmentation_dir, '{}.hdf5'.format(img_id))
        with h5py.File(mask_path, 'r') as hf:
            mask = np.array(hf['label'])
        mask_h, mask_w = mask.shape[:2]

        # make chip
        region_box, contours = utils.generate_box_from_mask(mask)

        region_box = utils.generate_crop_region(region_box, mask, (mask_w, mask_h), (width, height), self.gbm)

        region_box = utils.resize_box(region_box, (mask_w, mask_h), (width, height))

        if args.show:
            utils.show_image(image, np.array(region_box))

        # if imgset == 'train':
        #     if len(region_box):
        #         region_box = np.vstack((region_box, np.array([0, 0, width, height])))
        #     else:
        #         region_box = np.array([[0, 0, width, height]])

        gt_bboxes, gt_cls = sample['bboxes'], sample['cls']

        chip_gt_list, chip_label_list, neglect_list = self.generate_region_gt(
            region_box, gt_bboxes, gt_cls)

        chip_loc = self.write_chip_and_anno(
            image, img_id, region_box, chip_gt_list, chip_label_list, neglect_list, imgset)

        return chip_loc

    def write_chip_and_anno(self, image, img_id,
                            chip_list, chip_gt_list,
                            chip_label_list, neglect_list, imgset):
        """write chips of one image to disk and make xml annotations
        """
        chip_loc = dict()
        chip_num = 0
        if args.augment:
            self.getRoadMask(img_id)

        for i, chip in enumerate(chip_list):
            if len(chip_gt_list[i]) == 0:
                continue
            img_name = '{}_{}.jpg'.format(img_id, chip_num)
            xml_name = '{}_{}.xml'.format(img_id, chip_num)
            chip_loc[img_name] = [int(x) for x in chip]
            chip_size = (chip[2] - chip[0], chip[3] - chip[1])  # w, h
            chip_img = image[chip[1]:chip[3], chip[0]:chip[2], :].copy()
            assert len(chip_img.shape) == 3

            if imgset in args.padding and neglect_list is not None:
                for neg_box in neglect_list[i]:
                    neg_w = neg_box[2] - neg_box[0]
                    neg_h = neg_box[3] - neg_box[1]
                    random_box = np.random.randint(0, 256, (neg_h, neg_w, 3))
                    chip_img[neg_box[1]:neg_box[3], neg_box[0]:neg_box[2], :] = random_box

            bbox = np.array(chip_gt_list[i], dtype=np.int)
            label = np.array(chip_label_list[i], dtype=np.int)

            if args.augment:
                chip_img, bbox, label = self.augmentation(chip_img, chip, bbox, label)

            dom = self.make_xml(chip, bbox, label, img_name, chip_size)
            with open(osp.join(self.anno_dir, xml_name), 'w') as f:
                f.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8')) 

            cv2.imwrite(osp.join(self.image_dir, img_name), chip_img)
            chip_num += 1

        return chip_loc


if __name__ == "__main__":
    makedataset = MakeDataset()
    makedataset()
