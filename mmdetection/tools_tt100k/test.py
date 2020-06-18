import os
import json
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from mmdet.ops import nms
from mmdet.core import eval_map
import xml.etree.ElementTree as ET
from mmdet.apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Test chip')
    parser.add_argument('--checkpoint', default='/home/twsf/work/CRGNet/mmdetection/tools_tt100k/work_dirs/ATSS_x101_fpn_giou/epoch_60.pth')
    parser.add_argument('--config', default="/home/twsf/work/CRGNet/mmdetection/tools_tt100k/configs/density/ATSS_x101_fpn_giou.py")
    parser.add_argument('--root-dir', default='/home/twsf/data/TT100K')
    parser.add_argument('--nclass', default=45, type=int)
    parser.add_argument('--score_thr', default=0.3, type=float)
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--result-path', default='/home/twsf/work/CRGNet/workshops')
    args = parser.parse_args()
    args.chip_dir = args.root_dir + '/density_chip'
    return args


def getGtFromXml(xml_file):
    box_all = []
    gt_cls = []
    xml = ET.parse(xml_file).getroot()
    pts = ['xmin', 'ymin', 'xmax', 'ymax']
    # bounding boxes
    for obj in xml.iter('object'):
        bbox = obj.find('bndbox')
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        box_all += [bndbox]
        gt_cls.append(int(obj.find('name').text))

    return np.array(box_all), np.array(gt_cls)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    # paths
    chip_img = args.chip_dir + '/JPEGImages'
    source_anno = args.root_dir + '/Annotations'
    source_img = args.root_dir + '/JPEGImages'

    annotations = []
    img_list = []
    set_file = osp.join(args.root_dir, 'ImageSets/val.txt')
    with open(set_file, 'r') as f:
        for line in f.readlines():
            img_list.append(line.strip())

    if args.inference:
        if not osp.exists(args.result_path):
            os.mkdir(args.result_path)
        with open(args.chip_dir + '/Locations/val_chip.json', 'r') as f:
            chip_loc = json.load(f)

        # build the model from a config file and a checkpoint file
        model = init_detector(args.config, args.checkpoint, device='cuda:0')

        chip_list = []
        set_file = osp.join(args.chip_dir, 'ImageSets/Main/val.txt')
        with open(set_file, 'r') as f:
            for line in f.readlines():
                chip_list.append(line.strip())

        detecions = dict()
        iter = 1
        for img_id in tqdm(chip_list):
            # if iter > 5: break
            iter += 1
            img_name = img_id + '.jpg'
            newImg = '_'.join(img_id.split('_')[:-1]) + osp.splitext(img_name)[1]
            img_file = osp.join(chip_img, img_id+'.jpg')
            loc = chip_loc[img_name]

            # predict
            result = inference_detector(model, img_file)
            for i, boxes in enumerate(result):
                for bbox in boxes:
                    box = [bbox[0] + loc[0], bbox[1] + loc[1], bbox[2] + loc[0], bbox[3] + loc[1]]
                    if newImg in detecions:
                        detecions[newImg].append(box + [bbox[4], i])
                    else:
                        detecions[newImg] = [box + [bbox[4], i]]

            # show
            # model.show_result(img_file, result, out_file='chip_result.jpg')

        with open(os.path.join(args.result_path, 'tt100k_results.json'), "w") as f:
            json.dump(detecions, f, cls=MyEncoder)
            print("results json saved.")
    else:
        with open(os.path.join(args.result_path, 'tt100k_results.json'), "r") as f:
            detecions = json.load(f)
            print("load results json.")

    # merge
    results = []
    iter = 1
    for img_id in tqdm(img_list):
        iter += 1
        img_name = img_id + '.jpg'
        det_nms = []
        if img_name in detecions:
            det = np.array(detecions[img_name])
            det = det[det[:, -2] > args.score_thr]
            for i in range(args.nclass):
                det_nms.append(nms(det[det[:, -1] == i, :5], iou_thr=0.5)[0])
        else:
            det_nms = [np.array([]).reshape(0, 5) for i in range(args.nclass)]
        results.append(det_nms)

        # ground truth
        xml_file = osp.join(source_anno, img_id+'.xml')
        bboxes, labels = getGtFromXml(xml_file)
        annotations.append({"bboxes": bboxes, "labels": labels})

        # show
        img_file = osp.join(source_img, img_name)
        # model.show_result(img_file, det_nms, out_file='source_result.jpg')

    # voc metric
    eval_results = eval_map(results, annotations, iou_thr=0.5, logger="print")[0]
    pass
