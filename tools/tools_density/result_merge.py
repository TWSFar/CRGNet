import cv2
import os, sys
import glob
import json
import shutil
import argparse
import numpy as np
import utils
import pdb

from datasets import VisDrone

def parse_args():
    parser = argparse.ArgumentParser(description='VisDrone submit')
    parser.add_argument('--split', type=str, default='val', help='split')
    parser.add_argument('--result_file', type=str, default='')
    parser.add_argument('--loc_file', type=str, default='')
    parser.add_argument('--val_anno', type=str, default='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    with open(args.val_anno, 'r') as f:
        annos = json.load(f)
    with open(args.result_file, 'r') as f:
        results = json.load(f)
    with open(args.loc_file, 'r') as f:
        chip_loc = json.load(f)

    img_id2name = dict()
    for img in annos['images']:
        img_id2name[img['id']] = img['file_name']

    detecions = dict()
    for det in results:
        img_id = det['image_id']
        cls_id = det['category_id'] + 1
        bbox = det['bbox']
        score = det['score']
        if args.split == 'val':
            loc = chip_loc[img_id2name[img_id]]
            img_name = '_'.join(img_id2name[img_id].split('_')[:-1]) + '.jpg'
            bbox = [bbox[0] + loc[0], bbox[1] + loc[1], bbox[2] - 1, bbox[3] - 1]
        elif args.split == 'test':
            loc = chip_loc[img_id[:-4]]['loc']
            img_name = '_'.join(img_id.split('_')[:-1]) + '.jpg'
            bbox = [bbox[0] + loc[0], bbox[1] + loc[1], bbox[2], bbox[3]]
        
        if img_name in detecions:
            detecions[img_name].append(bbox + [score, cls_id])
        else:
            detecions[img_name] = [bbox + [score, cls_id]]

    output_dir = 'DET_results-%s' % args.split
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    for img_name, det in detecions.items():
        det = utils.nms(det)
        txt_name = img_name[:-4] + '.txt'
        with open(os.path.join(output_dir, txt_name), 'w') as f:
            for bbox in det:
                bbox = [str(x) for x in (list(bbox[0:5]) + [int(bbox[5])] + [-1, -1])]
                f.write(','.join(bbox) + '\n')