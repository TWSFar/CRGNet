import os
import mmcv
import json
import argparse
import numpy as np
import os.path as osp
from mmdet.apis import init_detector, inference_detector, show_result
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Test chip')
    parser.add_argument('--checkpoint', default='/home/twsf/work/CRGNet/mmdetection/tools_visdrone/work_dirs/retinanet_r50_fpn_1x/epoch_29.pth', help='model')
    parser.add_argument('--config', default='/home/twsf/work/CRGNet/mmdetection/tools_visdrone/retinanet_r50_fpn_1x.py')
    parser.add_argument('--test-dir', default='/home/twsf/data/Visdrone/region_chip')
    parser.add_argument('--result-path', default='./result')
    args = parser.parse_args()
    return args


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
    if not osp.exists(args.result_path):
        os.makedirs(args.result_path)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    img_list = []
    # set_file = osp.join(args.test_dir, 'ImageSets/Main/val.txt')
    # with open(set_file, 'r') as f:
    #     for line in f.readlines():
    #         img_list.append(line.strip())

    img_list = os.listdir(args.test_dir)

    results = []
    for img_id in tqdm(img_list):
        img_path = osp.join(args.test_dir, 'JPEGImages', img_id+'.jpg')
        result = inference_detector(model, img_path)
        with open(os.path.join(args.result_path, img_id+'.txt'), "w") as f:
            for i, boxes in enumerate(result):
                for box in boxes:
                    bbox = [str(x) for x in (list(box[0:5]) + [i+1] + [-1, -1])]
                    f.write(','.join(bbox) + '\n')
