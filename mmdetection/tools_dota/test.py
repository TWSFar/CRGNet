import os
import argparse
import numpy as np
import os.path as osp
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Test chip')
    parser.add_argument('--checkpoint', default="/home/twsf/work/CRGNet/mmdetection/tools_dota/work_dirs/retinanet_x101/epoch_52.pth", help='model')
    parser.add_argument('--config', default='/home/twsf/work/CRGNet/mmdetection/tools_dota/configs/source/retinanet_x101.py')
    parser.add_argument('--test-dir', default='/home/twsf/data/DOTA/test')
    parser.add_argument('--result-path', default='./results')
    args = parser.parse_args()
    return args


classes = ('plane', 'ship', 'storage-tank', 'baseball-diamond',
            'tennis-court', 'basketball-court', 'ground-track-field',
            'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',
            'roundabout', 'soccer-ball-field', 'swimming-pool')


if __name__ == "__main__":
    args = parse_args()
    if not osp.exists(args.result_path):
        os.makedirs(args.result_path)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    img_list = []
    img_list = os.listdir(args.test_dir)
    # set_file = osp.join(args.test_dir, 'ImageSets/val_all.txt')
    # with open(set_file, 'r') as f:
    #     for line in f.readlines():
    #         img_list.append(line.strip())

    # img_list = os.listdir(args.test_dir)

    results = dict()
    for i in range(len(classes)):
        results[str(i)] = []

    # item = 1
    for img_name in tqdm(img_list):
        # if item > 10: break
        # item += 1
        img_path = osp.join(args.test_dir, img_name)
        result = inference_detector(model, img_path)
        for i, boxes in enumerate(result):
            for box in boxes:
                temp = img_name[:-4] + ' ' + str(box[4])
                for v in box[:4]:
                    temp += ' ' + str(v)
                results[str(i)].append(temp)

        model.show_result(img_path, result, out_file='result.jpg')

    for i, cls in enumerate(classes):
        with open(osp.join(args.result_path, 'Task2_'+cls+'.txt'), 'w') as f:
            for line in results[str(i)]:
                f.writelines(line+'\n')
