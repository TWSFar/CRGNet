import os
import json
import argparse
import numpy as np
import os.path as osp
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Test chip')
    parser.add_argument('--checkpoint', default="/home/twsf/work/CRGNet/mmdetection/tools_visdrone/work_dirs/ATSS_res2net/epoch_29.pth", help='model')
    parser.add_argument('--config', default='/home/twsf/work/CRGNet/mmdetection/tools_visdrone/configs/density/ATSS_res2net_bs.py')
    parser.add_argument('--test-dir', default='/home/twsf/data/Visdrone/density_chip')
    parser.add_argument('--result-path', default='/home/twsf/work/CRGNet/')
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
    set_file = osp.join(args.test_dir, 'ImageSets/Main/val.txt')
    with open(set_file, 'r') as f:
        for line in f.readlines():
            img_list.append(line.strip())

    # img_list = os.listdir(args.test_dir)

    results = []
    for img_name in tqdm(img_list):
        img_path = osp.join(args.test_dir, 'JPEGImages', img_name+'.jpg')
        result = inference_detector(model, img_path)
        for i, boxes in enumerate(result):
            for box in boxes:
                results.append({"image_id": img_name+'.jpg',
                                "category_id": i,
                                "bbox": np.round(box[:4]),
                                "score": box[4]})
        # model.show_result(img_path, result, out_file='result.jpg')

    with open(os.path.join(args.result_path, 'chip_results.json'), "w") as f:
        json.dump(results, f, cls=MyEncoder)
        print("results json saved.")
