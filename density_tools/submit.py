import os
import cv2
import json
import utils
import zipfile
import numpy as np
import os.path as osp
from tqdm import tqdm

hyp = {
    'result': "/home/twsf/work/CRGNet/chip_results_{}.json",
    'local': "/home/twsf/data/Visdrone/challenge/density_loc/test_chip.json",
    'submit_dir': '/home/twsf/work/CRGNet/results',
    'show': False,
    'srcimg_dir': "/home/twsf/data/Visdrone/challenge/images/"
}


class Submit(object):
    def __init__(self):
        self.srcimg_dir = hyp['srcimg_dir']
        self.img_list = os.listdir(hyp['srcimg_dir'])
        if not osp.exists(hyp['submit_dir']):
            os.mkdir(hyp['submit_dir'])

    def __call__(self):
        with open(hyp['local'], 'r') as f:
            chip_loc = json.load(f)
        detecions = dict()
        # get val predict box
        scales = [800, 1300, 1500]
        for scale in scales:
            with open(hyp['result'].format(scale), 'r') as f:
                results = json.load(f)
            for det in tqdm(results):
                img_id = det['image_id']
                cls_id = det['category_id']
                bbox = det['bbox']
                score = det['score']
                loc = chip_loc[img_id]
                bbox = [bbox[0] + loc[0], bbox[1] + loc[1], bbox[2] + loc[0], bbox[3] + loc[1]]
                img_name = '_'.join(img_id.split('_')[:-1]) + osp.splitext(img_id)[1]
                if img_name in detecions:
                    detecions[img_name].append(bbox + [score, cls_id])
                else:
                    detecions[img_name] = [bbox + [score, cls_id]]

        # merge
        results = []
        for img_name in tqdm(self.img_list):
            det = []
            if img_name in detecions:
                det = detecions[img_name]
                det = utils.nms(det, score_threshold=0.05, iou_threshold=0.6, overlap_threshold=1).astype(np.float32)
                # det = utils.soft_nms(det).astype(np.float32)

            # show
            if hyp['show']:
                img = cv2.imread(osp.join(self.srcimg_dir, img_name))
                utils.show_image(img, det[det[:, 4] > 0.3])

            # save
            with open(osp.join(hyp['submit_dir'], img_name[:-4]+'.txt'), "w") as f:
                for box in det:
                    box[2:4] -= box[:2]
                    line = []
                    for idx, v in enumerate(list(box[0:5]) + [box[5]+1] + [-1, -1]):
                        line.append(str(int(v)) if idx != 4 else str(v))
                    f.write(','.join(line) + '\n')

        # Zip
        result_files = [osp.join(hyp['submit_dir'], file) for file in os.listdir(hyp['submit_dir'],)]
        zip_path = 'result.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip:
            for file in result_files:
                if ".txt" in file:
                    zip.write(file)


if __name__ == '__main__':
    det = Submit()
    det()
