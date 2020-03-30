import os
import cv2
import json
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils import nms, plot_img
hyp = {
    'gt': "/home/twsf/data/Visdrone/val/labels/",
    'result': "results.json",
    'local': "/home/twsf/data/Visdrone/region_chip/Locations/val_chip.json",
    'show': False,
    'srcimg_dir': "/home/twsf/data/Visdrone/VisDrone2019-DET-val/images/"
}


class DET_toolkit(object):
    def __init__(self):
        self.coco_true = COCO(hyp['gt'])
        self.img_dir = hyp['srcimg_dir']

    def __call__(self):
        coco_true = COCO(hyp['gt'])
        # get val predict box
        with open(hyp['result'], 'r') as f:
            results = json.load(f)
        with open(hyp['local'], 'r') as f:
            chip_loc = json.load(f)
        detecions = dict()
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

        # metrics
        results = []
        for img_name, det in tqdm(detecions.items()):
            pred_bboxes = nms(det, score_threshold=0.05)[:, [0, 1, 2, 3, 5, 4]].astype(np.float32)
            gt_bboxes = load_annotations(img_name)
            for bbox in pred_bboxes:
                results.append({"image_id": img_name,
                                "category_id": bbox[4],
                                "bbox": np.round(bbox[:4]),
                                "score": bbox[5]})
            if hyp['show']:
                img = cv2.imread(osp.join(self.img_dir, img_name))[:, :, ::-1]
                gt_bbox[:, 4] = 0
                pred_bbox[:, 4] = 1
                gt_img = plot_img(img, gt_bboxes)
                pred_img = plot_img(img, pred_bboxes)
                plt.figure(figsize=(10, 10))
                plt.subplot(2, 1, 1).imshow(gt_img)
                plt.subplot(2, 1, 2).imshow(pred_img)
                plt.show()

        coco_pred = coco_true.loadRes(hyp['result'])
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = coco_true.getImgIds()
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


if __name__ == '__main__':
    det = DET_toolkit
    det()