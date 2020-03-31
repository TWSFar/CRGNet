import cv2
import json
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils import nms, nms2, plot_img, MyEncoder
hyp = {
    'gt': "/home/twsf/data/Visdrone/VisDrone2019-DET-val/annotations_json/instances_val.json",
    'result': "/home/twsf/work/CRGNet/chip_results.json",
    'local': "/home/twsf/data/Visdrone/region_chip/Locations/val_chip.json",
    'show': False,
    'srcimg_dir': "/home/twsf/data/Visdrone/VisDrone2019-DET-val/images/"
}


class DET_toolkit(object):
    def __init__(self):
        self.coco = COCO(hyp['gt'])
        self.srcimg_dir = hyp['srcimg_dir']
        self.img_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.loadCats(self.coco.getCatIds())

    def __call__(self):
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
        for img_id in tqdm(self.img_ids):
            img_info = self.coco.loadImgs(img_id)[0]
            det = detecions[img_info['file_name']]
            det = nms(det, score_threshold=0.05)[:, [0, 1, 2, 3, 5, 4]].astype(np.float32)
            det = nms2(det)
            # gt_bboxes = self.load_annotations(img_name)
            if hyp['show']:
                img = cv2.imread(osp.join(self.srcimg_dir, img_info['file_name']))[:, :, ::-1]
                gt = self.load_annotations(img_id)
                gt_img = plot_img(img, gt, self.cat_ids)
                pred_img = plot_img(img, det, self.cat_ids)
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 1, 1).imshow(gt_img)
                plt.show()
                plt.subplot(1, 1, 1).imshow(pred_img)
                plt.show()

            for bbox in det:
                bbox[2:4] = bbox[2:4] - bbox[:2]
                results.append({"image_id": img_id,
                                "category_id": bbox[4],
                                "bbox": np.round(bbox[:4]),
                                "score": bbox[5]})

        coco_pred = self.coco.loadRes(results)
        coco_eval = COCOeval(self.coco, coco_pred, 'bbox')
        coco_eval.params.imgIds = self.coco.getImgIds()
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4, cls=MyEncoder)

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.img_ids[image_index], iscrowd=False)
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
            annotation[0, 4] = a['category_id']
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


if __name__ == '__main__':
    det = DET_toolkit()
    det()
