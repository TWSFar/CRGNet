import cv2
import json
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils import nms, plot_img, MyEncoder, iou_calc1
hyp = {
    'gt': "/home/twsf/data/Visdrone/VisDrone2019-DET-val/annotations_json/instances_val.json",
    'result': "/home/twsf/work/CRGNet/chip_results.json",
    'local': "/home/twsf/data/Visdrone/density_chip/Locations/val_chip.json",
    'show': False,
    'srcimg_dir': "/home/twsf/data/Visdrone/VisDrone2019-DET-val/images/",
    'gt_txt': "/home/twsf/data/Visdrone/VisDrone2019-DET-val/annotations"
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
            gt = self.load_annotations(img_info['file_name'])
            gt, det = self.dropObjectsInIgr(gt, det)
            det = nms(det, score_threshold=0.05, iou_threshold=0.5, overlap_threshold=1)[:, [0, 1, 2, 3, 5, 4]].astype(np.float32)
            # det = nms2(det)
            if hyp['show']:
                img = cv2.imread(osp.join(self.srcimg_dir, img_info['file_name']))[:, :, ::-1]
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

    def load_annotations(self, img_id):
        anno_file = osp.join(hyp['gt_txt'], img_id[:-4]+'.txt')
        with open(anno_file, 'r') as f:
            bboxes = np.array([x.strip().split(',')[:6] for x in f.readlines()], dtype=np.float64)
        bboxes[:, 2:4] += bboxes[:, :2]

        return bboxes

    def dropObjectsInIgr(self, gt, det):
        det = np.array(det)
        idxFr = gt[:, 4] != 0
        idxIgr = gt[:, 4] == 0
        igrRegion = gt[idxIgr, 0:4]

        igrDet = np.ones(len(det), dtype=np.bool)
        for i in range(len(igrRegion)):
            iou = iou_calc1(igrRegion[i], det[:, :4])
            igrDet[iou > 0.5] = False

        return gt[idxFr][:, [0, 1, 2, 3, 5]], det[igrDet]


if __name__ == '__main__':
    det = DET_toolkit()
    det()