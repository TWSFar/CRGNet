import json
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
gt_file = '/home/twsf/data/Visdrone/VisDrone2019-DET-val/annotations_json/instances_val.json'
pred_file = "/home/twsf/work/CRGNet/results.json"


if __name__ == '__main__':
    coco_true = COCO(gt_file)
    tep = coco_true.loadCats(coco_true.getCatIds())
    coco_pred = coco_true.loadRes(pred_file)
    # coco_pred = json.load(open(pred_file))
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = coco_true.getImgIds()
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
