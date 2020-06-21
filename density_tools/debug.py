import numpy as np


def py_cpu_softnms(prediction, iou_threshold=0.3, sigma=0.5, score_threshold=0.001, method=2, top=1000):
    """
    :param prediction:
    (x, y, w, h, conf, cls)
    :return: best_bboxes
    """
    prediction = np.array(prediction)
    detections = prediction[(-prediction[:,4]).argsort()]
    detections = detections[:topN]
    # Iterate through all predicted classes
    unique_labels = np.unique(detections[:, -1])
    best_bboxes = []
    for cls in unique_labels:
        cls_mask = (detections[:, 5] == cls)
        cls_bboxes = detections[cls_mask]

        # python code
        while len(cls_bboxes) > 0:
            best_bbox = cls_bboxes[0]
            best_bboxes.append(best_bbox)
            cls_bboxes = cls_bboxes[1:]

            iou = iou_calc1(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            mask = iou > iou_threshold

            # Three methods: 1.linear 2.gaussian 3.original NMS
            if method == 1:  # linear
                cls_bboxes[mask, 4] = (cls_bboxes[mask, 4] - iou[mask]) * cls_bboxes[mask, 4]
            elif method == 2:  # gaussian
                cls_bboxes[mask, 4] = np.exp(-(iou[mask] * iou[mask]) / sigma) * cls_bboxes[mask, 4]
            else:  # original NMS
                cls_bboxes[mask, 4] = 0

            score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]

    # select the boxes and keep the corresponding indexes
    best_bboxes = np.array(best_bboxes)
    return best_bboxes