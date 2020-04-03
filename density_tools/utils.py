import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pdb


def bbox_merge(bbox1, bbox2):
    """ (box1 cup box2) / box2
    Args:
        box1: [xmin, ymin, xmax, ymax]
        box2: [xmin, ymin, xmax, ymax]
    Return:
        overlap box1 and box2
    """
    left_up = np.minimum(bbox1[:2], bbox2[:2])
    right_down = np.maximum(bbox1[2:], bbox2[2:])

    return np.hstack((left_up, right_down))


def enlarge_box(mask_box, image_size, ratio=2):
    """
    Args:
        mask_box: list of box
        image_size: (width, height)
        ratio: int
    """
    new_mask_box = []
    for box in mask_box:
        w = box[2] - box[0]
        h = box[3] - box[1]
        center_x = w / 2 + box[0]
        center_y = h / 2 + box[1]
        w = w * ratio / 2
        h = h * ratio / 2
        new_box = [center_x-w if center_x-w > 0 else 0,
                   center_y-h if center_y-h > 0 else 0,
                   center_x+w if center_x+w < image_size[0] else image_size[0]-1,
                   center_y+h if center_y+h < image_size[1] else image_size[1]-1]
        new_box = [int(x) for x in new_box]
        new_mask_box.append(new_box)
    return new_mask_box


def generate_box_from_mask(mask):
    """
    Args:
        mask: 0/1 array
    """
    regions = []
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        regions.append([x, y, x+w, y+h])
        # temp = np.array(regions[-1])
    # show_image(mask, temp[None])
    # v = mask[temp[1]:temp[3], temp[0]:temp[2]].sum()
    return regions, contours


def generate_crop_region(regions, mask, mask_size):
    """
    generate final regions
    enlarge regions < 300
    """
    width, height = mask_size
    final_regions = []
    for box in regions:
        # show_image(mask, np.array(box)[None])
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        center_x, center_y = box[0] + box_w / 2.0, box[1] + box_h / 2.0

        mask_chip = mask[box[1]:box[3], box[0]:box[2]]
        obj_area = max(np.where(mask_chip > 0, 1, 0).sum(), 1)
        obj_num = max(mask_chip.sum(), 1.0)
        chip_area = box_w * box_h
        # weight = np.exp(0.5 * chip_area/chip_nobj)
        # weight = np.log(1 + chip_area ** 1.5 / (obj_num * 35)) + 1
        if box_w < min(mask_size) * 0.4 and box_h < min(mask_size) * 0.4:
            # weight = 16.0*obj_area/(obj_num*chip_area)
            weight = np.clip(16.0*obj_area/(obj_num*chip_area), 1, 4)
        else:
            weight = 1

        rect = np.sqrt(chip_area * weight)
        if max(box_w, box_h) <= rect:
            half_w = 0.5 * rect
            half_h = 0.5 * rect
        elif box_w > rect:
            half_w = 0.5 * box_w
            half_h = 0.5 * chip_area * weight / half_w
        else:
            half_h = 0.5 * box_h
            half_w = 0.5 * chip_area * weight / half_h
        half_w = np.clip(half_w, max(box_w/2.0, 3), max(box_w/2.0, 24))
        half_h = np.clip(half_h, max(box_h/2.0, 3), max(box_h/2.0, 24))

        center_x = half_w if center_x < half_w else center_x
        center_y = half_h if center_y < half_h else center_y
        center_x = width - half_w if center_x > width - half_w else center_x
        center_y = height - half_h if center_y > height - half_h else center_y

        new_box = [center_x - half_w if center_x - half_w > 0 else 0,
                   center_y - half_h if center_y - half_h > 0 else 0,
                   center_x + half_w if center_x + half_w < width else width,
                   center_y + half_h if center_y + half_h < height else height]
        for x in new_box:
            if x < 0:
                pdb.set_trace()
        final_regions.append(new_box)
        # show_image(mask, np.array(final_regions)[None, -1])

    regions = np.array(final_regions)
    while(1):
        idx = np.zeros((len(regions)))
        for i in range(len(regions)):
            for j in range(len(regions)):
                if i == j or idx[i] == 1 or idx[j] == 1:
                    continue
                if overlap(regions[i], regions[j]):
                    regions[i] = bbox_merge(regions[i], regions[j])
                    idx[j] = 1
        if sum(idx) == 0:
            break
        regions = regions[idx == 0]

    return regions


def resize_box(box, original_size, dest_size):
    """
    Args:
        box: array, [xmin, ymin, xmax, ymax]
        original_size: (width, height)
        dest_size: (width, height)
    """
    h_ratio = 1.0 * dest_size[1] / original_size[1]
    w_ratio = 1.0 * dest_size[0] / original_size[0]
    box = np.array(box)
    if len(box) > 0:
        box = box * np.array([w_ratio, h_ratio, w_ratio, h_ratio])
    return list(box.astype(np.int32))


def region_cluster(regions, mask_shape):
    """
    层次聚类
    """
    regions = np.array(regions)
    centers = (regions[:, [2, 3]] + regions[:, [0, 1]]) / 2.0

    model = AgglomerativeClustering(
                n_clusters=None,
                linkage='complete',
                distance_threshold=min(mask_shape) * 0.4,
                compute_full_tree=True)

    labels = model.fit_predict(centers)

    cluster_regions = []
    for idx in np.unique(labels):
        boxes = regions[labels == idx]
        new_box = [min(boxes[:, 0]), min(boxes[:, 1]),
                   max(boxes[:, 2]), max(boxes[:, 3])]
        cluster_regions.append(new_box)

    return cluster_regions


def region_morphology(contours, mask_shape):
    mask_w, mask_h = mask_shape
    binary = np.zeros((mask_h, mask_w)).astype(np.uint8)
    cv2.drawContours(binary, contours, -1, 1, cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # 开操作
    region_open, _ = generate_box_from_mask(binary_open)

    binary_rest = binary ^ binary_open
    region_rest, _ = generate_box_from_mask(binary_rest)

    regions = []
    alpha = 1
    for box in region_open:
        box = [max(0, box[0] - alpha), max(0, box[1] - alpha),
               min(mask_w, box[2] + alpha), min(mask_h, box[3] + alpha)]
        regions.append(box)
    return regions + region_rest


def region_postprocess(regions, contours, mask_shape):
    mask_w, mask_h = mask_shape

    # 1. get big contours
    small_regions = []
    big_contours = []
    for i, box in enumerate(regions):
        w, h = box[2] - box[0], box[3] - box[1]
        if w > mask_w / 2 or h > mask_h / 2:
            big_contours.append(contours[i])
        else:
            small_regions.append(box)

    # 2. image open
    regions = region_morphology(big_contours, mask_shape) + small_regions

    # 3. delete inner box
    regions = np.array(regions)
    idx = np.zeros((len(regions)))
    for i in range(len(regions)):
        for j in range(len(regions)):
            if i == j or idx[i] == 1 or idx[j] == 1:
                continue
            if overlap(regions[i], regions[j], 0.9):
                regions[i] = bbox_merge(regions[i], regions[j])
                idx[j] = 1
    regions = regions[idx == 0]

    # 4. process small regions and big regions
    small_regions = []
    big_regions = []
    for box in regions:
        w, h = box[2] - box[0], box[3] - box[1]
        if max(w, h) > min(mask_w, mask_h) * 0.4:
            big_regions.append(box)
        else:
            small_regions.append(box)
    if len(big_regions) > 0:
        big_regions = region_split(big_regions, mask_shape)
    if len(small_regions) > 1:
        small_regions = region_cluster(small_regions, mask_shape)

    regions = np.array(small_regions + big_regions)

    return regions


def region_split(regions, mask_shape):
    alpha = 1
    mask_w, mask_h = mask_shape
    new_regions = []
    for box in regions:
        width, height = box[2] - box[0], box[3] - box[1]
        if width / height > 1.5:
            mid = int(box[0] + width / 2.0)
            new_regions.append([box[0], box[1], mid + alpha, box[3]])
            new_regions.append([mid - alpha, box[1], box[2], box[3]])
        elif height / width > 1.5:
            mid = int(box[1] + height / 2.0)
            new_regions.append([box[0], box[1], box[2], mid + alpha])
            new_regions.append([box[0], mid - alpha, box[2], box[3]])
        elif width > mask_w * 0.6 and height > mask_h * 0.7:
            mid_w = int(box[0] + width / 2.0)
            mid_h = int(box[1] + height / 2.0)
            new_regions.append([box[0], box[1], mid_w + alpha, mid_h + alpha])
            new_regions.append([mid_w - alpha, box[1], box[2], mid_h + alpha])
            new_regions.append([box[0], mid_h - alpha, mid_w - alpha, box[3]])
            new_regions.append([mid_w - alpha, mid_h - alpha, box[2], box[3]])
        else:
            new_regions.append(box)
    return new_regions


def overlap(box1, box2, thresh=0.75):
    """ (box1 cup box2) / box2
    Args:
        box1: [xmin, ymin, xmax, ymax]
        box2: [xmin, ymin, xmax, ymax]
    """
    box1_area = np.product(box1[2:] - box1[:2])
    box2_area = np.product(box2[2:] - box2[:2])
    if box1_area < box2_area:
        box1, box2 = box2, box1
    matric = np.array([box1, box2])
    u_xmin = max(matric[:,0])
    u_ymin = max(matric[:,1])
    u_xmax = min(matric[:,2])
    u_ymax = min(matric[:,3])
    u_w = u_xmax - u_xmin
    u_h = u_ymax - u_ymin
    if u_w <= 0 or u_h <= 0:
        return False
    u_area = u_w * u_h
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if u_area / box2_area < thresh:
        return False
    else:
        return True


def iou_calc1(boxes1, boxes2):
    """
    array
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + 1e-16 + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def iou_calc2(boxes1, boxes2):
    """
    array
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    IOU = 1.0 * inter_area / boxes2_area
    return IOU


def nms(prediction, score_threshold=0.05, iou_threshold=0.5, overlap_threshold=0.95):
    """
    :param prediction:
    (x, y, w, h, conf, cls)
    :return: best_bboxes
    """
    prediction = np.array(prediction)
    detections = prediction[(-prediction[:,4]).argsort()]
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
            # iou
            iou = iou_calc1(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            iou_mask = iou > iou_threshold
            # overlap
            overlap = iou_calc2(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            overlap_mask = overlap > overlap_threshold

            weight = np.ones((len(iou),), dtype=np.float32)
            weight[iou_mask] = 0.0
            weight[overlap_mask] = 0.0

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    best_bboxes = np.array(best_bboxes)
    return best_bboxes


def show_image(img, labels=None):
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10))
    plt.imshow(img)
    if labels is not None:
        if labels.shape[0] > 0:
            plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-')
    plt.show()
    pass


def plot_img(img, bboxes, id2name):
    box_colors = ((0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1),
                  (0.541, 0.149, 0.341), (0.541, 0.169, 0.886),
                  (0.753, 0.753, 0.753), (0.502, 0.165, 0.165),
                  (0.031, 0.180, 0.329), (0.439, 0.502, 0.412),
                  (0, 0, 0)) # others
    img = img.astype(np.float64) / 255.0 if img.max() > 1.0 else img
    for bbox in bboxes:
        try:
            if -1 in bbox:
                continue
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            id = int(bbox[4])
            label = id2name[id]

            if len(bbox) >= 6:
                # if bbox[5] < 0.5:
                #     continue
                label = label + '|{:.2}'.format(bbox[5])

            # plot
            box_color = box_colors[min(id, len(box_colors)-1)]
            text_color = (1, 1, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)[0]
            c1 = (x1, y1 - t_size[1] - 4)
            c2 = (x1 + t_size[0], y1)
            cv2.rectangle(img, c1, c2, color=box_color, thickness=-1)
            cv2.putText(img, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX, 0.4, text_color, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=box_color, thickness=3)

        except Exception as e:
            print(e)
            continue

    return img
