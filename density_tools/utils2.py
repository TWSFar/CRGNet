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
        chip_area = max(np.where(mask_chip > 0, 1, 0).sum(), 1)
        obj_num = max(mask_chip.sum(), 1.0)
        # weight = np.exp(0.5 * chip_area/chip_nobj)
        # weight = np.log(1 + chip_area ** 1.5 / (obj_num * 35)) + 1
        if box_w < min(mask_size) * 0.4 and box_h < min(mask_size) * 0.4:
            weight = 1 + 1 / (1 + np.exp(obj_num / chip_area))
        else:
            weight = 1

        crop_size_w = 0.5 * box_w * weight
        crop_size_h = 0.5 * box_h * weight
        crop_size_w = np.clip(crop_size_w, max(box_w/2.0, 6), max(box_w/2.0, 24))
        crop_size_h = np.clip(crop_size_h, max(box_h/2.0, 6), max(box_h/2.0, 24))

        center_x = crop_size_w if center_x < crop_size_w else center_x
        center_y = crop_size_h if center_y < crop_size_h else center_y
        center_x = width - crop_size_w if center_x > width - crop_size_w else center_x
        center_y = height - crop_size_h if center_y > height - crop_size_h else center_y

        new_box = [center_x - crop_size_w if center_x - crop_size_w > 0 else 0,
                   center_y - crop_size_h if center_y - crop_size_h > 0 else 0,
                   center_x + crop_size_w if center_x + crop_size_w < width else width,
                   center_y + crop_size_h if center_y + crop_size_h < height else height]
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
                if overlap(regions[i], regions[j], 0.8):
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
    '''
    待改进: 目标平均面积/区域面积 > 1/2 不切割, 防止破坏大目标
    '''
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


def show_image(img, labels=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img)
    if labels is not None:
        plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-')
    plt.show()
    pass
