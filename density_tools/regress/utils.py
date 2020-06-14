import csv
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
import cv2


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


def delete_inner_region(regions, mask_shape, thresh=0.99):
    """
    Args:
        regions: xmin, ymin, xmax, ymax
        mask_shape: width, height
    """
    regions = np.array(regions)
    regions_temp = np.round(regions.copy()).astype(np.int)
    mask_w, mask_h = mask_shape
    areas = np.product(regions_temp[:, 2:] - regions_temp[:, :2], axis=1)
    sort_idx = (-areas).argsort()
    regions_temp = regions_temp[sort_idx]
    regions = regions[sort_idx]
    areas = areas[sort_idx]

    mask = np.zeros((mask_h, mask_w), dtype=np.int)
    del_idx = np.ones(len(regions_temp), dtype=np.bool)
    for i, region in enumerate(regions_temp):
        if mask[region[1]:region[3], region[0]:region[2]].sum() >= thresh*areas[i]:
            del_idx[i] = False
        else:
            mask[region[1]:region[3], region[0]:region[2]] = 1

    return regions[del_idx]


def generate_box_from_mask(mask):
    """
    Args:
        mask: 0/1 array
    """
    # temp = mask.copy()
    regions = []
    mask = (mask > 0).astype(np.uint8)
    mask = region_morphology(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        regions.append([x, y, x+w, y+h])
    return regions


def generate_crop_region(regions, mask, mask_shape, img_shape, gbm=None):
    """
    generate final regions
    enlarge regions < 300
    """
    mask_w, mask_h = mask_shape
    img_h, img_w = img_shape
    final_regions = []
    info = []
    # info = False
    split = 0
    enlarge = 0
    for box in regions:
        # get weight
        mask_chip = mask[box[1]:box[3], box[0]:box[2]]
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        obj_area = max(np.where(mask_chip > 0, 1, 0).sum(), 1)
        obj_num = max(mask_chip.sum(), 1.0)
        chip_area = box_w * box_h
        weight = gbm.predict([[obj_num, obj_area, chip_area, img_shape[0]*img_shape[1]]])[0]
        # info.append([obj_num, obj_area, chip_area])  # 1
        # final_regions.append(box)  # 2

        # resize
        det_w = box_w * img_w / mask_w
        det_h = box_h * img_h / mask_h
        det_area = det_w * det_h
        alpha = mask_w / mask_h
        weight = min(max(weight, 65536 / det_area), 9)  # enlarge minsize: 65536=256*256
        # weight = min(weight, 9)
        if weight <= 0.6 and (box_w > 0.3 * mask_w or box_h > 0.3 * alpha * mask_h):
            split += 1
            # show_image(mask, np.array([box]))
            final_regions.extend(region_split(box, mask_shape, weight))
        elif weight > 1 and (box_w < 0.5 * mask_w and box_h < 0.5 * alpha * mask_h):
            enlarge += 1
            final_regions.append(region_enlarge(box, mask_shape, weight))
        else:
            final_regions.append(box)

    final_regions = np.array(final_regions)
    while(1):
        idx = np.zeros((len(final_regions)))
        for i in range(len(final_regions)):
            for j in range(len(final_regions)):
                if i == j or idx[i] == 1 or idx[j] == 1:
                    continue
                if overlap(final_regions[i], final_regions[j], thresh=0.8):
                    final_regions[i] = bbox_merge(final_regions[i], final_regions[j])
                    idx[j] = 1
        if sum(idx) == 0:
            break
        final_regions = final_regions[idx == 0]

    if len(final_regions) > 0:
        final_regions = delete_inner_region(final_regions.copy(), mask_shape)

    return np.array(final_regions), (info, split, enlarge)


def region_morphology(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭操作

    return mask


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


def region_enlarge(region, mask_shape, weight):
    """
    Args:
        mask_box: list of box
        image_size: (width, hight)
        ratio: int
    """
    width, hight = mask_shape
    rgn_w, rgn_h = region[2] - region[0], region[3] - region[1]
    center_x, center_y = region[0] + rgn_w / 2.0, region[1] + rgn_h / 2.0
    chip_area = rgn_w * rgn_h
    rect = np.sqrt(chip_area * weight)
    if max(rgn_w, rgn_h) <= rect:
        half_w = 0.5 * rect
        half_h = 0.5 * rect
    elif rgn_w > rect:
        half_w = 0.5 * rgn_w
        half_h = 0.5 * chip_area * weight / rgn_w
    else:
        half_h = 0.5 * rgn_h
        half_w = 0.5 * chip_area * weight / rgn_h
    half_w = min(half_w, width/2.0)
    half_h = min(half_h, hight/2.0)

    center_x = half_w if center_x < half_w else center_x
    center_y = half_h if center_y < half_h else center_y
    center_x = width - half_w if center_x > width - half_w else center_x
    center_y = hight - half_h if center_y > hight - half_h else center_y

    new_box = [center_x - half_w if center_x - half_w > 0 else 0,
               center_y - half_h if center_y - half_h > 0 else 0,
               center_x + half_w if center_x + half_w < width else width,
               center_y + half_h if center_y + half_h < hight else hight]

    return new_box


def region_split(region, mask_shape, weight):
    alpha = 1
    mask_w, mask_h = mask_shape
    new_region = []
    width, height = region[2] - region[0], region[3] - region[1]
    #  and max(width, height) / min(width, height) < 1.5
    if weight <= 0.3 and max(width, height) / min(width, height) < 1.5:
        mid_w = int(region[0] + width / 2.0)
        mid_h = int(region[1] + height / 2.0)
        new_region.append([region[0], region[1], mid_w + alpha, mid_h + alpha])
        new_region.append([mid_w - alpha, region[1], region[2], mid_h + alpha])
        new_region.append([region[0], mid_h - alpha, mid_w + alpha, region[3]])
        new_region.append([mid_w - alpha, mid_h - alpha, region[2], region[3]])
    elif width / height >= 1.5:
        mid = int(region[0] + width / 2.0)
        new_region.append([region[0], region[1], mid + alpha, region[3]])
        new_region.append([mid - alpha, region[1], region[2], region[3]])
    elif height / width >= 1.5:
        mid = int(region[1] + height / 2.0)
        new_region.append([region[0], region[1], region[2], mid + alpha])
        new_region.append([region[0], mid - alpha, region[2], region[3]])
    else:
        new_region.append(region)
    return new_region


def overlap(box1, box2, thresh=0.9):
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


def show_image(img, labels=None, img_name=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # plt.figure(figsize=(10, 10))
    fig = plt.figure(frameon=False)
    if img_name is not None:
        plt.title(img_name) 
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(img[..., ::-1], cmap=cm.jet)
    if labels is not None:
        if labels.shape[0] > 0:
            plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-', color='green', linewidth=1)
    plt.savefig("chip_utils.png")
    plt.show()
    ax.set_axis_off()
    pass


def get_log():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s')
    logger = logging.getLogger(__name__)
    # f_handler = logging.FileHandler('', mode='a')
    # logger.addHandler(f_handler)
    return logger


def get_dataset(file, aim, transform=True, scaler=True):
    # load dataset
    feature = []
    target = []
    aim_ratio = aim*aim/(640*480)
    csv_file = csv.reader(open(file))
    for content in csv_file:
        content = list(map(float, content))
        if len(content) != 0:
            feature.append(content[0:-1])
            target.append(content[-1]/aim_ratio)

    if transform:
        for i in range(len(feature)):
            feature[i][0] = 1 / feature[i][0]
            feature[i][3] = 1 / feature[i][3]

    # dataset transform
    if scaler:
        scaler = StandardScaler().fit(feature)
        feature = scaler.transform(feature)

    return feature, target
