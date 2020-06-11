import os
import json
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

hyp = {
    'dataset': 'VisDrone2019',
    'img_type': '.jpg',
    'mode': 'test',  # for save instance_train.json
    'data_dir': '/home/twsf/data/Visdrone/test',
}
hyp['json_dir'] = osp.join(hyp['data_dir'], 'annotations_json')
hyp['txt_dir'] = osp.join(hyp['data_dir'], 'annotations')
hyp['img_dir'] = osp.join(hyp['data_dir'], 'images')


class getItem(object):
    def __init__(self):
        # self.classes = ('pedestrian', 'person', 'bicycle', 'car', 'van',
        #                 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def get_img_item(self, file_name, image_id, size):
        """Gets a image item."""
        image = OrderedDict()
        image['file_name'] = file_name
        image['height'] = int(size['height'])
        image['width'] = int(size['width'])
        image['id'] = image_id
        return image

    def get_ann_item(self, bbox, img_id, cat_id, anno_id):
        """Gets an annotation item."""
        x1 = bbox[0]
        y1 = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        annotation = OrderedDict()
        annotation['segmentation'] = [[x1, y1, x1, (y1 + h), (x1 + w), (y1 + h), (x1 + w), y1]]
        annotation['area'] = w * h
        annotation['iscrowd'] = 0
        annotation['image_id'] = img_id
        annotation['bbox'] = [x1, y1, w, h]
        annotation['category_id'] = cat_id
        annotation['id'] = anno_id
        return annotation

    def get_cat_item(self):
        """Gets an category item."""
        categories = []
        for idx, cat in enumerate(self.classes):
            cate = {}
            cate['supercategory'] = cat
            cate['name'] = cat
            cate['id'] = idx
            categories.append(cate)

        return categories


def getGTBox(anno_path, **kwargs):
    box_all = []
    gt_cls = []
    with open(anno_path, 'r') as f:
        data = [x.strip().split(',')[:8] for x in f.readlines()]
        annos = np.array(data)

    bboxes = annos[annos[:, 4] == '1'][:, :6].astype(np.float64)
    for bbox in bboxes:
        if bbox[2] <= 0 or bbox[3] <= 0:
            print(osp.basename(anno_path) + ' exist an illegal side:')
            print(bbox)
            print('illegal side has been abandoned')
            continue
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        box_all.append(bbox[:4].tolist())
        gt_cls.append(int(bbox[5]) - 1)  # index begin idx 0

    return box_all, gt_cls


def make_json():
    item = getItem()
    images = []
    annotations = []
    anno_id = 0

    # categories
    categories = item.get_cat_item()

    txt_files = os.listdir(hyp['txt_dir'])
    for id, file_name in enumerate(tqdm(txt_files)):
        img_id = id

        # anno info
        anno_txt = os.path.join(hyp['txt_dir'], file_name)
        box_all, gt_cls = getGTBox(anno_txt)
        for ii in range(len(box_all)):
            annotations.append(
                item.get_ann_item(box_all[ii], img_id, gt_cls[ii], anno_id))
            anno_id += 1

        # image info
        img_name = file_name[:-4] + hyp['img_type']  # image name
        img_path = osp.join(hyp['img_dir'], img_name)  # image path
        tsize = plt.imread(img_path).shape[:2]  # (h, w)
        size = {'height': tsize[0], 'width': tsize[1]}
        image = item.get_img_item(img_name, img_id, size)
        images.append(image)

    # all info
    ann = OrderedDict()
    ann['images'] = images
    ann['categories'] = categories
    ann['annotations'] = annotations

    # saver
    if not osp.exists(hyp['json_dir']):
        os.makedirs(hyp['json_dir'])
    save_file = os.path.join(hyp['json_dir'], 'instances_{}.json'.format(hyp['mode']))
    print('Saving annotations to {}'.format(save_file))
    json.dump(ann, open(save_file, 'w'), indent=4)


if __name__ == '__main__':
    make_json()
