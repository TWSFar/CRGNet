import os
import os.path as osp
import cv2
import sys
import random
import numpy as np
from pycocotools.coco import COCO

import torch


class CocoDataset(object):
    """Coco dataset."""
    CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14')
    def __init__(self):
        self.coco = COCO("/home/twsf/data/DOTA/Annotations_json/instances_train_all.json")
        self.image_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join("/home/twsf/data/DOTA/JPEGImages/", image_info['file_name'])
        # read img and BGR to RGB before normalize
        img = cv2.imread(path)[:, :, ::-1] / 255.0
        return img.astype(np.float32)

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
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations





def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-')
    plt.savefig('test.jpg')
    plt.show()
    pass


dataset = CocoDataset()

for i in range(10):
    sample = dataset.__getitem__(i)
    show_image(sample['img'], sample['annot'])
    