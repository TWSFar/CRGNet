import os
import cv2
import glob
import pickle
import numpy as np
import os.path as osp
from PIL import Image


class VisDrone(object):
    def __init__(self, db_root):
        self._init_path(db_root)

    def _init_path(self, db_root):
        self.src_traindir = db_root + '/VisDrone2019-DET-train'
        self.src_valdir = db_root + '/VisDrone2019-DET-val'
        self.src_testdir = db_root + '/VisDrone2019-DET-test-challenge'
        self.region_voc_dir = db_root + '/region_voc'
        self.density_voc_dir = db_root + '/density_voc'
        self.region_detc
        cache_path = osp.join(db_root, 'cache')
        if not osp.exists(cache_path):
            os.makedirs(cache_path)
        self.cache_dir = cache_path

    def _get_imglist(self, split='train'):
        """ return list of all image paths
        """
        if split == 'train':
            return glob.glob(self.src_traindir + '/images/*.jpg')
        elif split == 'val':
            return glob.glob(self.src_valdir + '/images/*.jpg')
        elif split == 'test':
            return glob.glob(self.src_testdir + '/images/*.jpg')
        else:
            raise('error')

    def _get_annolist(self, split):
        """ annotation type is '.txt'
        return list of all image annotation path
        """
        img_list = self._get_imglist(split)
        return [img.replace('images', 'annotations').replace('jpg', 'txt') for img in img_list]

    def _get_gtbox(self, anno_path):
        box_all = []
        with open(anno_path, 'r') as f:
            data = [x.strip().split(',')[:8] for x in f.readlines()]
            annos = np.array(data)

        bboxes = annos[annos[:, 4] == '1'][:, :6].astype(np.float64)
        for bbox in bboxes:
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            box_all.append(bbox[:4].tolist())

        return {'bboxes': np.array(box_all, dtype=np.float64),
                'cls': bboxes[:, 5]}

    def _load_samples(self, split):
        cache_file = osp.join(self.cache_dir, split + '_samples.pkl')

        # load bbox and save to cache
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                samples = pickle.load(fid)
            print('gt samples loaded from {}'.format(cache_file))
            return samples

        img_list = self._get_imglist(split)
        anno_path = [img.replace('images', 'annotations').replace('jpg', 'txt')
                     for img in img_list]

        # load information of image and save to cache
        sizes = [Image.open(img).size for img in img_list]

        samples = [self._get_gtbox(ann) for ann in anno_path]

        for i, img in enumerate(img_list):
            samples[i]['image'] = img  # image path
            samples[i]['width'] = sizes[i][0]
            samples[i]['height'] = sizes[i][1]

        with open(cache_file, 'wb') as fid:
            pickle.dump(samples, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt samples to {}'.format(cache_file))

        return samples


if __name__ == "__main__":
    dataset = VisDrone("E:\CV\data\\visdrone")
    out = dataset._load_samples('train')
    pass