import os
import pickle
import numpy as np
import os.path as osp
from PIL import Image
import xml.etree.ElementTree as ET
IMG_ROOT = "JPEGImages"
ANNO_ROOT = "Annotations"


class TT100K(object):
    def __init__(self, db_root):
        self.set_dir = db_root + '/ImageSets'
        self.img_dir = osp.join(db_root, IMG_ROOT)
        self.anno_dir = osp.join(db_root, ANNO_ROOT)
        self.density_voc_dir = db_root + '/density_mask'
        self.detect_voc_dir = db_root + '/density_chip'
        self.cache_dir = osp.join(db_root, 'cache')
        self._init_path()

    def _init_path(self):
        if not osp.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_imglist(self, split='train'):
        """ return list of all image paths
        """
        set_file = osp.join(self.set_dir, split+'.txt')
        img_list = []
        with open(set_file) as f:
            for line in f.readlines():
                img_list.append(osp.join(self.img_dir, line.strip()+'.jpg'))
        return img_list

    def _get_annolist(self, split):
        """ annotation type is '.txt'
        return list of all image annotation path
        """
        img_list = self._get_imglist(split)
        return [img.replace(IMG_ROOT, ANNO_ROOT).replace('jpg', 'xml')
                for img in img_list]

    def _get_gtbox(self, anno_xml, **kwargs):
        img_path = anno_xml.replace(ANNO_ROOT, IMG_ROOT).replace('xml', 'jpg')
        box_all = []
        gt_cls = []
        xml = ET.parse(anno_xml).getroot()
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        size = xml.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        # bounding boxes
        for obj in xml.iter('object'):
            bbox = obj.find('bndbox')
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            box_all += [bndbox]
            gt_cls.append(obj.find('name').text)
        return {'bboxes': np.array(box_all, dtype=np.float64),
                'cls': gt_cls,
                'width': width,
                'height': height,
                'image': img_path}  # cls id run from 0

    def _load_samples(self, split):
        cache_file = osp.join(self.cache_dir, split + '_samples.pkl')

        # load bbox and save to cache
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                samples = pickle.load(fid)
            print('gt samples loaded from {}'.format(cache_file))
            return samples

        # load information of image and save to cache
        img_list = self._get_imglist(split)
        anno_path = [img_path.replace(IMG_ROOT, ANNO_ROOT).replace('jpg', 'xml')
                     for img_path in img_list]
        samples = [self._get_gtbox(ann) for ann in anno_path]

        with open(cache_file, 'wb') as fid:
            pickle.dump(samples, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt samples to {}'.format(cache_file))

        return samples


if __name__ == "__main__":
    dataset = TT100K("/home/twsf/data/TT100K")
    out = dataset._load_samples('train')
    pass
