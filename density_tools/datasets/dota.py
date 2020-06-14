import os
import pickle
import numpy as np
import os.path as osp
from PIL import Image
IMG_ROOT = "JPEGImages"
ANNO_ROOT = "Annotations_txt"


class DOTA(object):
    classes = ('plane', 'ship', 'storage-tank', 'baseball-diamond',
               'tennis-court', 'basketball-court', 'ground-track-field',
               'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',
               'roundabout', 'soccer-ball-field', 'swimming-pool')

    def __init__(self, db_root):
        self.set_dir = db_root + '/ImageSets'
        self.img_dir = osp.join(db_root, IMG_ROOT)
        self.anno_dir = osp.join(db_root, ANNO_ROOT)
        self.density_voc_dir = db_root + '/density_mask'
        self.detect_voc_dir = db_root + '/density_chip'
        self.cache_dir = osp.join(db_root, 'cache')
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.classes)}
        self._init_path()

    def _init_path(self):
        if not osp.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_imglist(self, split='train'):
        """ return list of all image paths
        """
        set_file = osp.join(self.set_dir, split+'_all.txt')
        img_list = []
        with open(set_file) as f:
            for line in f.readlines():
                img_list.append(osp.join(self.img_dir, line.strip()+'.png'))
        return img_list

    def _get_annolist(self, split):
        """ annotation type is '.txt'
        return list of all image annotation path
        """
        img_list = self._get_imglist(split)
        return [img.replace(IMG_ROOT, ANNO_ROOT).replace('png', 'txt')
                for img in img_list]

    def _get_gtbox(self, anno_txt, tsize, **kwargs):
        """
            tisze: (w, h), Image.open is diffed with cv2.imread
        """
        box_all = []
        gt_cls = []
        with open(anno_txt, 'r') as f:
            for line in f.readlines():
                data = line.split()
                # First, data is a grountruth info
                if len(data) == 10:
                    box = [float(data[0]), float(data[1]), float(data[4]), float(data[5])]
                    # Second, left and top must less than scale
                    if box[2] > tsize[0]:
                        box[2] = tsize[0]
                    if box[3] > tsize[1]:
                        box[3] = tsize[1]
                    # Third, box must be legal
                    if box[0] >= tsize[0] or box[1] >= tsize[1] or box[0] >= box[2] or box[1] >= box[3]:
                        continue
                    box_all.append(box)
                    gt_cls.append(self.cat2label[str(data[8].strip())])
        return {'bboxes': np.array(box_all, dtype=np.float64),
                'cls': gt_cls}  # cls id run from 0

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
        sizes = [Image.open(img).size for img in img_list]
        anno_path = [img_path.replace(IMG_ROOT, ANNO_ROOT).replace('png', 'txt')
                     for img_path in img_list]
        samples = [self._get_gtbox(ann, sizes[i]) for i, ann in enumerate(anno_path)]

        for i, img_path in enumerate(img_list):
            samples[i]['image'] = img_path  # image path
            samples[i]['width'] = sizes[i][0]
            samples[i]['height'] = sizes[i][1]

        with open(cache_file, 'wb') as fid:
            pickle.dump(samples, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt samples to {}'.format(cache_file))

        return samples


if __name__ == "__main__":
    dataset = DOTA("/home/twsf/data/DOTA")
    out = dataset._load_samples('train')
    pass
