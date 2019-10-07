import torch
from torch.utils.data import Dataset

import os
import cv2
import pickle
import numpy as np
from PIL import Image
import os.path as osp
import xml.etree.ElementTree as ET
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from dataloaders.transforms import train_transforms, test_transforms, normalize


def getGTBox(anno_path, **kwargs):
    box_all = []
    gt_cls = []
    if 'xml' in anno_path:
        xml = ET.parse(anno_path).getroot()
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        # bounding boxes
        for obj in xml.iter('object'):
            bbox = obj.find('bndbox')
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                bndbox.append(cur_pt)
            box_all += [bndbox]
            cls = obj.find('name').text
            gt_cls.append(kwargs['class_to_id'][cls])

    elif 'txt' in anno_path:
        with open(anno_path, 'r') as f:
            data = [x.strip().split(',')[:8] for x in f.readlines()]
            annos = np.array(data)

        bboxes = annos[annos[:, 4] == '1'][:, :6].astype(np.int32)
        for bbox in bboxes:
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            box_all.append(bbox[:4].tolist())
            gt_cls.append(int(bbox[5]))  # index 5 is classes id

    elif 'json' in anno_path:
        pass

    else:
        print('No such type {}'.format(type))
        raise NotImplementedError

    return {'bboxes': np.array(box_all),
            'cls': np.array(gt_cls)}


def cre_cache_path(data_dir):
    cache_path = osp.join(data_dir, 'cache')
    if not osp.exists(cache_path):
        os.makedirs(cache_path)
    return cache_path


def _load_samples(self):
    cache_file = self.cache_file
    anno_file = self.anno_file

    # load bbox and save to cache
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            samples = pickle.load(fid)
        print('{} gt samples loaded from {}'.
              format(self.mode, cache_file))
        return samples

    # load information of image and save to cache
    sizes = [Image.open(osp.join(self.img_dir, index+self.img_type)).size
             for index in self.im_ids]

    samples = [getGTBox(anno_file.format(index), class_to_id=self.class_to_id)
               for index in self.im_ids]

    for i, index in enumerate(self.im_ids):
        samples[i]['image'] = osp.join(self.img_dir, index+self.img_type)
        samples[i]['width'] = sizes[i][0]
        samples[i]['height'] = sizes[i][1]

    with open(cache_file, 'wb') as fid:
        pickle.dump(samples, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt samples to {}'.format(cache_file))

    return samples


def getitem(self, index):
    assert index <= len(self), 'index range error'
    img_path = self.samples[index]['image']
    img = cv2.imread(img_path)  # BGR
    scrimg_shape = np.array(img.shape[0:2])
    assert img is not None, 'File Not Found ' + img_path

    bboxes = self.samples[index]['bboxes']
    gt_cls = self.samples[index]['cls']
    target = np.hstack((bboxes, np.expand_dims(gt_cls, axis=1)))

    if self.input_size is None:
        self.input_size = scrimg_shape
    if self.mode in ['train', 'trainval', 'trainaug']:
        img, target = train_transforms(img,
                                       target,
                                       self.input_size)
    elif self.mode in ['test', 'val']:
        img, target = test_transforms(img,
                                      target,
                                      self.input_size)

    show_image(img, target)
    nL = len(target)
    if nL > 0:
        target[:, [1, 3]] = np.clip(target[:, [1, 3]], 0, img.shape[0])
        target[:, [0, 2]] = np.clip(target[:, [0, 2]], 0, img.shape[1])
        target[:, [1, 3]] = target[:, [1, 3]] / img.shape[0]  # height
        target[:, [0, 2]] = target[:, [0, 2]] / img.shape[1]  # width

    # Normalize
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)
    img = normalize(img)

    bboxes = torch.zeros((nL, 4))
    labels = torch.zeros((nL, 1))
    if nL:
        bboxes[:, :] = torch.from_numpy(target[:, :4])
        labels[:, :] = torch.from_numpy(target[:, 4:5])

    img = torch.from_numpy(img).float()
    bboxes, labels = bboxes.float(), labels.float()
    scale = (self.input_size[0] / scrimg_shape[0],
             self.input_size[1] / scrimg_shape[1])

    return img, bboxes, labels, scale


class HKB(Dataset):
    classes = ('__background__',  # always index 0
               'Vehicle')

    def __init__(self, opt, mode='train'):
        super().__init__()
        self.mode = mode
        self.opt = opt

        # Path and File
        self.data_dir = opt.data_dir
        self.img_dir = osp.join(self.data_dir, 'JPEGImages')
        self.ann_dir = osp.join(self.data_dir, 'Annotations')
        self.cache_path = cre_cache_path(self.data_dir)
        self.cache_file = osp.join(self.cache_path, self.mode + '_samples.pkl')
        self.anno_file = os.path.join(self.ann_dir, '{}.xml')

        # Dataset information
        self.img_type = '.jpg'
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.im_ids = self._load_image_set_index()
        self.num_images = len(self.im_ids)
        self.input_size = opt.input_size

        # bounding boxes and image information
        self.samples = _load_samples(self)

    def __getitem__(self, index):
        return getitem(self, index)

    def _load_image_set_index(self):
        """Load the indexes listed in this dataset's image set file.
        """
        image_index = []
        image_set_file = os.path.join(self.data_dir, 'ImageSets', 'Main',
                                      self.mode + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            for line in f.readlines():
                image_index.append(line.strip())
        return image_index

    def __len__(self):
        return self.num_images


class VOC(Dataset):
    classes = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, opt, mode='train'):
        super().__init__()
        self.mode = mode
        self.opt = opt

        # Path and File
        self.data_dir = opt.data_dir
        self.img_dir = osp.join(self.data_dir, 'JPEGImages')
        self.ann_dir = osp.join(self.data_dir, 'Annotations')
        self.cache_path = cre_cache_path(self.data_dir)
        self.cache_file = osp.join(self.cache_path, self.mode + '_samples.pkl')
        self.anno_file = os.path.join(self.ann_dir, '{}.xml')

        # Dataset information
        self.img_type = '.jpg'
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.im_ids = self._load_image_set_index()
        self.num_images = len(self.im_ids)
        self.input_size = opt.input_size

        # bounding boxes and image information
        self.samples = _load_samples(self)

    def __getitem__(self, index):
        return getitem(self, index)

    def _load_image_set_index(self):
        """Load the indexes listed in this dataset's image set file.
        """
        image_index = []
        image_set_file = os.path.join(self.data_dir, 'ImageSets', 'Main',
                                      self.mode + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            for line in f.readlines():
                image_index.append(line.strip())
        return image_index

    def __len__(self):
        return self.num_images


class VisDrone(Dataset):
    classes = ('__background__',  # always index 0
               'ignored regions', 'pedestrian', 'people',
               'bicycle', 'car', 'van', 'truck', 'tricycle',
               'awning-tricycle', 'bus', 'motor', 'others')

    def __init__(self, opt, mode='train'):
        super().__init__()
        self.mode = mode
        self.opt = opt

        # Path and File
        self.data_dir = opt.data_dir
        self.img_dir = osp.join(self.data_dir, 'images')
        self.ann_dir = osp.join(self.data_dir, 'annotations')
        self.cache_path = cre_cache_path(self.data_dir)
        self.cache_file = osp.join(self.cache_path, self.mode + '_samples.pkl')
        self.anno_file = os.path.join(self.ann_dir, '{}.txt')

        # Dataset information
        self.img_type = '.jpg'
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.im_ids = self._load_image_set_index()
        self.num_images = len(self.im_ids)
        self.input_size = opt.input_size

        # bounding boxes and image information
        self.samples = _load_samples(self)

    def _load_image_set_index(self):
        """Load the indexes listed in this dataset's image set file.
        """
        image_index = []
        image_set = os.listdir(self.img_dir)
        for line in image_set:
            image_index.append(line[:-4])  # type of image is .jpg
        return image_index

    def __getitem__(self, index):
        return getitem(self, index)

    def __len__(self):
        return self.num_images


class COCO(Dataset):
    classes = ('__background__',  # always index 0
               'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
               'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self, opt, mode='train'):
        pass

    def __getitem__(self, index):
        return getitem(self, index)

    def coco_class_weights(self):  # frequency of each class in coco train2014
        n = [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
             6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
             4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
             5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
             1877, 17630, 4337, 4624, 1075, 3468, 135, 1380]
        weights = 1 / torch.Tensor(n)
        weights /= weights.sum()
        # with open('data/coco.names', 'r') as f:
        #     for k, v in zip(f.read().splitlines(), n):
        #         print('%20s: %g' % (k, v))
        return weights

    def coco80_to_coco91_class(self):  # converts 80-index (val2014) to 91-index (paper)
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
             35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
             64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return x

    def __len__(self):
        return self.num_images


class DOTA(Dataset):
    classes = {'Vehicle', }

    def __init__(self, opt, mode='train'):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.num_images


def Datasets(opt, data_dir=None, mode='train'):
    if data_dir is not None:
        opt.data_dir = data_dir
    print('{} dataset from path of {}'.format(opt.dataset, opt.data_dir))

    if 'hkb' in opt.dataset:
        return HKB(opt, mode)

    elif 'visdrone' in opt.dataset:
        return VisDrone(opt, mode)

    elif 'voc' in opt.dataset:
        return VOC(opt, mode)

    elif 'coco' in opt.dataset:
        return COCO(opt, mode)

    elif 'dota' in opt.dataset:
        return DOTA(opt, mode)

    else:
        print('Dataset {} not available.'.format(opt.dataset))
        raise NotImplementedError


def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-')
    plt.show()
    pass


if __name__ == '__main__':
    from easydict import EasyDict as edict
    opt = edict()
    opt.dataset = 'hkb'
    opt.input_size = None
    opt.data_dir = "G:\\CV\\Reading\\Faster-RCNN\\data\\HKB"
    dataset = Datasets(opt, mode='test')
    img, target, _, _ = dataset.__getitem__(0)
    pass
