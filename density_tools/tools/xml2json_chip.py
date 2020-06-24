import os
import json
import os.path as osp
from tqdm import tqdm
# import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import OrderedDict

hyp = {
    'help': 'voc type transform to coco type',
    'mode': 'traintest',  # save instance_train.json
    'num_class': 10,  # visdrone: 10, dota: 15, tt100k: 45, uavdt: 3
    'data_dir': '/home/twsf/data/Visdrone/density_chip',
}
hyp['json_dir'] = osp.join(hyp['data_dir'], 'Annotations_json')
hyp['xml_dir'] = osp.join(hyp['data_dir'], 'Annotations')
hyp['img_dir'] = osp.join(hyp['data_dir'], 'JPEGImages')
hyp['set_file'] = osp.join(hyp['data_dir'], 'ImageSets', 'Main', hyp['mode'] + '.txt')


class getItem(object):
    def __init__(self):
        self.classes = [str(i) for i in range(hyp['num_class'])]
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.classes)}

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
            cate['supercategory'] = str(self.cat2label[cat])
            cate['name'] = str(self.cat2label[cat])
            cate['id'] = idx
            categories.append(cate)

        return categories


def getGTBox(anno_xml, item, **kwargs):
    box_all = []
    gt_cls = []
    xml = ET.parse(anno_xml).getroot()
    pts = ['xmin', 'ymin', 'xmax', 'ymax']
    # bounding boxes
    for obj in xml.iter('object'):
        bbox = obj.find('bndbox')
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        box_all += [bndbox]
        # cls = obj.find('name').text
        gt_cls.append(item.cat2label[obj.find('name').text])

    return box_all, gt_cls


def getImgInfo(anno_xml):
    xml = ET.parse(anno_xml).getroot()
    img_name = xml.find('filename').text  # image name
    tsize = xml.find('size')
    size = {'height': int(tsize.find('height').text),
            'width': int(tsize.find('width').text)}

    chip = []
    location = []
    pts = ['xmin', 'ymin', 'xmax', 'ymax']
    lct = xml.find('location')
    for pt in pts:
        cur_pt = int(lct.find(pt).text) - 1
        chip.append(cur_pt)
    location = [chip[0], chip[1], chip[2] - chip[0], chip[3] - chip[1]]

    return img_name, size, location


def make_json():
    item = getItem()
    images = []
    annotations = []
    anno_id = 0

    # categories
    categories = item.get_cat_item()

    with open(hyp['set_file'], 'r') as f:
        xml_list = f.readlines()
    for id, file_name in enumerate(tqdm(xml_list)):
        file_name = file_name.strip()
        img_id = id

        # anno info
        anno_xml = osp.join(hyp['xml_dir'], file_name + '.xml')
        box_all, gt_cls = getGTBox(anno_xml, item)
        for ii in range(len(box_all)):
            annotations.append(
                item.get_ann_item(box_all[ii], img_id, gt_cls[ii], anno_id))
            anno_id += 1

        # image info
        img_name, size, location = getImgInfo(anno_xml)
        image = item.get_img_item(img_name, img_id, size)
        image['location'] = location
        images.append(image)

    # all info
    ann = OrderedDict()
    ann['images'] = images
    ann['categories'] = categories
    ann['annotations'] = annotations

    # saver
    if not osp.exists(hyp['json_dir']):
        os.makedirs(hyp['json_dir'])
    save_file = osp.join(hyp['json_dir'], 'instances_{}.json'.format(hyp['mode']))
    print('Saving annotations to {}'.format(save_file))
    json.dump(ann, open(save_file, 'w'), indent=4)
    print('done!')


if __name__ == '__main__':
    print(hyp)
    make_json()
