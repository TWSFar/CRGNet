"""
TT100K:
    这个数据集仅仅保留实例个数不小于100的标签

    note:
        classes = ('i2', 'i4', 'i5', 'il100', 'il60', 'il80', 'io', 'ip', 'p10', 'p11', 'p12', 'p19',
                'p23', 'p26', 'p27', 'p3', 'p5', 'p6', 'pg', 'ph4', 'ph4.5', 'ph5', 'pl100', 'pl120',
                'pl20', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl70', 'pl80', 'pm20', 'pm30',
                'pm55', 'pn', 'pne', 'po', 'pr40', 'w13', 'w32', 'w55', 'w57', 'w59', 'wo')
"""
import os
import json
import os.path as osp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

hyp = {
    'dataset': 'TT100K',
    'img_type': '.jpg',
    'mode': 'train',  # for save Set: train.txt choose: train, test
    'data_dir': '/home/twsf/data/TT100K/',
}

hyp['xml_dir'] = osp.join(hyp['data_dir'], 'Annotations')
hyp['img_dir'] = osp.join(hyp['data_dir'], 'JPEGImages')
hyp['set_dir'] = osp.join(hyp['data_dir'], 'ImageSets')
hyp['anno_file'] = osp.join(hyp['data_dir'], 'annotations.json')

classes = ('i2', 'i4', 'i5', 'il100', 'il60', 'il80', 'io', 'ip', 'p10', 'p11', 'p12', 'p19',
           'p23', 'p26', 'p27', 'p3', 'p5', 'p6', 'pg', 'ph4', 'ph4.5', 'ph5', 'pl100', 'pl120',
           'pl20', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl70', 'pl80', 'pm20', 'pm30',
           'pm55', 'pn', 'pne', 'po', 'pr40', 'w13', 'w32', 'w55', 'w57', 'w59', 'wo')
cat2label = {cat_id: i for i, cat_id in enumerate(classes)}


def getGTBox(objects):

    box_all = []
    gt_cls = []
    temp = ['xmin', 'ymin', 'xmax', 'ymax']
    for obj in objects:
        if obj['category'] not in classes:
            continue
        gt_cls.append(obj['category'])
        box = []
        for lab in temp:
            box.append(obj['bbox'][lab])
        box_all.append(box)

    return box_all, gt_cls


def make_xml(box_list, label_list, image_name, tsize):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = hyp['dataset']

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_object_num = SubElement(node_root, 'object_num')
    node_object_num.text = str(len(box_list))

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(tsize[1])  # tsize: (h, w)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(tsize[0])  # tsize: (h, w)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i in range(len(box_list)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(cat2label[label_list[i]])
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        # voc dataset is 1-based
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(box_list[i][0]) + 1)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(box_list[i][1]) + 1)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(box_list[i][2] + 1))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(box_list[i][3] + 1))

    xml = tostring(node_root, encoding='utf-8')
    dom = parseString(xml)
    # print(xml)
    return dom


if __name__ == '__main__':
    with open(hyp['anno_file']) as f:
        annotations = json.load(f)

    setList = []

    for img_id, sample in tqdm(annotations['imgs'].items()):
        if hyp['mode'] not in sample['path']:
            continue

        objects = sample['objects']
        box_all, gt_cls = getGTBox(objects)
        if len(gt_cls) == 0:
            continue

        setList.append(img_id)

        # image info
        img_name = img_id + hyp['img_type']  # image name
        img_path = osp.join(hyp['img_dir'], img_name)  # image path
        tsize = plt.imread(img_path).shape[:2]

        dom = make_xml(box_all, gt_cls, img_name, tsize)

        # save
        anno_xml = os.path.join(hyp['xml_dir'], img_id + '.xml')
        with open(anno_xml, 'w') as fx:
            fx.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8'))

    # save ImageSet
    if not osp.exists(hyp['set_dir']):
        os.makedirs(hyp['set_dir'])
    setPath = osp.join(hyp['set_dir'], hyp['mode'] + '.txt')
    with open(setPath, 'w') as f:
        for line in setList:
            f.writelines(line + '\n')
    print('{} samples generated'.format(len(setList)))
    pass
