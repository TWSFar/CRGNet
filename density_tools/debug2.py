"""
visdrone:
    if we use mmdetection train and val,
    id begin idx 1 is best. becase, in mmdetection, the default
    classes's number is n+1, we don't know the background's idx.
    However, in other detector, the idx begin 0 or 1 is need conside.

    note:
        classes = (plane, ship, storage-tank, baseball-diamond,
        tennis-court, basketball-court, ground-track-field,
        harbor, bridge, small-vehicle, large-vehicle, helicopter,
        roundabout, soccer-ball-field, swimming-pool, container-crane)
"""
import os
import cv2
import os.path as osp
import numpy as np
from tqdm import tqdm
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import matplotlib.pyplot as plt


hyp = {
    'dataset': 'DOTA15',
    'img_type': '.png',
    'mode': 'val',  # for save Set: train.txt
    'data_dir': '/home/twsf/data/DOTA15/',
    'show': False
}
hyp['xml_dir'] = osp.join(hyp['data_dir'], 'Annotations_all')
hyp['txt_dir'] = osp.join(hyp['data_dir'], 'Annotations_txt')
hyp['img_dir'] = osp.join(hyp['data_dir'], 'JPEGImages')
hyp['set_dir'] = osp.join(hyp['data_dir'], 'ImageSets')

classes = ('plane', 'ship', 'storage-tank', 'baseball-diamond',
               'tennis-court', 'basketball-court', 'ground-track-field',
               'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',
               'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane')


def show_image(img, labels):
    plt.figure(figsize=(10, 10))
    plt.subplot().imshow(img)
    plt.plot(labels[:, [0, 0, 2, 2, 0]].T, labels[:, [1, 3, 3,  1, 1]].T, '-')
    plt.savefig('test_0.jpg')
    plt.show()


def getGTBox_DOTA(anno_path, **kwargs):
    box_all = []
    gt_cls = []
    difficult = []
    with open(anno_path, 'r') as f:
        for line in f.readlines():
            data = line.split()
            if len(data) == 10 and data[8].strip() in classes:
                box_all.append([float(data[0]), float(data[1]), float(data[4]), float(data[5])])
                gt_cls.append(str(data[8].strip()))
                difficult.append(int(data[9]))

    return box_all, gt_cls, difficult


def make_xml(box_list, label_list, difficult_list, image_name, tsize):
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
        node_name.text = str(label_list[i])
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = str(difficult_list[i])

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
    setList = []
    if not osp.exists(hyp['xml_dir']):
        os.makedirs(hyp['xml_dir'])

    # txt_files = os.listdir(hyp['txt_dir'])
    set_list = []
    with open(hyp['set_dir']+'/{}_sall.txt'.format(hyp['mode'])) as f:
        for line in f.readlines():
            set_list.append(line.strip())

    for line in tqdm(set_list):
        file_name = line + '.png'
        anno_txt = osp.join(hyp['txt_dir'], line+'.txt')
        box_all, gt_cls, difficult = getGTBox_DOTA(anno_txt)

        # image info
        img_name = file_name[:-4] + hyp['img_type']  # image name
        img_path = osp.join(hyp['img_dir'], img_name)  # image path
        img = plt.imread(img_path)
        tsize = img.shape[:2]

        # del
        del_list = np.zeros(len(gt_cls), dtype=np.bool)
        for i, box in enumerate(box_all):
            if box[2] > tsize[1]:
                box_all[i][2] = tsize[1]
            if box[3] > tsize[0]:
                box_all[i][3] = tsize[0]
            if box[0] >= tsize[1] or box[1] >= tsize[0] or box[0] >= box[2] or box[1] >= box[3]:
                del_list[i] = True

        box_all = np.array(box_all)[~del_list]
        gt_cls = np.array(gt_cls)[~del_list]
        if len(gt_cls) == 0:
            continue

        dom = make_xml(box_all, gt_cls, difficult, img_name, tsize)

        setList.append(file_name[:-4])
        # save
        anno_xml = os.path.join(hyp['xml_dir'], file_name[:-4] + '.xml')
        with open(anno_xml, 'w') as fx:
            fx.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8'))

        if hyp['show']:
            show_image(img, np.array(box_all))

    # save ImageSet
    if not osp.exists(hyp['set_dir']):
        os.makedirs(hyp['set_dir'])
    setPath = osp.join(hyp['set_dir'], hyp['mode'] + '_all.txt')
    with open(setPath, 'w') as f:
        for line in setList:
            f.writelines(line + '\n')
    pass
