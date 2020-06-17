import os
import shutil
import numpy as np
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, tostring


def getGTBox(anno_path, **kwargs):
    box_all = []
    xml = ET.parse(anno_path).getroot()

    size = xml.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)

    # y1, x1, y2, x2
    pts = ['xmin', 'ymin', 'xmax', 'ymax']
    # bounding boxes
    for obj in xml.iter('object'):
        bbox = obj.find('bndbox')
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        bndbox.append(int(obj.find('name').text))
        box_all += [bndbox]

    return box_all, (int(height), int(width))


def make_xml(box_list, image_name, tsize):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = "UAVDT"

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
        node_name.text = str(box_list[i][4])
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

    xml_list = os.listdir("/home/twsf/data/UAVDT/Annotations")
    for xml in tqdm(xml_list):
        xml_file = "/home/twsf/data/UAVDT/Annotations/" + xml
        img_name = xml[:-4] + '.jpg'
        bbox, tsize = getGTBox(xml_file)
        dom = make_xml(bbox, img_name, tsize)

        # save
        with open(xml_file, 'w') as fx:
            fx.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8'))
