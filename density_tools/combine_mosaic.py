import os
import cv2
import utils
import random
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring
user_dir = osp.expanduser('~')
# accord shiff y,x
hy = [
     {'0': [0, 0], '1': [0, 150], '2': [150, 0], '3': [150, 150]},
     {'0': [0, 350], '1': [0, 0], '2': [150, 350], '3':[150, 0]},
     {'0': [350, 0], '1': [350, 150], '2': [0, 0], '3':[0, 150]},
     {'0': [350, 350], '1': [350, 0], '2': [0, 350], '3':[0, 0]},
]


def make_xml(img, boxList, labelList, image_name, xmldir):
    bboxes = []
    labels = []
    for i in range(len(boxList)):
        for box, label in zip(boxList[i], labelList[i]):
            bboxes.append(box)
            labels.append(label)

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'Visdrone'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_object_num = SubElement(node_root, 'object_num')
    node_object_num.text = str(len(bboxes))

    node_location = SubElement(node_root, 'location')
    node_loc_xmin = SubElement(node_location, 'xmin')
    node_loc_xmin.text = str(1)
    node_loc_ymin = SubElement(node_location, 'ymin')
    node_loc_ymin.text = str(1)
    node_loc_xmax = SubElement(node_location, 'xmax')
    node_loc_xmax.text = str(1)
    node_loc_ymax = SubElement(node_location, 'ymax')
    node_loc_ymax.text = str(1)

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(img.shape[1])
    node_height = SubElement(node_size, 'height')
    node_height.text = str(img.shape[0])
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i, bbox in enumerate(bboxes):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(labels[i])
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        # voc dataset is 1-based
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(bbox[0] + 1)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(bbox[1] + 1)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(bbox[2] + 1)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(bbox[3]+ 1)

    xml = tostring(node_root, encoding='utf-8')
    dom = parseString(xml)
    with open(xmldir+image_name[:-4]+".xml", 'w') as f:
        f.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8'))
    return dom


def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--dataset', type=str, default='DOTA',
                        choices=['DOTA', 'Visdrone', 'TT100K', 'UAVDT'])
    parser.add_argument('--scaleRange', type=int, default=[150, 350])
    parser.add_argument('--mosaicNum', type=int, default=5000)
    parser.add_argument('--show', type=bool, default=False)
    args = parser.parse_args()
    args.db_root = user_dir + f'/data/{args.dataset}/'
    args.Imgdir = args.db_root + '/Mosaic/JPEGImages/'  # 候选图像的位置
    args.Annodir = args.db_root + '/Mosaic/Annotations/'  # 候选图像的标签
    args.AugImg = args.db_root + '/density_chip/JPEGImages/'  # 马赛克存储位置
    args.Xmldir = args.db_root + '/density_chip/Annotations/'  # 标签存储位置
    args.Setdir = args.db_root + '/density_chip/ImageSets/Main'  # 标签存储位置
    return args


def scaleSpilt(addre, listRange):
    scaleList = [[], [], [], []]
    imgName = os.listdir(addre)
    for file_name in imgName:
        img = cv2.imread(addre+file_name)
        if img.shape[0] == listRange[0] and img.shape[1] == listRange[0]:
            scaleList[0].append(file_name)
        elif img.shape[0] == listRange[0] and img.shape[1] == listRange[1]:
            scaleList[1].append(file_name)
        elif img.shape[0] == listRange[1] and img.shape[1] == listRange[0]:
            scaleList[2].append(file_name)
        else:
            scaleList[3].append(file_name)
    for listOne in scaleList:
        random.shuffle(listOne)
    return scaleList


def getGt(anno_path):
    box_all = []
    gt_cls = []
    xml = ET.parse(anno_path).getroot()
    # y1, x1, y2, x2
    pts = ['xmin', 'ymin', 'xmax', 'ymax']
    # bounding boxes
    for obj in xml.iter('object'):
        bbox = obj.find('bndbox')
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        box_all.append(bndbox)
        cls = obj.find('name').text
        gt_cls.append(cls)
    return box_all, gt_cls


def getImg(scaleList, listIndex, Imgdir, Annodir, index):
    img = cv2.imread(Imgdir+scaleList[listIndex][index % len(scaleList[listIndex])])
    anno_path = Annodir + scaleList[listIndex][index % len(scaleList[listIndex])][0:-4] + ".xml"
    gtBox, gtClass = getGt(anno_path)
    return img, img.shape[0], img.shape[1], gtBox, gtClass


def drawBox(img, boxlist):
    for i in range(len(boxlist)):
        for box in boxlist[i]:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    return img


def pastImage(scaleList, scaleRange, Imgdir, Annodir, Xmldir, maxNum, show):
    boxlist = [[], [], [], []]
    classlist = [[], [], [], []]
    img_list = []
    # weight = float(1000)/sum(scaleRange)
    weight = 1
    for j in tqdm(range(maxNum)):
        img = np.zeros((500, 500, 3))
        beginImg = np.random.randint(4)
        for i in range(4):
            pastImg, y, x, boxlist[i], classlist[i] = getImg(scaleList, i, Imgdir, Annodir, j)
            xbegin = hy[beginImg][str(i)][1]
            ybegin = hy[beginImg][str(i)][0]
            img[ybegin:ybegin+y, xbegin:xbegin+x, :] = pastImg
            # b ox according shift and enlarge
            for box in boxlist[i]:
                box[0] = int((box[0] + xbegin) * weight)
                box[2] = int((box[2] + xbegin) * weight)
                box[1] = int((box[1] + ybegin) * weight)
                box[3] = int((box[3] + ybegin) * weight)

        # drawBox
        if show:
            MosaicImg = drawBox(img, boxlist)
            utils.show_image(MosaicImg/255.0)
        image_name = "mosaic_"+str(j)+".jpg"

        cv2.imwrite(args.AugImg+image_name, img.astype(np.uint8))
        make_xml(img, boxlist, classlist, image_name, Xmldir)
        img_list.append("mosaic_"+str(j))
    with open(args.Setdir + '/mosaic.txt', 'w') as f:
        for line in img_list:
            f.writelines(line+'\n')


if __name__ == "__main__":
    args = parse_args()
    if not osp.exists(args.Xmldir):
        os.makedirs(args.Xmldir)
    if not osp.exists(args.AugImg):
        os.makedirs(args.AugImg)
    if not osp.exists(args.Setdir):
        os.makedirs(args.Setdir)
    scaleList = scaleSpilt(args.Imgdir, args.scaleRange)
    pastImage(scaleList, args.scaleRange, args.Imgdir, args.Annodir, args.Xmldir, args.mosaicNum, args.show)