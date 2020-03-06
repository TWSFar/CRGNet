import xml.etree.ElementTree as ET
import cv2
import numpy as np
img_id = '0000249_01635_d_0000006_3'


def getGTBox(anno_xml, **kwargs):
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
        cls = obj.find('name').text
        gt_cls.append(int(cls))

    return box_all, gt_cls


def show_image(img, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
    plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-')
    plt.show()
    pass



if __name__ == "__main__":
    anno_xml = "/home/twsf/data/Visdrone/detect_voc/Annotations/" + img_id + '.xml'
    img_file = "/home/twsf/data/Visdrone/detect_voc/JPEGImages/" + img_id + '.jpg'
    bbox, _ = getGTBox(anno_xml)
    show_image(cv2.imread(img_file), np.array(bbox))