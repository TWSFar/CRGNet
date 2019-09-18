import xml.etree.ElementTree as ET
import numpy as np


def getGTBox(img_path, type='xml'):
    anno_path = img_path.replace('JPEGImages', 'Annotations')
    anno_path = anno_path.replace('jpg', type)
    box_all = []

    if type == 'xml':
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

    elif type == 'txt':
        with open(anno_path, 'r') as f:
            data = [x.strip().split(',')[:8] for x in f.readlines()]
            annos = np.array(data)

        boxes = annos[annos[:, 4] == '1'][:, :6].astype(np.int32)
        for box in boxes:
            box[2] += box[0]
            box[3] += box[1]
            box_all += box.tolist()

    else:
        print('No such type {}'.format(type))

    return box_all
