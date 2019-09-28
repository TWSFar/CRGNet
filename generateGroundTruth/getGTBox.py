import xml.etree.ElementTree as ET
import numpy as np


def getGTBox(anno_path, index=None):
    box_all = []
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

    elif 'txt' in anno_path:
        with open(anno_path, 'r') as f:
            data = [x.strip().split(',')[:8] for x in f.readlines()]
            annos = np.array(data)

        bboxes = annos[annos[:, 4] == '1'][:, :6].astype(np.int32)
        for bbox in bboxes:
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            box_all.append(bbox[:4].tolist())

    elif 'json' in anno_path:
        pass

    else:
        print('No such type {}'.format(type))

    return box_all
