import os
import zipfile
import argparse
import numpy as np
import os.path as osp
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm

temp = os.listdir('/home/twsf/data/DOTA/source_images/')

imgs = []
with open('/home/twsf/data/DOTA/ImageSets/train_sall.txt', 'r') as f:
    imgs = [line.strip() for line in f.readlines()]

with open('/home/twsf/data/DOTA/ImageSets/val_sall.txt', 'r') as f:
    for line in f.readlines():
        imgs.append(line.strip())

imgs2 = [line[:-4] for line in temp]

last = []
with open('lt.txt', 'w') as f:
    for line in imgs:
        if line not in imgs2:
            f.writelines(line + '.png\n')