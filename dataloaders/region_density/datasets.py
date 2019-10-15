import os
import cv2
import math
import h5py
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))
from dataloaders.density_region.transforms import transfrom


def BiBubic(x):
    x = abs(x)
    if x <= 1:
        return 1 - 2 * (x**2) + (x**3)
    elif x < 2:
        return 4 - 8 * x + 5 * (x**2) - (x**3)
    else:
        return 0


def BiCubic_interpolation(input, out_scale=(30, 40)):
    scrH, scrW = input.shape
    dstH, dstW = out_scale
    output = np.zeros(out_scale)
    ratio_x = (scrH / dstH)
    ratio_y = (scrW / dstW)
    for i in range(dstH):
        for j in range(dstW):
            scrx = i * ratio_x
            scry = j * ratio_y
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx - x
            v = scry - y
            tmp = 0
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if x+ii < 0 or y+jj < 0 or x+ii >= scrH or y+jj >= scrW:
                        continue
                    tmp += input[x+ii, y+jj] * BiBubic(ii-u) * BiBubic(jj-v)
            output[i, j] = tmp

    return output


class Datasets(Dataset):
    def __init__(self, data_dir, train=True):
        super().__init__()
        self.data_dir = data_dir
        self.img_dir = osp.join(self.data_dir, 'images')
        self.img_list = [file for file in os.listdir(self.img_dir)]
        self.train = train
        self.img_number = len(self.img_list)

        # transform
        self.stf = transfrom(self.train)

    def __getitem__(self, index):
        img_path = osp.join(self.img_dir, self.img_list[index])
        gt_path = img_path.replace('.jpg', '.h5').replace('images', 'density_map')
        assert osp.isfile(img_path), '{} not exist'.format(img_path)
        assert osp.isfile(gt_path), '{} not exist'.format(gt_path)

        img = cv2.imread(img_path)[:, :, ::-1]  # BGR2RGB
        gt_file = h5py.File(gt_path, 'r')
        target = np.asarray(gt_file['density'])

        o_h, o_w = img.shape[:2]

        img, target = self.stf(img, target)
        scale = torch.tensor([o_h / img.shape[1],
                              o_w / img.shape[2]])
        return img, target, scale

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = Datasets("/home/twsf/data/Shanghai/part_B_final/train_data", train=False)
    data = dataset.__getitem__(0)
    dataloader = DataLoader(dataset, batch_size=20)
    for (img, label, scale) in dataloader:
        pass
    pass
