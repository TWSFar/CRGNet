import os
import cv2
import math
import numpy as np
import os.path as osp

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))
from dataloaders import deeplab_transforms as dtf


class VisDroneSegmentation(Dataset):
    """
    Visdrone dataset
    """
    n_class = 2

    def __init__(self, opt, train=True):
        super().__init__()
        self.data_dir = opt.data_dir
        if train:
            self.data_dir = osp(self.data_dir, "VisDrone2019-DET-train")
        else:
            self.data_dir = osp(self.data_dir, "VisDrone2019-DET-val")

        self.img_dir = osp.join(self.data_dir, 'images', '{}.jpg')
        self.den_dir = osp.join(self.data_dir, 'DensityMask', '{}.png')
        self.img_ids = [osp.splitext(file)[0]
                        for file in os.listdir(self.img_dir)
                        if osp.splitext(file)[-1] == '.jpg']
        self.train = train
        self.img_number = len(self.img_list)

        # transform
        self.train_dtf = transforms.Compose([
            dtf.ImgFixedResize(size=self.opt.input_size),
            dtf.RandomColorJeter(0.3, 0.3, 0.3, 0.3),
            dtf.RandomHorizontalFlip(),
            dtf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            dtf.ToTensor()])
        self.test_dtf = transforms.Compose([
            dtf.ImgFixedResize(crop_size=self.opt.input_size),  # default = 513
            dtf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            dtf.ToTensor()])

    def __getitem__(self, index):
        id = self.img_ids[index]
        img_path = self.img_dir.format(id)
        den_path = self.den_dir.format(id)
        rgn_path = self.rgn_dir.format(id)
        assert osp.isfile(img_path), '{} not exist'.format(img_path)
        assert osp.isfile(den_path), '{} not exist'.format(den_path)
        assert osp.isfile(rgn_path), '{} not exist'.format(rgn_path)

        img = cv2.imread(img_path)[:, :, ::-1]  # BGR2RGB
        den = cv2.imread(den_path, cv2.IMREAD_GRAYSCALE)
        rgn = cv2.imread(rgn_path, cv2.IMREAD_GRAYSCALE)
        # gt_file = h5py.File(gt_path, 'r')
        # target = np.asarray(gt_file['density'])

        o_h, o_w = img.shape[:2]

        scale = torch.tensor([o_h / img.shape[1],
                              o_w / img.shape[2]])
        return img, target, scale

    def __len__(self):
        return len(self.img_ids)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = Datasets("/home/twsf/data/Shanghai/part_B_final/train_data", train=False)
    data = dataset.__getitem__(0)
    dataloader = DataLoader(dataset, batch_size=20)
    for (img, label, scale) in dataloader:
        pass
    pass
