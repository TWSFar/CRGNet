import os
import cv2
import h5py
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import CRG2Net as Model
from dataloaders import deeplab_transforms as dtf

import torch
from torchvision import transforms
import multiprocessing

# from configs.cdm_visdrone import opt
# from configs.cdm_uavdt import opt
# from configs.cdm_dota import opt
from configs.cdm_tt100k import opt

multiprocessing.set_start_method('spawn', True)
user_dir = osp.expanduser('~')


def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--dataset', type=str, default='DOTA',
                        choices=['Visdrone', 'TT100K', 'DOTA', 'UAVDT'], help='dataset name')
    parser.add_argument('--checkpoint', type=str, default="/home/twsf/work/CRGNet/object-count/run/DOTA/20201115_15_train/model_best.pth.tar")
    # parser.add_argument('--set_dir', type=str, default="/home/twsf/data/TT100K/ImageSets/val.txt")
    # parser.add_argument('--img_dir', type=str, default="/home/twsf/data/TT100K/JPEGImages/")
    # parser.add_argument('--results_dir', type=str, default="/home/twsf/data/TT100K/predict_mask")
    parser.add_argument('--show', type=bool, default=False)
    args = parser.parse_args()
    args.set_dir = user_dir + f"/data/{args.dataset}/ImageSets/val.txt"
    args.img_dir = user_dir + f"/data/{args.dataset}/JPEGImages/"
    args.results_dir = user_dir + f"/data/{args.dataset}/predict_mask"
    args.imgType = '.png'
    return args


args = parse_args()
opt._parse({})


def test():
    input_w, input_h = opt.input_size
    mask_size = (int(input_h / 16), int(input_w / 16))
    if not osp.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # data
    imgs_name = []
    with open(args.set_dir, 'r') as f:
        for line in f.readlines():
            imgs_name.append(line.strip() + args.imgType)

    transform = transforms.Compose([
        dtf.FixedNoMaskResize(size=opt.input_size),  # 513
        dtf.Normalize(**opt.norm_cfg),
        dtf.ToTensor()])

    # model
    model = Model(opt).to(opt.device)

    # resume
    if osp.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.checkpoint, checkpoint['epoch']))
    else:
        raise FileNotFoundError

    model.eval()
    with torch.no_grad():
        for ii, img_name in enumerate(tqdm(imgs_name)):
            img_path = osp.join(args.img_dir, img_name)
            img = cv2.imread(img_path)[:, :, ::-1]  # BGR2RGB
            sample = {"image": img, "label": None}
            sample = transform(sample)

            # predict
            region_pred, density_pred = model(sample['image'].unsqueeze(0).to(opt.device))

            region_pred = np.argmax(region_pred.cpu().numpy(), axis=1).reshape(mask_size)
            density_pred = torch.clamp(density_pred, min=0.00018).cpu().numpy().reshape(mask_size)
            pred = region_pred * density_pred * opt.norm_cfg['para']

            file_name = osp.join(
                args.results_dir, osp.splitext(img_name)[0] + ".hdf5")
            with h5py.File(file_name, 'w') as hf:
                hf['label'] = pred

            if args.show:
                plt.figure()
                plt.subplot(2, 1, 1).imshow(img)
                plt.subplot(2, 1, 2).imshow(pred)
                plt.show()

    # with open(os.path.join( + '.txt'), 'w') as f:
    #     temp = [osp.splitext(img_name)[0]+'\n' for img_name in imgs_name]
    #     f.writelines(temp)


if __name__ == '__main__':
    test()
