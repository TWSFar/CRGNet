import os
import cv2
import fire
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

# from configs.deeplabv3_region_sample import opt
from configs.deeplabv3_region import opt
# from configs.deeplabv3_density_sample import opt
# from configs.deeplabv3_density import opt

from models import DeepLab
# from models import CSRNet
from dataloaders import deeplab_transforms as dtf
# from utils import Saver

import torch
from torchvision import transforms

import multiprocessing
multiprocessing.set_start_method('spawn', True)

show = False
results_dir = "/home/twsf/work/CRGNet/data/Visdrone_Region/SegmentationClass/"
images_dir = "/home/twsf/data/Visdrone/VisDrone2019-DET-val/images"


def test(**kwargs):
    opt._parse(kwargs)
    # saver = Saver(opt, "test")

    # data
    imgs_name = os.listdir(images_dir)
    transform = transforms.Compose([
        dtf.FixedNoMaskResize(size=opt.input_size),  # 513
        dtf.Normalize(opt.mean, opt.std),
        dtf.ToTensor()])

    # model
    model = DeepLab(opt).to(opt.device)

    # resume
    if os.path.isfile(opt.pre):
        print("=> loading checkpoint '{}'".format(opt.pre))
        checkpoint = torch.load(opt.pre)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.pre, checkpoint['epoch']))
    # else:
    #     raise FileNotFoundError

    model.eval()
    with torch.no_grad():
        for ii, img_name in enumerate(tqdm(imgs_name)):
            img_path = osp.join(images_dir, img_name)
            img = cv2.imread(img_path)[:, :, ::-1]  # BGR2RGB
            sample = {"image": img, "label": None}
            sample = transform(sample)

            # predict
            output = model(sample['image'].unsqueeze(0).to(opt.device))

            if opt.output_channels > 1:
                pred = np.argmax(output.cpu().numpy(), axis=1)
            else:
                pred = torch.round(output.cpu().data).numpy()
            pred = pred.reshape(pred.shape[-2:]).astype(np.uint8)

            file_name = osp.join(
                results_dir, osp.splitext(img_name)[0] + ".png")
            cv2.imwrite(file_name, pred)

            if show:
                plt.figure(figsize=(10, 10))
                plt.subplot(2, 1, 1).imshow(img)
                plt.subplot(2, 1, 2).imshow(pred)
                plt.show()

    # with open(os.path.join( + '.txt'), 'w') as f:
    #     temp = [osp.splitext(img_name)[0]+'\n' for img_name in imgs_name]
    #     f.writelines(temp)


if __name__ == '__main__':
    fire.Fire(test)
