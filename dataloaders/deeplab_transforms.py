import cv2
import random
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter
import torch


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        gt = sample['label']
        img = img.astype(np.float32)
        gt = gt.astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': gt}


class RandomColorJeter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.tr = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        sample['image'] = self.tr(Image.fromarray(sample['image']))
        sample['image'] = np.array(sample['image'])

        return sample


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        gt = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': gt}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        gt = sample['label']
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            gt = gt[:, ::-1]

        return {'image': img,
                'label': gt}


class FixedNoMaskResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size  # size: (w, h)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        img = cv2.resize(img, self.size)

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        gt = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        gt = np.array(gt).astype(np.float32)

        img = torch.from_numpy(img).float()
        gt = torch.from_numpy(gt).float()

        return {'image': img,
                'label': gt}


if __name__ == "__main__":
    from torchvision import transforms
    img = cv2.imread("/home/twsf/work/CRGNet/data/Visdrone_Region/JPEGImages/0000001_02999_d_0000005.jpg")
    gt = cv2.imread("/home/twsf/work/CRGNet/data/Visdrone_Region/SegmentationClass/0000001_02999_d_0000005.png")
    pair = {'image': img, 'label': gt}
    model = transforms.Compose([
            FixedNoMaskResize(size=(640, 480)),
            RandomColorJeter(0.3, 0.3, 0.3, 0.3),
            RandomHorizontalFlip(),
            Normalize(),
            ToTensor()])
    sample = model(pair)
    pass
