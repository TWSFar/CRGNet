import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import ColorJitter
import torch


class ImgFixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):

        sample['image'] = sample['image'].resize(self.size, Image.BILINEAR)

        return sample


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(127.5., 127.5, 127.5), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        gt = sample['target']
        img = np.array(img).astype(np.float32)
        gt = np.array(gt).astype(np.float32)
        img -= self.mean
        img /= self.std
        img /= 255.0

        return {'image': img,
                'target': gt}


class RandomColorJeter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.tr = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        sample['image'] = self.tr(sample['image'])

        return sample


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        den = sample['density']
        rgn = sample['region']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'density': den,
                'region': rgn}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        den = sample['density']
        rgn = sample['region']
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            den = den[:, ::-1]
            rgn = rgn[:, ::-1]

        return {'image': img,
                'density': den,
                'region': rgn}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        den = sample['density']
        rgn = sample['region']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        den = den.rotate(rotate_degree, Image.NEAREST)
        rgn = rgn.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'density': den,
                'region': rgn}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        den = sample['density']
        rgn = sample['region']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        den = np.array(den).astype(np.float32)
        rgn = np.array(rgn).astype(np.float32)

        img = torch.from_numpy(img).float()
        den = torch.from_numpy(den).float()
        rgn = torch.from_numpy(rgn).float()

        return {'image': img,
                'density': den,
                'region': rgn}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from glob import glob
    from PIL import Image
    path = glob('G:\\CV\\Dataset\\PASCAL VOC 2012\\VOCdevkit\\VOC2012\\JPEGImages\\2007_004052*')[0]
    img = Image.open(path).convert('RGB')
    pair = {'image':img, 'label':img}
    model = RandomScaleCrop(400, 600)
    sample = model(pair)
    img2 = sample['image']
    plt.figure()
    plt.title('display')
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(img2)
    plt.show(block=True)
    pass
