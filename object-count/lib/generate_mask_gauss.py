"""convert VOC format
+ density_voc
    + JPEGImages
    + SegmentationClass
"""

import os
import cv2
import h5py
import shutil
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
import concurrent.futures
from scipy.ndimage.filters import gaussian_filter

from datasets import get_dataset
user_dir = osp.expanduser('~')


def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--dataset', type=str, default='VisDrone',
                        choices=['VisDrone'], help='dataset name')
    parser.add_argument('--mode', type=str, default=['train', 'val'],
                        nargs='+', help='for train or val')
    parser.add_argument('--db_root', type=str,
                        default=user_dir+"/data/Visdrone",
                        # default="E:\\CV\\data\\visdrone",
                        help="dataset's root path")
    parser.add_argument('--mask_size', type=list, default=[30, 40],
                        help="Size of production target mask")
    parser.add_argument('--maximum', type=int, default=4,
                        help="maximum of mask")
    parser.add_argument('--show', type=bool, default=False,
                        help="show image and region mask")
    args = parser.parse_args()
    return args


def show_image(img, labels, mask):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1).imshow(img)
    plt.plot(labels[:, [0, 0, 2, 2, 0]].T, labels[:, [1, 3, 3,  1, 1]].T, '-')
    plt.subplot(2, 1, 2).imshow(mask)
    # plt.savefig('test_0.jpg')
    plt.show()


# copy train and test images
def _copy(src_image, dest_path):
    shutil.copy(src_image, dest_path)


def _myaround_up(value, maxv):
    """0.05 * stride = 0.8"""
    tmp = np.floor(value).astype(np.int32)
    return min(maxv, tmp + 1 if value - tmp > 0.05 else tmp)


def _myaround_down(value):
    """0.05 * stride = 0.8"""
    tmp = np.ceil(value).astype(np.int32)
    return max(0, tmp - 1 if tmp - value > 0.05 else tmp)

def gaussian_filter_density():
    pt2d = np.zeros((mask_h, mask_w), dtype=np.float32)
def _generate_mask(sample, mask_scale=(30, 40)):
    try:
        height, width = sample["height"], sample["width"]

        # Chip mask 40 * 30, model input size 640x480
        mask_h, mask_w = mask_scale
        density_mask = np.zeros((mask_h, mask_w), dtype=np.float32)

        for box in sample["bboxes"]:
            xmin = _myaround_down(1.0 * box[0] / width * mask_w)
            ymin = _myaround_down(1.0 * box[1] / height * mask_h)
            xmax = _myaround_up(1.0 * box[2] / width * mask_w, mask_w-1)
            ymax = _myaround_up(1.0 * box[3] / height * mask_h, mask_h-1)
            if xmin == xmax or ymin == ymax:
                continue
            if args.mask_type = 
            gaussian_filter_density()
            pt2d = np.zeros((mask_h, mask_w), dtype=np.float32)

            density_mask += pt2d[]

        return density_mask.clip(min=0, max=args.maximum)

    except Exception as e:
        print(e)
        print(sample["images"])


if __name__ == "__main__":
    args = parse_args()

    dataset = get_dataset(args.dataset, args.db_root)
    dest_datadir = dataset.density_voc_dir
    image_dir = dest_datadir + '/JPEGImages'
    mask_dir = dest_datadir + '/SegmentationClass'
    annotation_dir = dest_datadir + '/Annotations'
    list_folder = dest_datadir + '/ImageSets'

    if not osp.exists(dest_datadir):
        os.mkdir(dest_datadir)
        os.mkdir(image_dir)
        os.mkdir(mask_dir)
        os.mkdir(annotation_dir)
        os.mkdir(list_folder)

    for split in args.mode:
        img_list = dataset._get_imglist(split)
        samples = dataset._load_samples(split)

        with open(osp.join(list_folder, split + '.txt'), 'w') as f:
            temp = [osp.splitext(osp.basename(x))[0]+'\n' for x in img_list]
            f.writelines(temp)

        print('copy {} images....'.format(split))
        with concurrent.futures.ThreadPoolExecutor() as exector:
            exector.map(_copy, img_list, [image_dir]*len(img_list))

        print('generate {} masks...'.format(split))
        for sample in tqdm(samples):
            density_mask = _generate_mask(sample, args.mask_size)
            basename = osp.basename(sample['image'])
            maskname = osp.join(mask_dir, osp.splitext(basename)[0]+'.hdf5')
            with h5py.File(maskname, 'w') as hf:
                hf['label'] = density_mask

            if args.show:
                img = cv2.imread(sample['image'])
                show_image(img, sample['bboxes'], density_mask)

        print('copy {} box annos...'.format(split))
        anno_list = dataset._get_annolist(split)
        with concurrent.futures.ThreadPoolExecutor() as exector:
            exector.map(_copy, anno_list, [annotation_dir]*len(anno_list))

        print('done.')
