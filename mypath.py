class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'hkb':
            return 'G:/CV/Reading/CRGNet/data/hkb'

        elif dataset == 'visdrone':
            return 'G:/CV/Reading/CRGNet/data/visdrone'

        elif dataset == 'coco':
            return 'G:/CV/Reading/CRGNet/data/coco'

        elif dataset == 'voc':
            return 'G:/CV/Reading/CRGNet/data/voc'

        elif dataset == 'dota':
            return 'G:/CV/Reading/CRGNet/data/dota'

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

    @staticmethod
    def weights_root_dir(backbone):
        if backbone == 'resnet101':
            return "/home/twsf/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth"
        else:
            print('weights {} not available.'.format(backbone))