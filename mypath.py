class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if 'hkb' in dataset:
            return '/home/twsf/work/CRGNet/data/HKB'

        elif 'visdrone' in dataset:
            return '/home/twsf/data/visdrone/VisDrone2018-DET-val'

        elif 'coco' in dataset:
            return '/home/twsf/work/CRGNet/data/COCO'

        elif 'voc' in dataset:
            return 'G:/CV/Reading/CRGNet/data/voc'

        elif 'dota' in dataset:
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