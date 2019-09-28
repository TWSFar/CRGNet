class Path(object):
    def init()
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'hkb':
            return 'G:/CV/Reading/ClusterRegionGenerationNetwork/data/hkb'

        elif dataset == 'visdrone':
            return 'G:/CV/Reading/ClusterRegionGenerationNetwork/data/visdrone'

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

    def weights_root_dir(backbone):
        if backbone == 'resnet101':
            return "/home/twsf/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth"
        else:
            print('weights {} not available.'.format(backbone))