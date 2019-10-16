from pprint import pprint


class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if 'hkb' in dataset:
            return '/home/twsf/work/CRGNet/data/HKB'

        elif 'visdrone' in dataset:
            return '/home/twsf/data/Visdrone'
            # return '/home/twsf/work/CRGNet/data/VisDrone'

        elif 'coco' in dataset:
            return '/home/twsf/work/CRGNet/data/COCO'

        elif 'voc' in dataset:
            return '/home/twsf/work/CRGNet/data/VOC2012'

        elif 'dota' in dataset:
            return 'G:/CV/Reading/CRGNet/data/dota'

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError


class Config:
    # data
    dataset = 'visdrone'
    data_dir = Path.db_root_dir(dataset)
    mode = 'train'
    show = False
    # no use, but 'min' must euqal 'max'
    min_size = 600
    max_size = 600

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        self.data_dir = Path.db_root_dir(self.dataset)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
