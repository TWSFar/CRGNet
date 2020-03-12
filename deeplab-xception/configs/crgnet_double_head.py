import os
from pprint import pprint
from utils import select_device
user_dir = os.path.expanduser('~')


class Config:
    # data
    dataset = "visdrone"
    root_dir = user_dir + "/data/Visdrone/density_mask"
    test_dir = ""
    input_size = (640, 480)
    mean = [0.382, 0.383, 0.367]
    std = [0.164, 0.156, 0.164]
    num_classes = 5
    resume = False
    pre = "/home/twsf/work/CRGNet/run/visdrone/20200227_235050_train/model_best.pth.tar"

    # model
    backbone = 'mobilenetv2'
    output_stride = 16
    sync_bn = False

    # train
    batch_size = 16  # assert bs > 2
    epochs = 200
    freeze_bn = False

    loss_region = dict(
        type="CrossEntropyLoss",
        ignore_index=-1,
        weight=None
    )

    loss_density = dict(
        type="CrossEntropyLoss",
        ignore_index=-1,
        weight=[1, 2]
    )

    # optimizer
    use_balanced_weights = True
    lr_scheduler = 'poly'  # choices = 'poly', 'step', 'cos'
    lr = 0.0005
    momentum = 0.9
    decay = 5e-4
    steps = [0.8, 0.9]
    gamma = 0.3
    workers = 1

    # eval
    # parameters
    region_thd = 0.5

    # visual
    print_freq = 50
    plot_every = 100  # every n batch plot
    saver_freq = 1

    seed = 1

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        self.device, self.gpu_id = select_device()

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
