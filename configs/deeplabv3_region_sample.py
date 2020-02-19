import os
import time
from pprint import pprint
from utils import select_device
user_dir = os.path.expanduser('~')


class Config:
    # data
    dataset = "visdrone"
    # root_dir = user_dir + "/work/CRGNet/data/Visdrone_Region"
    root_dir = "E:\CV\\code\\CRGNet\\data\\Visdrone_Region"
    input_size = (640, 480)
    mean = [0.382, 0.383, 0.367]
    std = [0.164, 0.156, 0.164]
    resume = False
    pre = None

    # model
    backbone = 'mobilenetv2'
    output_stride = 16
    output_channels = 2
    sync_bn = False

    # train
    batch_size = 2  # assert bs > 2
    start_epoch = 0
    epochs = 3
    freeze_bn = False

    # loss
    loss = dict(
        type="CrossEntropyLoss",
        ignore_index=255,
        weight=[1, 2]
    )

    # param for optimizer
    use_balanced_weights = False
    lr_scheduler = 'poly'  # choices = 'poly', 'step', 'cos'
    lr = 0.0005
    momentum = 0.9
    decay = 5e-4
    steps = [0.8, 0.9]
    scales = 0.3
    workers = 16

    # eval
    # parameters
    pst_thd = 0.05

    # visual
    print_freq = 1
    plot_every = 1  # every n batch plot
    saver_freq = 1

    seed = time.time()

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
