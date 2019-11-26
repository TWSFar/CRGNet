import os
import time
import numpy as np
from mypath import Path
from pprint import pprint
from utils.devices import select_device
user_dir = os.path.expanduser('~')


class Config:
    # data
    dataset = "visdrone"
    root_dir = Path.db_root_dir(dataset)
    resume = False
    input_size = (640, 480)
    pre = None

    # model
    backbone = 'mobilenetv2'
    hrnet_cfg = user_dir + '/work/RetinaNet/lib/hrnet_config/hrnet_w48.yaml'

    # train
    batch_size = 24
    start_epoch = 0
    epochs = 200
    workers = 16

    # param for optimizer
    loss_type = 'ce'  # Choices: 'ce' or 'focal'
    lr_scheduler = 'poly'  # choices = 'poly', 'step', 'cos'
    adam = True
    lr = 0.0005
    momentum = 0.9
    decay = 5e-4
    steps = [0.8, 0.9]
    scales = 0.3

    # eval
    # parameters
    pst_thd = 0.05

    # visual
    visualize = True
    print_freq = 10
    plot_every = 50  # every n batch plot
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