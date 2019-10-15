import time
import torch
import os.path as osp
from pprint import pprint


class Config:
    # data
    dataset = "SHTB"
    root_dir = "/home/twsf/data/Shanghai/part_B_final"
    train_dir = osp.join(root_dir, "train_data")
    test_dir = osp.join(root_dir, "test_data")
    pre = '/home/twsf/work/CSRNet/run/SHTB/model_best.pth.tar'

    # train
    batch_size = 3
    input_size = (768, 576)  # (x, y)
    start_epoch = 0
    epochs = 201
    workers = 8
    mean = [0.452016860247, 0.447249650955, 0.431981861591]
    std = [0.23242045939, 0.224925786257, 0.221840232611]
    log_para = 100.  # density need a factor
    downrate = 8
    gtdownrate = 8

    # param for optimizer
    original_lr = 0.0005
    lr = 0.00005
    momentum = 0.995
    decay = 5*1e-4
    steps = [0.8, 0.9]
    scales = 0.3

    use_mulgpu = False
    gpu_id = [0, 1, 2]
    device = torch.device('cuda:0')
    visualize = True
    resume = False
    print_freq = 10
    plot_every = 10  # every n batch plot
    seed = time.time()

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
