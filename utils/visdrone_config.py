import time
import torch
import numpy as np
from pprint import pprint
from mypath import Path


def select_device(force_cpu=False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    ng = 0
    if not cuda:
        print('Using CPU\n')
    if cuda:
        c = 1024 ** 2
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        for i in range(ng):
            print('Using CUDA device{} _CudaDeviceProperties(name={}, total_memory={}MB'.\
                  format(i, x[i].name, round(x[i].total_memory/c)))
        print('')
    return device, np.arange(0, ng).tolist()


class Config:
    # data
    dataset = 'visdrone'
    data_dir = Path.db_root_dir(dataset)
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

        self.device, self.gpu_id = select_device()
        self.sync_bn = len(self.gpu_id) > 1

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
