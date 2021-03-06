import os
from pprint import pprint
from utils import select_device
user_dir = os.path.expanduser('~')


class Config:
    # data
    dataset = "DOTA"
    img_type = 'png'
    root_dir = user_dir + "/data/DOTA/density_mask"
    num_classes = 1
    input_size = (1024, 1024)
    norm_cfg = dict(mean=[0.51, 0.535, 0.556], std=[0.196, 0.208, 0.246], para=1)
    resume = False
    pre = ""

    # model
    backbone = 'mobilenetv2'
    output_stride = 16
    sync_bn = False

    # train
    batch_size = 8  # assert bs > 2
    epochs = 50
    freeze_bn = False

    loss_region = dict(
        type="CrossEntropyLoss",
        ignore_index=-1,
        weight=None
    )

    loss_density = dict(
        type="MSELoss",
        reduction="mean"
    )

    # optimizer
    use_balanced_weights = True
    lr = 0.01
    momentum = 0.9
    decay = 5e-4
    # decay = 1e-5
    steps = [0.8, 0.9]
    gamma = 0.3
    workers = 16

    # visual
    print_freq = 50
    plot_every = 200  # every n batch plot

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
