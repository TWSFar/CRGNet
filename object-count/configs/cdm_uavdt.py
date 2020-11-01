import os
from pprint import pprint
from utils import select_device
user_dir = os.path.expanduser('~')


class Config:
    # data
    dataset = "UAVDT"
    img_type = 'jpg'
    root_dir = user_dir + "/data/UAVDT/density_mask"
    num_classes = 1
    input_size = (640, 480)
    norm_cfg = dict(mean=[0.368, 0.378, 0.371], std=[0.17, 0.16, 0.17], para=10)
    resume = False
    pre = ""

    # model
    backbone = 'mobilenetv2'
    output_stride = 16
    sync_bn = False

    # train
    batch_size = 32  # assert bs > 2
    epochs = 30
    freeze_bn = False

    loss_region = dict(
        type="CrossEntropyLoss",
        ignore_index=-1,
        weight=[1, 100]
    )

    loss_density = dict(
        type="MSELoss",
        reduction="mean"
    )

    # optimizer
    use_balanced_weights = False
    lr = 0.01
    momentum = 0.9
    decay = 5e-4
    # decay = 1e-5
    steps = [0.8, 0.9]
    gamma = 0.3
    workers = 16

    # visual
    print_freq = 50
    plot_every = 2000  # every n batch plot

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
