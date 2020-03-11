import h5py
import numpy as np
import torch
import torch.nn as nn


class CONV(nn.Module):
    def __init__(self):
        super(CONV, self).__init__()
        self.conv1 = nn.Conv2d(192, 64, 1, bias=False)
        self.s = nn.Sequential(
            nn.Conv2d(192, 64, 1, bias=False),
            nn.Conv2d(64, 64, 1, bias=False)
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.conv1 = CONV()
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

if __name__=='__main__':
    model = test()


m = np.zeros(20)

def fz(x, y):
    return x*y


a = np.arange(1, 3)
b = np.arange(4, 6)

z = np.zeros(3, 3)


pass