import h5py
import numpy as np
import torch
import torch.nn as nn


class CONV(nn.Module):
    def __init__(self):
        super(CONV, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 1, bias=False)
        self.s = nn.Sequential(
            nn.Conv2d(192, 64, 1, bias=False),
            nn.Conv2d(64, 64, 1, bias=False)
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.conv1(x)

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
    model = CONV()
    input = torch.rand(2, 3, 2, 2)
    
    m = model(input)
    m.sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
    pass