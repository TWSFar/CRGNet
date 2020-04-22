import torch
import torch.nn as nn


class test1(nn.Module):
    def __init__(self):
        super(test1, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, bias=False)

    def forward(self, x):
        return self.conv1(x)


class test2(nn.Module):
    def __init__(self):
        super(test2, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, bias=False)

    def forward(self, x):
        return self.conv1(x)


if __name__=='__main__':
    device = torch.device('cuda:0')
    model1 = nn.DataParallel(test1().to(device))
    model2 = nn.DataParallel(test2().to(device))
    input = torch.rand(2, 3, 2, 2).to(device)
    for i in range(100):
        m = model1(input)
        t = model2(m)

    pass