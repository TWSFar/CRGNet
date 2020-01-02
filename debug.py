import torch
import torch.nn as nn


class DeepLab(nn.Module):
    def __init__(self, num_classes=21):
        super(DeepLab, self).__init__()

        self.last_conv = nn.Sequential(nn.BatchNorm2d(128))

        self.freeze_bn()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

if __name__ == "__main__":
    model = DeepLab()
    model.eval()
    model.train()
    model.eval()
    model.train()
    pass
