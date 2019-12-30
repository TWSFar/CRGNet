import torch
import torch.nn as nn


class CrossEntropyLoss(object):
    def __init__(self, weight=None, ignore_index=255):
        self.ignore_index = ignore_index
        self.weight = weight

    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        device = logit.device
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        reduction='mean').to(device)

        loss = criterion(logit, target.long())

        return loss


if __name__ == "__main__":
    loss = CrossEntropyLoss()
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss(a, b))
