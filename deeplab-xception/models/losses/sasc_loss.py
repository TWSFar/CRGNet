"""Spatial Abastraction Loss
arXiv: https://arxiv.org/pdf/1903.00853.pdf (CVPR2019)
"""
import torch
import torch.nn as nn


class SALoss(object):
    def __init__(self, reduction="mean", levels=4):
        self.reduction = reduction
        self.levels = levels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, pred, target):
        loss = []
        device = pred.device
        pred.squeeze_(0)
        criterion = nn.MSELoss(reduction=self.reduction).to(device)
        loss.append(criterion(pred, target))
        for i in range(self.levels-1):
            pred = self.pool(pred)
            target = self.pool(target)
            loss.append(criterion(pred, target))

        return torch.stack(loss).sum()


class SCLoss(object):
    def __call__(self, pred, target):
        pred.squeeze_(1)
        pred_2 = pred.pow(2)
        target_2 = target.pow(2)
        sum_pXt = (pred * target).sum(dim=(1, 2))
        mul_p2Xt2 = pred_2.sum(dim=(1, 2)) * target_2.sum(dim=(1, 2))
        loss = (1 - sum_pXt / (mul_p2Xt2).sqrt())

        return loss.mean()


class SASCLoss(object):
    def __init__(self, reduction="mean", levels=4):
        self.sal = SALoss(reduction, levels)
        self.scl = SCLoss()

    def __call__(self, pred, target):
        saloss = self.sal(pred, target)
        scloss = self.scl(pred, target)

        return saloss + scloss


if __name__ == "__main__":
    # torch.manual_seed(1)
    loss_c = SCLoss()
    loss_a = SALoss()
    a = torch.rand(16, 1, 30, 40)  * 20

    b = torch.rand(16, 30, 40) * 20

    print(loss_a(a, b))
    pass
