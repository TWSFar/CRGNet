import torch
import torch.nn as nn
# import sys
# import os.path as osp
# sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from .necks import ASPP, SELayer, BasicRFB, Inception
from .backbones import build_backbone


class CRG2Net(nn.Module):
    def __init__(self, opt):
        super(CRG2Net, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(opt.backbone, opt.output_stride, BatchNorm)
        # self.aspp = ASPP(opt.backbone,
        #                  opt.output_stride,
        #                  self.backbone.low_outc,
        #                  BatchNorm)
        self.inception = Inception(self.backbone.low_outc)
        # self.link_conv = nn.Sequential(nn.Conv2d(
        #     64, 64, kernel_size=1, stride=1, padding=0, bias=False))
        # self.rfb = BasicRFB(self.backbone.low_outc, 64)
        self.region = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    SELayer(64),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 2, kernel_size=1, stride=1),
                                    nn.Softmax())

        self.density = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     SELayer(64),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 1, kernel_size=1, stride=1))

        self._init_weight()
        if opt.freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        # low_level_feat = self.rfb(low_level_feat)
        # low_level_feat = self.link_conv(low_level_feat)
        # low_level_feat = self.aspp(low_level_feat)
        low_level_feat = self.inception(low_level_feat)
        # x = torch.cat((x, low_level_feat), dim=1)
        region = self.region(low_level_feat)
        density = self.density(low_level_feat)
        return region, density

    def _init_weight(self):
        for module in [self.density, self.region]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def freeze_bn(self):
        for m in self.modules():
            m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d)  or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.link_conv, self.density, self.region]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = CRG2Net(backbone='mobilenetv2', output_stride=16)
    model.eval()
    input = torch.rand(5, 3, 640, 480)
    output = model(input)
    pass
