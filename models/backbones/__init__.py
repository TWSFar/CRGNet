from models.backbones import resnet, xception, mobilenetv2, mobilenetv3


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'mobilenetv2':
        return mobilenetv2.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'mobilenetv3':
        return mobilenetv3.MobileNetV3_Small()
    else:
        raise NotImplementedError
