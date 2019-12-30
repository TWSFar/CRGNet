from .focal_loss import FocalLoss
from .cross_entropy_loss import CrossEntropyLoss


def build_loss(args):
    obj_type = args.pop('type')

    if obj_type == "FocalLoss":
        return FocalLoss(**args)

    elif obj_type == "CrossEntropyLoss":
        return CrossEntropyLoss(**args)

    else:
        raise NotImplementedError
