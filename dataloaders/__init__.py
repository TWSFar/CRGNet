from .datasets import VisDroneRegion
from torch.utils.data import DataLoader


def make_data_loader(opt, mode="train"):

    if opt.dataset in ['visdrone', 'VisDrone', 'Visdrone']:
        dataset = VisDroneRegion(opt, mode="train")
        dataloader = DataLoader(dataset,
                                batch_size=opt.batch_size,
                                num_workers=opt.workers,
                                shuffle=True if mode == "train" else False,
                                pin_memory=True)

        return dataset, dataloader

    else:
        raise NotImplementedError
