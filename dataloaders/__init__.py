from .datasets import VisDroneRegion, VisDroneDensity
from torch.utils.data import DataLoader


def make_data_loader(opt, train=True):

    if opt.dataset in ['visdrone', 'VisDrone', 'Visdrone']:
        dataset = VisDroneRegion(opt, train=train)
        dataloader = DataLoader(dataset,
                                batch_size=opt.batch_size,
                                num_workers=opt.workers,
                                shuffle=train,
                                pin_memory=True)

        return dataset, dataloader

    if opt.dataset in ['visdrone', 'VisDrone', 'Visdrone']:
        dataset = VisDroneDensity(opt, train=train)
        dataloader = DataLoader(dataset,
                                batch_size=opt.batch_size,
                                num_workers=opt.workers,
                                shuffle=train,
                                pin_memory=True)

        return dataset, dataloader

    else:
        raise NotImplementedError
