from dataloaders.datasets import visdrone_region, underwater
from torch.utils.data import DataLoader


def make_data_loader(opt, mode="train"):

    if opt.dataset in ['visdrone', 'VisDrone', 'Visdrone']:
        dataset = visdrone_region.VisDroneRegion(opt, mode)
        dataloader = DataLoader(dataset,
                                batch_size=opt.batch_size,
                                num_workers=opt.workers,
                                shuffle=True if mode == "train" else False,
                                pin_memory=True)

        return dataset, dataloader

    elif opt.dataset.lower() == 'underwater':
        dataset = underwater.UnderWater(opt, mode)
        dataloader = DataLoader(dataset,
                                batch_size=opt.batch_size,
                                num_workers=opt.workers,
                                shuffle=True if mode == "train" else False,
                                pin_memory=True)

        return dataset, dataloader

    else:
        raise NotImplementedError
