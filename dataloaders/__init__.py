from dataloaders.datasets import visdrone
from torch.utils.data import DataLoader


def make_data_loader(opt, train=True):
    if opt.dataset in ['visdrone', 'VisDrone', 'Visdrone']:
        dataset = visdrone.VisDroneSegmentation(opt, train=train)
        dataloader = DataLoader(dataset,
                                batch_size=opt.batch_size,
                                num_workers=opt.workers,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)

        return dataset, dataloader

    else:
        raise NotImplementedError
