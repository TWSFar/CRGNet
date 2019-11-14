from dataloaders.datasets import visdrone
from torch.utils.data import DataLoader


def make_data_loader(opt, train=True):
    if opt.dataset in ['visdrone', 'VisDrone', 'Visdrone']:
        dataset = visdrone.VisDroneSegmentation(opt, train=train)
        dataloader = DataLoader(dataset,
                                num_workers=opt.workers,
                                batch_size=opt.batch_size,
                                collate_fn=dataset.collater)

        return dataset, dataloader

    else:
        raise NotImplementedError
