from dataloaders.datasets import voc
from torch.utils.data import DataLoader


def make_data_loader(opt, mode="train"):

    dataset = voc.VOC(opt, mode)
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size if mode == "train" else 1,
                            num_workers=opt.workers,
                            shuffle=True if mode == "train" else False,
                            pin_memory=True)

    return dataset, dataloader
