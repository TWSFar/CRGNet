import os
import fire
import time
import collections
import numpy as np
from tqdm import tqdm

# from models_demo import model_demo
from config.visdrone_deeplabv3 import opt
from dataloaders import make_data_loader
from models import csrnet, deeplab
from models.functions import loss, metrics
from utils.saver import Saver
from utils.timer import Timer
from utils.visualization import TensorboardSummary
from utils.lr_scheduler import LR_Scheduler

import torch
import torch.nn as nn
import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Trainer(object):
    def __init__(self):
        torch.cuda.manual_seed(opt.seed)
        self.best_pred = 0.0
        self.start_epoch = opt.start_epoch

        # Define Saver
        self.saver = Saver(opt)

        # visualize
        if opt.visualize:
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()

        # Dataset dataloader
        self.train_dataset, self.train_loader = make_data_loader(opt, train=True)
        self.val_dataset, self.val_loader = make_data_loader(opt.test_dir, train=False)
        self.nclass = self.train_dataset.nclass

        # model
        model = deeplab.DeepLab(backbone=opt.backbone,
                                num_classes=self.train_dataset.nclass,
                                sync_bn=opt.sync_bn)
        self.model = model.to(opt.device)

        # Define Optimizer
        train_params = [{'params': model.get_1x_lr_params(), 'lr': opt.lr},
                        {'params': model.get_10x_lr_params(), 'lr': opt.lr * 10}]
        self.optimizer = torch.optim.SGD(train_params, momentum=opt.momentum,
                                         weight_decay=opt.decay)

        # Loss
        weight = torch.tensor([1, 2]).float()
        self.criterion = loss.SegmentationLosses(weight=weight, cuda=len(opt.gpu_id)>0).build_loss(mode=opt.loss_type)

        # Define Evaluator
        self.evaluator = metrics.Evaluator(self.nclass)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(opt.lr_scheduler, opt.lr,
                                      opt.epochs, len(self.train_loader), 140)

        if opt.resume:
            if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                opt.start_epoch = checkpoint['epoch']
                self.best_pred = checkpoint['best_pred']
                self.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(opt.pre, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(opt.pre))

        if len(opt.gpu_id) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=opt.gpu_id)

        self.timer = Timer(opt.epochs, len(self.trian_loader), self.num_bt_val)
        self.step_time = collections.deque(maxlen=opt.print_freq)

    def train(self, epoch):
        self.model.train()
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        num_img_tr = len(self.train_loader)
        print('epoch %d, processed %d samples, lr %.10f' %
              (epoch, epoch * len(self.train_loader.dataset), opt.lr))

        end = time.time()
        for i, (img, target, _)in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            img = img.to(opt.device)
            target = target.type(torch.FloatTensor).unsqueeze(1).to(opt.device)

            output = self.model(img)

            loss = self.criterion(output, target)
            losses.update(loss.item(), img.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            # visualize
            # if opt.visualize:
            #     update_vis_plot(vis, i, [loss.cpu().tolist()], batch_plot, 'append')
            global_step = i + num_img_tr * epoch
            self.writer.add_scalar('train/total_loss_epoch', loss.cpu().item(), global_step)
            if (i + 1) % opt.plot_every == 0:
                self.summary.visualize_image(self.writer, opt.dataset, img, target, output, global_step)

            if i % opt.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      .format(
                        epoch, i, len(self.train_loader),
                        batch_time=batch_time,
                        data_time=data_time, loss=losses))

    def validate(self, epoch):
        maes = AverageMeter()
        mses = AverageMeter()
        self.model.eval()

        for i, (img, target, scale) in enumerate(tqdm(self.test_loader)):
            img = img.to(opt.device)
            output = self.model(img)
            output = output.data.cpu().numpy()
            target = target.data.numpy()
            for i_img in range(output.shape[0]):
                pred_count = np.sum(output[i_img]) / opt.log_para
                gt_count = np.sum(target[i_img]) / opt.log_para
                maes.update(abs(gt_count - pred_count))
                mses.update((gt_count - pred_count) ** 2)

        mae = maes.avg
        mse = np.sqrt(mses.avg)

        # visualize
        # if opt.visualize:
        #     update_vis_plot(vis, epoch, [mae], val_plot, 'append')
        self.writer.add_scalar('val/mae', mae, epoch)
        self.writer.add_scalar('val/mse', mse, epoch)
        print(' * MAE {mae:.3f} | * MSE {mse:.3f}'.format(mae=mae, mse=mse))

        return mae


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(**kwargs):
    opt._parse(kwargs)
    trainer = Trainer()
    for epoch in range(opt.start_epoch, opt.epochs):
        # train
        trainer.train(epoch)

        # val
        mae = trainer.validate(epoch)

        is_best = mae < trainer.best_pred
        trainer.best_pred = min(mae, trainer.best_pred)
        print(' * best MAE {mae:.3f}'.format(mae=trainer.best_pred))
        if (epoch % 20 == 0 and epoch != 0) or is_best:
            trainer.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': trainer.model.module.state_dict() if opt.use_mulgpu
                else trainer.model.state_dict(),
                'best_pred': trainer.best_pred,
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best)


if __name__ == '__main__':
    train()
    fire.Fire()
