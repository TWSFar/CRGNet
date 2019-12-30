import os
import fire
import time
import collections
import numpy as np
from tqdm import tqdm

# from models_demo import model_demo
from configs.visdrone_deeplabv3 import opt

from models import DeepLab
# from models import CSRNet
from models.functions import Evaluator, LR_Scheduler
from models.losses import build_loss
from dataloaders import make_data_loader

from utils import (Saver, Timer, TensorboardSummary,
                   calculate_weigths_labels)

import torch

import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Trainer(object):
    def __init__(self, mode):
        torch.cuda.manual_seed(opt.seed)
        # Define Saver
        self.saver = Saver(opt, mode)

        # visualize
        if opt.visualize:
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()

        # Dataset dataloader
        self.train_dataset, self.train_loader = make_data_loader(opt, train=True)  # train
        self.nbatch_train = len(self.train_loader)
        self.nclass = self.train_dataset.nclass
        self.val_dataset, self.val_loader = make_data_loader(opt.test_dir, train=False)  # val
        self.nbatch_val = len(self.val_loader)

        # model
        if opt.sync_bn is None and len(opt.gpu_id) > 1:
            opt.sync_bn = True
        else:
            opt.sync_bn = False
        model = DeepLab(opt, self.nclass)
        self.model = model.to(opt.device)

        # Define Optimizer
        train_params = [{'params': model.get_1x_lr_params(), 'lr': opt.lr},
                        {'params': model.get_10x_lr_params(), 'lr': opt.lr * 10}]
        self.optimizer = torch.optim.SGD(train_params, momentum=opt.momentum,
                                         weight_decay=opt.decay)

        # Loss
        if opt.use_balanced_weights:
            calculate_weigths_labels(opt.dataset,
                                     self.train_loader,
                                     self.nclass,)
        self.loss = build_loss(opt.loss)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(opt.lr_scheduler, opt.lr,
                                      opt.epochs, len(self.train_loader), 140)

        # Resuming Checkpoint
        self.best_pred = 0.0
        self.start_epoch = opt.start_epoch
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
            print("Using multiple gpu")
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=opt.gpu_id)

        self.loss_hist = collections.deque(maxlen=500)
        self.timer = Timer(opt.epochs, len(self.trian_loader), self.num_bt_val)
        self.step_time = collections.deque(maxlen=opt.print_freq)

    def train(self, epoch):
        self.model.train()
        if len(opt.gpu_id) > 1:
            self.model.module.freeze_bn()
        else:
            self.model.freeze_bn()
        epoch_loss = []
        for iter_num, sample in enumerate(self.train_loader):
            # if iter_num > 3: break
            try:
                temp_time = time.time()
                imgs = sample["image"].to(opt.device)
                labels = sample["label"].type(torch.FloatTensor).to(opt.device)

                output = self.model(imgs)

                loss = self.loss(output, labels.unsqueeze(1))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                loss.backward()
                self.loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler(self.optimizer, iter_num, epoch, self.best_pred)

                # Visualize
                global_step = iter_num + self.nbatch_train * epoch + 1
                self.writer.add_scalar('train/loss', loss.cpu().item(), global_step)
                if global_step % opt.plot_every == 0:
                    self.summary.visualize_image(self.writer,
                                                 opt.dataset,
                                                 imgs,
                                                 labels,
                                                 output,
                                                 global_step)

                batch_time = time.time() - temp_time
                eta = self.timer.eta(global_step, batch_time)
                if global_step % opt.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'lr: (1x:{}, 10x:{}),\t'
                          'eta: {}, time: {:1.3f},\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          .format(
                            epoch, iter_num+1, self.nbatch_train,
                            self.optimizer.param_groups[0]['lr'],
                            self.optimizer.param_groups[1]['lr'],
                            eta, np.sum(self.step_time),
                            loss=np.mean(self.loss_hist)))
                del loss

            except Exception as e:
                print(e)
                continue

    def validate(self, epoch):
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



def train(**kwargs):
    opt._parse(kwargs)
    trainer = Trainer()

    print('Num training images: {}'.format(len(trainer.train_dataset)))

    for epoch in range(opt.start_epoch, opt.epochs):
        # train
        trainer.train(epoch, 'train')

        # val
        val_time = time.time()
        mae = trainer.validate(epoch, 'val')
        trainer.timer.set_val_eta(epoch, time.time() - val_time)

        is_best = mae < trainer.best_pred
        trainer.best_pred = min(mae, trainer.best_pred)
        print(' * best MAE {mae:.3f}'.format(mae=trainer.best_pred))
        if (epoch % 20 == 0 and epoch != 0) or is_best:
            trainer.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': trainer.model.module.state_dict() if len(opt.gpu_id) > 1
                else trainer.model.state_dict(),
                'best_pred': trainer.best_pred,
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best)

    # cache result
    print("Backup result...")
    trainer.saver.backup_result()
    print("Done!")


if __name__ == '__main__':
    train()
    # fire.Fire()
