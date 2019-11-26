import os
import time
import os.path as osp
import shutil
import torch
import glob


class Saver(object):

    def __init__(self, opt):
        self.opt = opt
        self.directory = osp.join('run', opt.dataset)
        experiment_name = time.strftime("%Y%d%m_%H%M%S")
        self.experiment_dir = osp.join(self.directory, experiment_name)
        self.logfile = osp.join(self.experiment_dir, 'experiment.log')
        if not osp.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        for key, val in self.opt._state_dict().items():
            line = key + ': ' + str(val)
            self.save_experiment_log(line)

    def save_checkpoint(self, state, is_best, filename='checkpoint.path.tar'):
        ''' Saver checkpoint to disk '''
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(osp.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write('epoch {}: {}'.format(state['epoch'] - 1, best_pred))
            shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))

    def save_experiment_log(self, line):
        with open(self.logfile, 'a') as f:
            f.write(line + '\n')

    def save_coco_eval_result(self, epoch, stats):
        with open(os.path.join(self.experiment_dir, 'result.txt'), 'a') as f:
            f.writelines(
                "[epoch: {}, AP@50:95: {:.3%}, AP@50: {:.3%}]\n".format(
                    epoch, stats[0], stats[1]))
