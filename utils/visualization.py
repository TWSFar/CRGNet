import os
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        # images
        grid_image = make_grid(image[:2].clone().cpu().data, nrow=3, normalize=True)
        writer.add_image('Image', grid_image, global_step)

        # output
        grid_output = make_grid(output[:2].clone().cpu(), nrow=3, normalize=True)
        writer.add_image('Predicted label', grid_output, global_step)

        # target
        grid_target = make_grid(target[:2].clone().cpu(), nrow=3, normalize=True)
        writer.add_image('Groundtruth label', grid_target, global_step)
