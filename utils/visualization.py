import os
import cv2
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
# from dataloaders.utils import decode_seg_map_sequence


def create_vis_plot(vis, X_, Y_, title_, legend_):
    return vis.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, len(legend_))).cpu(),
        opts=dict(
            xlabel=X_,
            ylabel=Y_,
            title=title_,
            legend=legend_
        )
    )


def update_vis_plot(vis, item, loss, window, update_type):
    if item == 0:
        update_type = True

    vis.line(
        X=torch.ones((1, len(loss))).cpu() * item,
        Y=torch.Tensor(loss).unsqueeze(0).cpu(),
        win=window,
        update=update_type
    )


def plot_result(imgs, target, output):
    import matplotlib.pyplot as plt
    show_img = imgs[0].cpu().permute(1, 2, 0).data.numpy()
    show_img = (show_img - show_img.min())
    show_img = show_img / show_img.max()

    show_target = target[0].cpu().permute(1, 2, 0).data.numpy()
    show_target = show_target.repeat(3, 2) / show_target.max()

    show_output = output[0].cpu().permute(1, 2, 0).data.numpy()
    show_output = (show_output - show_output.min())
    show_output = show_output / show_output.max()
    show_output = show_output.repeat(3, 2)

    plt.subplot(1, 3, 1)
    plt.imshow(show_img)
    plt.subplot(1, 3, 2)
    plt.imshow(show_target)
    plt.subplot(1, 3, 3)
    plt.imshow(show_output)
    plt.show()
    # cv2.imshow("target", show_target)
    # cv2.waitKey(0)


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        # plot_result(image, target, output)
        # show_output = output[0].cpu().permute(1, 2, 0).data.numpy()
        # if show_output.max() > 1:
        #     print("..................................")
        #     print("global_step {}".format(global_step))

        # images
        grid_image = make_grid(image[:2].clone().cpu().data, nrow=3, normalize=True)
        writer.add_image('Image', grid_image, global_step)

        # output
        grid_output = make_grid(output[:2].clone().cpu(), nrow=3, normalize=True)
        writer.add_image('Predicted density', grid_output, global_step)

        # target
        grid_target = make_grid(target[:2].clone().cpu(), nrow=3, normalize=True)
        writer.add_image('Groundtruth density', grid_target, global_step)
