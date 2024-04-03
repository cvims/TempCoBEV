import torch
import torch.nn as nn

from einops import rearrange

from tqdm import tqdm
from opencood.loss import BaseLoss
from opencood.tools.runner_temporal.logger import BaseLogger

from opencood.loss import BaseLoss

import wandb


class IoULoss(BaseLoss):
    def __init__(self, args):
        super(IoULoss, self).__init__()

        self.target = args['target']

        self.sigmoid = nn.Sigmoid()


    def _forward(self, output_dict, gt_dict) -> torch.Tensor:
        """
        Perform loss function on the prediction.

        Parameters
        ----------
        output_dict : dict
            The dictionary contains the prediction.

        gt_dict : dict
            The dictionary contains the groundtruth.

        Returns
        -------
        Loss dictionary.
        """

        # static_pred = output_dict['static_seg']
        dynamic_pred = output_dict['dynamic_seg']

        # static_loss = torch.tensor(0, device=static_pred.device)
        dynamic_loss = torch.tensor(0, device=dynamic_pred.device)

        # during training, we only need to compute the ego vehicle's gt loss
        # static_gt = gt_dict['gt_static']
        dynamic_gt = gt_dict['gt_dynamic']
        # static_gt = rearrange(static_gt, 'b l h w -> (b l) h w')
        dynamic_gt = rearrange(dynamic_gt, 'b l h w -> (b l) h w')

        if self.target == 'dynamic':
            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            # only second channel (dynamic vehicle dim)
            dynamic_loss = self.sigmoid(dynamic_pred[:, 1, :, :])
            # Intersection
            intersection = torch.sum(dynamic_loss * dynamic_gt)
            # Union
            union = torch.sum(dynamic_loss) + torch.sum(dynamic_gt) - intersection
            # IoU
            dynamic_loss = 1 - (intersection + 1) / (union + 1)

            # mean over batch and time
            dynamic_loss = torch.mean(dynamic_loss)

        elif self.target == 'static':
            raise NotImplementedError
        else:
            raise NotImplementedError

        total_loss = dynamic_loss

        return dict(
            total_loss=total_loss,
            dynamic_loss=dynamic_loss
        )

    def logging(self, epoch, batch_id, batch_len, logger: BaseLogger, pbar: tqdm = None) -> None:
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training.
        """
        total_loss = self.loss_dict['total_loss']
        # static_loss = self.loss_dict['static_loss']
        dynamic_loss = self.loss_dict['dynamic_loss']

        if pbar is None:
            # print("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
            #     " || Dynamic Loss: %.4f" % (
            #         epoch, batch_id + 1, batch_len,
            #         total_loss.item(), static_loss.item(), dynamic_loss.item()))
            print("[epoch %d][%d/%d], || Loss: %.4f || Dynamic Loss: %.4f" % (
                epoch, batch_id + 1, batch_len,
                total_loss.item(), dynamic_loss.item()))
        else:
            # pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
            #       " || Dynamic Loss: %.4f" % (
            #           epoch, batch_id + 1, batch_len,
            #           total_loss.item(), static_loss.item(), dynamic_loss.item()))
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Dynamic Loss: %.4f" % (
                epoch, batch_id + 1, batch_len,
                total_loss.item(), dynamic_loss.item()))

        if logger:
            logger.log_metrics(
                metrics=dict(
                    total_loss=total_loss,
                    # static_loss=static_loss,
                    dynamic_loss=dynamic_loss
                ),
                epoch=epoch, iteration=batch_id, batch_len=batch_len
            )
    
    def logging_wandb(self, epoch, batch_id, batch_len, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        # static_loss = self.loss_dict['static_loss']
        dynamic_loss = self.loss_dict['dynamic_loss']

        if pbar is None:
            # print("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
            #     " || Dynamic Loss: %.4f" % (
            #         epoch, batch_id + 1, batch_len,
            #         total_loss.item(), static_loss.item(), dynamic_loss.item()))
            print("[epoch %d][%d/%d], || Loss: %.4f || Dynamic Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), dynamic_loss.item()))
        else:
            # pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
            #       " || Dynamic Loss: %.4f" % (
            #           epoch, batch_id + 1, batch_len,
            #           total_loss.item(), static_loss.item(), dynamic_loss.item()))
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Dynamic Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), dynamic_loss.item()))


        wandb.log({
            'train_loss': total_loss.item(),
            # 'train_static_loss': static_loss.item(),
            'train_dynamic_loss': dynamic_loss.item()},
            step=epoch*batch_len + batch_id
        )
