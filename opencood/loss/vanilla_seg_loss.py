import torch
import torch.nn as nn

from einops import rearrange

from tqdm import tqdm
from opencood.loss import BaseLoss
from opencood.tools.runner_temporal.logger import BaseLogger

from opencood.loss import BaseLoss

import wandb


class VanillaSegLoss(BaseLoss):
    def __init__(self, args):
        super(VanillaSegLoss, self).__init__()

        self.d_weights = args['d_weights']
        self.s_weights = args['s_weights']
        self.l_weights = 50 if 'l_weights' not in args else args['l_weights']

        # self.d_coe = args['d_coe']
        # self.s_coe = args['s_coe']
        self.d_coe = torch.as_tensor(args['d_coe']).cuda()
        self.s_coe = torch.as_tensor(args['s_coe']).cuda()
        self.target = args['target']

        # self.loss_func_static = \
        #     nn.CrossEntropyLoss(
        #         weight=torch.Tensor([1., self.s_weights, self.l_weights]).cuda())
        self.loss_func_static = \
            nn.BCEWithLogitsLoss(
                weight=torch.Tensor([1., self.s_weights, self.l_weights]).cuda())
        self.loss_func_dynamic = \
            nn.CrossEntropyLoss(
                weight=torch.Tensor([1., self.d_weights]).cuda())

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

        static_loss = torch.tensor(0)
        dynamic_loss = torch.tensor(0)

        # during training, we only need to compute the ego vehicle's gt loss
        if self.target == 'dynamic':
            device = output_dict['dynamic_seg'].device
            dynamic_pred = output_dict['dynamic_seg']
            dynamic_gt = gt_dict['gt_dynamic']
            dynamic_gt = rearrange(dynamic_gt, 'b l h w -> (b l) h w')

            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)

        elif self.target == 'static':
            device = output_dict['static_seg'].device
            static_pred = output_dict['static_seg']
            static_gt = gt_dict['gt_static']
            static_gt = rearrange(static_gt, 'b l h w -> (b l) h w')

            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            static_loss = self.loss_func_static(static_pred, static_gt)
        else:
            device = output_dict['dynamic_seg'].device
            dynamic_pred = output_dict['dynamic_seg']
            dynamic_gt = gt_dict['gt_dynamic']
            dynamic_gt = rearrange(dynamic_gt, 'b l h w -> (b l) h w')

            static_pred = output_dict['static_seg']
            static_gt = gt_dict['gt_static']
            static_gt = rearrange(static_gt, 'b l h w -> (b l) h w')

            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)
            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            static_loss = self.loss_func_static(static_pred, static_gt)

        # to device
        static_loss = static_loss.to(device)
        dynamic_loss = dynamic_loss.to(device)

        total_loss = self.s_coe * static_loss + self.d_coe * dynamic_loss

        return dict(
            total_loss=total_loss,
            static_loss=static_loss,
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
        static_loss = self.loss_dict['static_loss']
        dynamic_loss = self.loss_dict['dynamic_loss']

        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
                " || Dynamic Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), static_loss.item(), dynamic_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
                  " || Dynamic Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), static_loss.item(), dynamic_loss.item()))

        if logger:
            logger.log_metrics(
                metrics=dict(
                    total_loss=total_loss,
                    static_loss=static_loss,
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
        static_loss = self.loss_dict['static_loss']
        dynamic_loss = self.loss_dict['dynamic_loss']

        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
                " || Dynamic Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), static_loss.item(), dynamic_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
                  " || Dynamic Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), static_loss.item(), dynamic_loss.item()))


        wandb.log({
            'train_loss': total_loss.item(),
            'train_static_loss': static_loss.item(),
            'train_dynamic_loss': dynamic_loss.item()},
            step=epoch*batch_len + batch_id
        )