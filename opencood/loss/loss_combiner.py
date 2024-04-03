import torch
import torch.nn as nn

from einops import rearrange

from tqdm import tqdm
from opencood.loss import BaseLoss
from opencood.tools.runner_temporal.logger import BaseLogger

from opencood.loss import BaseLoss
from opencood.tools.training_temporal.train_utils import create_loss


class LossCombiner(BaseLoss):
    def __init__(self, args):
        super(LossCombiner, self).__init__()

        self.loss_weights = dict()
        self.losses = nn.ModuleDict()
        for key in args:
            _loss_hypes = dict(
                loss=args[key]
            )
            self.losses[key] = create_loss(_loss_hypes)
            if 'weight' in args[key]:
                self.loss_weights[key] = args[key]['weight']
            else:
                self.loss_weights[key] = 1.0

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

        loss = torch.tensor(0, device=output_dict['dynamic_seg'].device)
        separate_losses = dict()
        for key in self.losses:
            _loss = self.losses[key](output_dict, gt_dict)
            loss = _loss + _loss * self.loss_weights[key]
            separate_losses[key] = _loss

        return dict(
            total_loss=loss,
            **separate_losses
        )

    def logging(self, epoch, batch_id, batch_len, logger: BaseLogger, pbar: tqdm = None) -> None:
        loss_str = ' || '.join(f'{key}: {value.item():.4f}' for key, value in self.loss_dict.items())
        if pbar is None:
            # print all logs inside self.loss_dict dictionary where the keys are strings and the values are floats
            # fix them to 4 decimal places
            print(f"[epoch %d][%d/%d] {loss_str}" % (epoch, batch_id + 1, batch_len))
        else:
            pbar.set_description(f"[epoch %d][%d/%d] {loss_str}" % (epoch, batch_id + 1, batch_len))

        if logger:
            logger.log_metrics(
                metrics=self.loss_dict,
                epoch=epoch, iteration=batch_id, batch_len=batch_len
            )
