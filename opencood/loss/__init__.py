from abc import abstractmethod
import torch
import torch.nn as nn
from opencood.tools.runner_temporal.logger import BaseLogger
from tqdm import tqdm


class BaseLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_dict = {}
    
    @abstractmethod
    def get_current_loss_dict(self) -> dict:
        raise NotImplementedError
    
    @abstractmethod
    def _forward(self, output_dict, gt_dict) -> torch.Tensor:
        raise NotImplementedError
    
    def forward(self, output_dict, gt_dict) -> torch.Tensor:
        self.loss_dict = self._forward(output_dict, gt_dict)

        return self.loss_dict['total_loss']

    @abstractmethod
    def logging(self, epoch: int, iteration: int, batch_len: int, logger: BaseLogger, pbar: tqdm = None) -> dict:
        raise NotImplementedError
