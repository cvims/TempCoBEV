from abc import abstractmethod
import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, args):
        super(FusionModel, self).__init__()

    @abstractmethod
    def until_fusion(
        self,
        scenario,
        **kwargs
    ):
        raise NotImplementedError

    @abstractmethod
    def from_fusion(
        self,
        x: torch.Tensor,
        **kwargs
    ):
        raise NotImplementedError


class TemporalFusionModel(FusionModel):
    @abstractmethod
    def temporal_fusion(
        self,
        *fusion_inputs,
        history_cooperative_bev_embeddings,
        **kwargs
    ):
        raise NotImplementedError
    
    
