import torch

from opencood.tools.evaluation_temporal.utils import plot_embeddings


class IdentityEncoder(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


    def forward(self, current_bev_proposal: torch.Tensor, history_bevs: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return current_bev_proposal
