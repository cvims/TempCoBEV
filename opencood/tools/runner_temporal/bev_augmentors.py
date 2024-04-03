import torch
from opencood.tools.runner_temporal import BaseHistoryBEVEmbeddingAugmentor



class CooperativeHistoryBEVFullZeroOutAugmentor(BaseHistoryBEVEmbeddingAugmentor):
    def __init__(self, apply_probability: float = 1.0) -> None:
        super().__init__(apply_probability)
    
    def _augment_bev_embeddings(self, data: dict) -> dict:
        cooperative_bev_embeddings = data['cooperative_bev_embeddings']

        if isinstance(cooperative_bev_embeddings, list):
            return dict(
                cooperative_bev_embeddings=[torch.zeros_like(bev_embedding) if isinstance(bev_embedding, torch.Tensor) else None for bev_embedding in cooperative_bev_embeddings],
            )
        else:
            return dict(
                cooperative_bev_embeddings=torch.zeros_like(cooperative_bev_embeddings),
            )


class CooperativeHistoryBEVRandomZeroOutAugmentor(BaseHistoryBEVEmbeddingAugmentor):
    def __init__(self, apply_probability: float = 1.0, zero_probability: float = 0.1) -> None:
        super().__init__(apply_probability)
        self.zero_probability = zero_probability
    
    def _random_zero_out(self, bev_embedding: torch.Tensor):
        # shape of bev_embedding: (embedding_size, height, width)
        # zero out random pixels
        b, ch, height, width = bev_embedding.shape
        zero_mask = torch.rand(b, ch, height, width) < self.zero_probability
        bev_embedding[zero_mask] = 0.0

        return bev_embedding

    def _augment_bev_embeddings(self, data: dict) -> dict:
        cooperative_bev_embeddings = data['cooperative_bev_embeddings']

        return dict(
            cooperative_bev_embeddings=[self._random_zero_out(bev_embedding) if isinstance(bev_embedding, torch.Tensor) else None for bev_embedding in cooperative_bev_embeddings],
        )


class CooperativeHistoryBEVNoneAugmentor(BaseHistoryBEVEmbeddingAugmentor):
    def __init__(self, apply_probability: float = 1.0) -> None:
        super().__init__(apply_probability)
    
    def _augment_bev_embeddings(self, data: dict) -> dict:
        cooperative_bev_embeddings = data['cooperative_bev_embeddings']

        if isinstance(cooperative_bev_embeddings, list):
            return dict(
                cooperative_bev_embeddings=[None for _ in cooperative_bev_embeddings],
            )
        else:
            return dict(
                cooperative_bev_embeddings=None,
            )
