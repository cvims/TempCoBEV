"""
Implementation of Brady Zhou's cross view transformer
"""
import torch
from einops import rearrange
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead

from opencood.models.temporal_modules import build_temporal_module
from opencood.models import TemporalFusionModel
from opencood.tools.evaluation_temporal.utils import plot_embeddings, plot_channels
  


class TemporalV2VNet(TemporalFusionModel):
    def __init__(self, config):
        super(TemporalV2VNet, self).__init__(config)
        # temporal module
        self.temporal_module = build_temporal_module(config['temporal_fusion'])

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static different
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.seg_head = BevSegHead(self.target,
                                   config['seg_head_dim'],
                                   config['output_class'])


    def until_fusion(
        self,
        scenario,
        **kwargs
    ):
        # unpack scenario
        inputs = scenario['inputs']

        kwargs['scenario'] = scenario

        return inputs, kwargs
    
    def temporal_fusion(
        self,
        x,
        history_cooperative_bev_embeddings,
        **kwargs
    ):
        if history_cooperative_bev_embeddings is None:
            history_cooperative_bev_embeddings = []
        
        x = x.permute(0, 3, 1, 2)

        x = self.temporal_module(
            x,
            history_cooperative_bev_embeddings,
            **kwargs
        )

        return x

    def from_fusion(
        self,
        x: torch.Tensor,
        **kwargs
    ):
        # temporal bev embedding output (from fusion)
        bev_embedding = x

        x = x.unsqueeze(1)

        # dynamic head
        x = self.decoder(x)

        x = rearrange(x, 'b l c h w -> (b l) c h w')
        b = x.shape[0]
        output_dict = self.seg_head(x, b, 1)

        output_dict.update({'bev_embedding': bev_embedding})

        return output_dict


    def forward(
        self,
        scenario: torch.Tensor,
        history_cooperative_bev_embeddings,
        only_bev_embeddings: bool = False,
        **kwargs
    ):
        x, _kwargs = self.until_fusion(
            scenario,
            **kwargs
        )

        kwargs.update(_kwargs)

        x = self.temporal_fusion(
            x,
            history_cooperative_bev_embeddings,
            **kwargs
        )

        if only_bev_embeddings:
            return dict(
                bev_embedding=x
            )
    
        return self.from_fusion(
            x,
            **kwargs
        )
