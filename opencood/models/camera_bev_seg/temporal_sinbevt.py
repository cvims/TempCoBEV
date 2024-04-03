"""
Implementation of Brady Zhou's cross view transformer
"""
import torch
from einops import rearrange
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.naive_compress import NaiveCompressor

from opencood.models.camera_bev_seg.cobevt import STTF
from opencood.models.temporal_modules import build_temporal_module
from opencood.models import TemporalFusionModel
from opencood.tools.evaluation_temporal.utils import plot_embeddings, plot_channels


class TemporalSinBEVT(TemporalFusionModel):
    def __init__(self, config):
        super(TemporalSinBEVT, self).__init__(config)
        self.max_cav = config['max_cav']

        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        fax_params = config['fax']
        fax_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(fax_params)

        if config['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(128, config['compression'])
        else:
            self.compression = False

        # temporal module
        self.temporal_module = build_temporal_module(config['temporal_fusion'])

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
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
        x = scenario['inputs']
        b, l, m, _, _, _ = x.shape
        assert l == 1, 'TemporalSinBEVT only supports l=1'

        x = self.encoder(x)
        scenario.update({'features': x})
        x = self.fax(scenario)

        # B*L, C, H, W
        x = x.squeeze(1)

        # compressor
        if self.compression:
            x = self.naive_compressor(x)

        kwargs.update({'scenario': scenario})
        
        # spatial fusion
        return x, kwargs


    def temporal_fusion(
        self,
        x,
        history_cooperative_bev_embeddings,
        **kwargs
    ):
        if history_cooperative_bev_embeddings is None:
            history_cooperative_bev_embeddings = []

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
        scenario: dict,
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
