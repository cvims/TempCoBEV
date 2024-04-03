"""
Implementation of Brady Zhou's cross view transformer
"""
import torch
from einops import rearrange
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.cvt_modules import CrossViewModule
from opencood.models.sub_modules.f_cooper_fuse import SpatialFusionMask
from opencood.models.sub_modules.fuse_utils import regroup

from opencood.models.camera_bev_seg.cobevt import STTF
from opencood.models.temporal_modules import build_temporal_module
from opencood.models import TemporalFusionModel
from opencood.tools.evaluation_temporal.utils import plot_embeddings, plot_channels
  

class TemporalFCooper(TemporalFusionModel):
    def __init__(self, config):
        super(TemporalFCooper, self).__init__(config)
        self.max_cav = config['max_cav']

        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        cvm_params = config['cvm']
        cvm_params['backbone_output_shape'] = self.encoder.output_shapes
        self.cvm = CrossViewModule(cvm_params)

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']
        self.sttf = STTF(config['sttf'])

        # spatial fusion
        self.fusion_net = SpatialFusionMask()

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
        # shape: (B, max_cav, 4, 4)
        transformation_matrix = scenario['transformation_matrix']
        if transformation_matrix.shape[1] < self.max_cav:
            # add torch eye for the rest
            transformation_matrix = torch.cat(
                [
                    transformation_matrix,
                    torch.eye(4).unsqueeze(0).repeat(
                        b,
                        self.max_cav - transformation_matrix.shape[1],
                        1,
                        1
                    ).to(transformation_matrix.device)
                ],
                dim=1
            )
        record_len = scenario['record_len']

        x = self.encoder(x)
        scenario.update({'features': x})
        x = self.cvm(scenario)

        # B*L, C, H, W
        x = x.squeeze(1)
        
        # Reformat to (B, max_cav, C, H, W)
        x, _ = regroup(x, record_len, self.max_cav)
        # perform feature spatial transformation,  B, max_cav, H, W, C
        x = self.sttf(x, transformation_matrix)

        kwargs.update({'scenario': scenario})
        
        # spatial fusion
        return self.fusion_net(x), kwargs
    
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

        x = x.unsqueeze(1).permute(0, 1, 4, 2, 3)

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
