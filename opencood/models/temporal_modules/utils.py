import torch
import torch.nn as nn
import math
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch

from mmcv.utils import ext_loader
from mmcv.cnn import build_norm_layer, build_conv_layer
from mmdet.models.backbones.resnet import BasicBlock

from torchvision.models.resnet import Bottleneck
ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

from einops import rearrange
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine

from opencood.tools.evaluation_temporal.utils import plot_embeddings


ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d


class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

        self.init_weights()
    

    def init_weights(self):
        """Initialize the weights of embedding layers."""
        nn.init.xavier_uniform_(self.row_embed.weight)
        nn.init.xavier_uniform_(self.col_embed.weight)


    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos
    
    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str


class BEVCrossAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.num_levels = kwargs['num_levels']
        self.num_points = kwargs['num_points']

        self.num_heads = kwargs['num_heads']
        embed_dim = kwargs['embed_dim']

        self.dropout = nn.Dropout(kwargs['dropout'])

        self.sampling_offsets = nn.Linear(
            embed_dim, self.num_heads * self.num_levels * self.num_points * 2)
        self.attention_weights = nn.Linear(embed_dim,
                                           self.num_heads * self.num_levels * self.num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.init_weights()


    def init_weights(self):
        """Default initialization for Parameters of Module."""
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    
    def forward(self,
                query,
                key=None,
                value=None,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                im2col_step=None,
                level_start_index=None):
        if key is None:
            key = query
        
        if value is None:
            value = key
        
        if residual is None:
            residual = query

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        
        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy).contiguous()

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
    
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        
        output = self.output_proj(output)

        return self.dropout(output) + residual


class BEVMultiHistoryCrossAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.num_levels = kwargs['num_levels']
        self.num_points = kwargs['num_points']

        self.num_heads = kwargs['num_heads']
        embed_dim = kwargs['embed_dim']

        self.dropout = nn.Dropout(kwargs['dropout'])

        self.sampling_offsets = nn.Linear(
            embed_dim, self.num_heads * self.num_levels * self.num_points * 2)
        self.attention_weights = nn.Linear(embed_dim,
                                           self.num_heads * self.num_levels * self.num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.init_weights()


    def init_weights(self):
        """Default initialization for Parameters of Module."""
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    
    def forward(self,
                query,
                key=None,
                value=None,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                im2col_step=None,
                level_start_index=None):
        if key is None:
            key = query
        
        if value is None:
            value = key
        
        if residual is None:
            residual = query

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, embed_dims = query.shape

        queries_rebatch = query.new_zeros(
             [bs, reference_points.shape[0], num_query, embed_dims])
        reference_points_rebatch = reference_points.new_zeros(
            [bs, reference_points.shape[0], num_query, *reference_points.shape[-2:]])
        
        for j in range(bs):
            for i in range(reference_points.shape[0]):
                queries_rebatch[j, i] = query[j]
                reference_points_rebatch[j, i] = reference_points[i]
        
        query = queries_rebatch.view(bs * reference_points.shape[0], num_query, embed_dims)
        reference_points = reference_points_rebatch.view(bs * reference_points.shape[0], num_query, *reference_points.shape[-2:])


        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        
        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy).contiguous()

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
    
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs)
        output = output.view(num_query, embed_dims, 1, bs)
        output = output.mean(-1)

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        return self.dropout(output) + residual


class TemporalSelfAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.num_bev_queue=kwargs['num_bev_queue']
        self.num_levels=kwargs['num_levels']
        self.num_points=kwargs['num_points']
        embed_dim=kwargs['embed_dim']

        self.num_heads = kwargs['num_heads']

        self.dropout = nn.Dropout(kwargs['dropout'])

        self.sampling_offsets = nn.Linear(
            embed_dim*self.num_bev_queue, self.num_bev_queue*self.num_heads * self.num_levels * self.num_points * 2)
        self.attention_weights = nn.Linear(embed_dim*self.num_bev_queue,
                                           self.num_bev_queue*self.num_heads * self.num_levels * self.num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.init_weights()
    

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
   
    def forward(self,
                query,
                key,
                value,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                im2col_step,
                identity=None,
                key_padding_mask=None):

        if value is None:
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)

        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos
        
        bs, num_query, embed_dims = query.shape
        *_, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bs*self.num_bev_queue,
                              num_value, self.num_heads, -1)
    
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        # # referce points shape [BS, N, 1, 2] -> search for indices N where the last dimension has negative values
        # # batch-wise indices
        # neg_indices_y = reference_points[..., 0] < 0
        # neg_indices_x = reference_points[..., 1] < 0
        # neg_indices = torch.cat([neg_indices_y, neg_indices_x], 2)
        # # reduce to shape [BS, N]. Set to true if any of the last dimension is negative
        # neg_indices = neg_indices.any(-1)
        # neg_indices = neg_indices.view(reference_points.size(0), -1)
        # # mask sampling offsets weights for negative indices, zero out sampling offsets
        # # sampling offsets shape: [2, N, 8, 1, 4, 2]; neg_indices shape: [2, N, 2]
        # # set sampling offsets to zero for all negative indices
        # sampling_offsets[neg_indices, :, :, :, :] = 0.0

        
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1).to(device=query.device)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        sampling_locations = sampling_locations.contiguous()
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

         # output shape (bs*num_bev_queue, num_query, embed_dims)
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        return self.dropout(output) + identity


class ResNetFusion(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_layer, norm_cfg=dict(type='BN')):
        super(ResNetFusion, self).__init__()
        layers = []
        self.inter_channels = inter_channels
        # test layer
        self.pre_layer_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        for i in range(num_layer):
            if i == 0:
                if inter_channels == in_channels:
                    layers.append(BasicBlock(in_channels, inter_channels, stride=1, norm_cfg=norm_cfg))
                else:
                    downsample = nn.Sequential(
                        build_conv_layer(None, in_channels, inter_channels, 3, stride=1, padding=1, dilation=1,
                                         bias=False),
                        build_norm_layer(norm_cfg, inter_channels)[1])
                    layers.append(
                        BasicBlock(in_channels, inter_channels, stride=1, norm_cfg=norm_cfg, downsample=downsample))
            else:
                layers.append(BasicBlock(inter_channels, inter_channels, stride=1, norm_cfg=norm_cfg))
        self.layers = nn.Sequential(*layers)
        self.layer_norm = nn.Sequential(
                nn.Linear(inter_channels, out_channels),
                nn.LayerNorm(out_channels))

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1).contiguous()
        else:
            x = x.contiguous()
        # x should be [1, in_channels, bev_h, bev_w]
        b, c, bev_h, bev_w = x.shape

        # test layer norm
        x = x.reshape(shape=(b, c, -1)).permute(0, 2, 1)
        x = self.pre_layer_norm(x)
        x = x.permute(0, 2, 1).reshape(shape=(b, c, bev_h, bev_w))

        for lid, layer in enumerate(self.layers):
            x = layer(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # nchw -> n(hw)c
        x = self.layer_norm(x)
        # reshape to input format
        x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[2], bev_h, bev_w)
        return x


def shift_bev_with_ego_motion(
        bev: torch.Tensor,
        ego_offset_t_matrix: torch.Tensor
):
    downsample_rate = 8
    discrete_ratio = 100 / 256

    dist_correction_matrix = ego_offset_t_matrix[[0,1],:][:,[0, 1, 3]]
    dist_correction_matrix[:, -1] = dist_correction_matrix[:, -1] \
                                    / (discrete_ratio * downsample_rate)
    dist_correction_matrix.type(dtype=torch.float32)

    # transpose and flip to make the transformation correct
    bev = rearrange(bev, 'c h w  -> c w h')
    bev = torch.flip(bev, dims=(2,))
    # Only compensate non-ego vehicles
    C, H, W = bev.shape

    T = get_transformation_matrix(
        dist_correction_matrix.unsqueeze(0), (H, W)).type(dtype=torch.float32)
    bev = warp_affine(bev.unsqueeze(0), T,
                            (H, W))

    # flip and transpose back
    bev = bev[0]
    bev = torch.flip(bev, dims=(2,))
    bev = rearrange(bev, 'c w h -> c h w')

    bev = bev.unsqueeze(0)

    discrete_ratio = 256 / 100

    real_w = 50 - (-50)
    real_h = 50 - (-50)

    # spatial shape of bev
    bev_h = bev.shape[1]
    bev_w = bev.shape[2]

    grid_length_x = real_w / bev_w
    grid_length_y = real_h / bev_h

    delta_x = ego_offset_t_matrix[0, 3]
    delta_y = ego_offset_t_matrix[1, 3]
    translation_length = torch.sqrt(delta_x ** 2 + delta_y ** 2)

    # yaw
    bev_angle = torch.atan2(ego_offset_t_matrix[1, 0], ego_offset_t_matrix[0, 0]) / torch.pi * 180.0

    shift_y = translation_length * \
        torch.cos(bev_angle / 180 * torch.pi) / grid_length_y / bev_h
    shift_x = translation_length * \
        torch.sin(bev_angle / 180 * torch.pi) / grid_length_x / bev_w

    shift = bev.new_tensor(
        [[shift_x, shift_y]], dtype=torch.float32)
    
    return shift, bev


class BEVFormerFeedForward(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        sequence_length = kwargs['sequence_length']
        widening_factor = kwargs['widening_factor']
        dropout = kwargs['dropout']

        self.ff_block = nn.Sequential(
            nn.Linear(sequence_length, sequence_length * widening_factor),
            nn.Dropout(dropout),
            nn.Linear(sequence_length * widening_factor, sequence_length),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, identity=None) -> torch.Tensor:
        x = self.ff_block(x)
        if identity is not None:
            x = x + identity
        
        return x


def fill_history_bevs(history_bevs: list, cav_vehicle_offsets: list, mode: str = 'latest'):
    """
    param mode: 'latest' or 'zero'.
                If 'latest' then the 'Nones' in the history bev gets filled from latest to oldest.
                If 'zero' then the 'Nones' in the history bev gets filled with zeros.
    """

    # shape of history bevs [N, C, H, W], where N is the number of history bevs

    if mode == 'latest':
        # first from latest (last index) to oldest and fill space in between with the previous latest
        for i in range(len(history_bevs) - 2, -1, -1):
            if history_bevs[i] is None:
                history_bevs[i] = history_bevs[i + 1]
                cav_vehicle_offsets[i] = torch.eye(4, dtype=torch.float32, device=history_bevs[i].device)
        # now from oldest (first index) to latest and fill space in between with the previous oldest
        for i in range(1, len(history_bevs)):
            if history_bevs[i] is None:
                history_bevs[i] = history_bevs[i - 1]
                cav_vehicle_offsets[i] = cav_vehicle_offsets[i - 1]
                cav_vehicle_offsets[i-1] = torch.eye(4, dtype=torch.float32, device=history_bevs[i].device)
        # if still None, then fill with zeros
        for i in range(len(history_bevs)):
            if history_bevs[i] is None:
                history_bevs[i] = torch.zeros_like(history_bevs[i - 1])
                cav_vehicle_offsets[i] = torch.eye(4, dtype=torch.float32)
    if mode == 'zero':
        for i in range(len(history_bevs)):
            if history_bevs[i] is None:
                history_bevs[i] = torch.zeros_like(history_bevs[i - 1])
                cav_vehicle_offsets[i] = torch.eye(4, dtype=torch.float32)
    
    return cav_vehicle_offsets, history_bevs



class BEVDynamicRefPointCrossAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.num_levels = kwargs['num_levels']
        self.num_points = kwargs['num_points']

        self.num_heads = kwargs['num_heads']
        embed_dim = kwargs['embed_dim']

        self.dropout = nn.Dropout(kwargs['dropout'])

        self.sampling_offsets = nn.Linear(
            embed_dim, self.num_heads * self.num_levels * self.num_points * 2)
        self.attention_weights = nn.Linear(embed_dim,
                                           self.num_heads * self.num_levels * self.num_points)
        
        # this uses only two frames from history for cross attention (t, and t-1)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.init_weights()


    def init_weights(self):
        """Default initialization for Parameters of Module."""
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)
    
    def forward(self,
                query,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                im2col_step=None,
                level_start_index=None,
                # conf_map=None,
                # conf_mask=None,
                **kwargs):

        if query_pos is not None:
            query = query + query_pos

        # query = torch.mul(query, conf_mask.unsqueeze(-1).expand(-1, -1, query.shape[-1]))

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        # use the sampling offsets from the reference points
        # sampling_offsets = torch.mul(sampling_offsets, conf_mask.unsqueeze(-1).expand(-1, -1, sampling_offsets.shape[-1]))
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)
        
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1).to(sampling_offsets) 

            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        
        output = self.output_proj(output)

        if residual is not None:
            return self.dropout(output) + residual
        else:
            return self.dropout(output)


import torch
from opencood.models.temporal_modules.utils import fill_history_bevs, shift_bev_with_ego_motion


def prepare_history_vehicle_offsets(
        vehicle_offsets: list, history_bevs: list, history_bev_fillup_mode: str = 'zero'
):
    if not vehicle_offsets:
        return vehicle_offsets, history_bevs

    # first cav id of cav_ids is ego
    ego_id = list(vehicle_offsets[0].keys())[0]
    # assert that ego id is in all vehicle offsets
    assert all([ego_id in hist for hist in vehicle_offsets]), 'Ego id is not in all vehicle offsets'

    cav_vehicle_offsets = [hist[ego_id] for hist in vehicle_offsets]
    
    cav_vehicle_offsets, history_bevs = fill_history_bevs(
        history_bevs, mode=history_bev_fillup_mode, cav_vehicle_offsets=cav_vehicle_offsets
    )
    
    return cav_vehicle_offsets, history_bevs


def prepare_history_bevs(history_bevs: torch.Tensor, vehicle_offsets: torch.Tensor) -> torch.Tensor:
    ego_vehicle_offsets, history_bevs = prepare_history_vehicle_offsets(
        vehicle_offsets=vehicle_offsets,
        history_bevs=history_bevs,
        history_bev_fillup_mode='zero'
    )

    if len(history_bevs) > 0:
        # iterate through histories and warp them to ego motion
        # start from the latest history and go back in time
        for i in range(len(history_bevs) - 1, -1, -1):
            ego_offset_t_matrix = ego_vehicle_offsets[i]

            # e.g. to calculate the motion from t-2 to t, we need to warp t-1 to t and then t-2 to t-1
            _i = i
            while (len(history_bevs) - 1) > _i:
                # future ego offset
                future_ego_offset_t_matrix = ego_vehicle_offsets[_i + 1]
                # from t-n to t-n+1
                ego_offset_t_matrix = torch.matmul(future_ego_offset_t_matrix, ego_offset_t_matrix)
                _i += 1
            
            _, latest_history = shift_bev_with_ego_motion(
                bev=history_bevs[i],
                ego_offset_t_matrix=ego_offset_t_matrix
            )

            history_bevs[i] = latest_history

    return history_bevs
