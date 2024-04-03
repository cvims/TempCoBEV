import math
import torch
import torch.nn as nn

from torch import einsum
from einops import rearrange

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch
from opencood.models.temporal_modules.utils import get_reference_points

from opencood.tools.evaluation_temporal.utils import plot_embeddings


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.0, out_attention=False):
        super().__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == dim)

        self.heads = num_heads
        self.scale = dim_head**-0.5
        self.out_attention = out_attention

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)  

        out = einsum("b h i j, b h j d -> b h i d", attn, v)  
        out = rearrange(out, "b h n d -> b n (h d)")

        if self.out_attention:
            return self.to_out(out), attn
        else:
            return self.to_out(out)


class DeformableCrossAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.num_levels = kwargs['num_levels']
        self.num_points = kwargs['num_points']

        self.num_heads = kwargs['num_heads']
        embed_dim = kwargs['dim']

        self.dropout = nn.Dropout(kwargs['dropout'])

        self.im2col_step = kwargs['im2col_step']

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
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):

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
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        
        output = self.output_proj(output)


        return self.dropout(output)


class FeedForward(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        dim = kwargs['dim']
        widening_factor = kwargs['widening_factor']
        dropout = kwargs['dropout']

        self.ff_block = nn.Sequential(
            nn.Linear(dim, dim * widening_factor),
            nn.Dropout(dropout),
            nn.Linear(dim * widening_factor, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, identity=None) -> torch.Tensor:
        x = self.ff_block(x)
        if identity is not None:
            x = x + identity
        
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_CA(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y, **kwargs):
        return self.fn(self.norm(x), self.norm(y), **kwargs)


class TemporalDeformableAttention(nn.Module):
    def __init__(self, **kwargs):
        super(TemporalDeformableAttention, self).__init__()
        self.recurrent = kwargs['recurrent']
        self.use_latest_history_only = kwargs['use_latest_history_only']
        self.use_advanced_query = kwargs['use_advanced_query']
        if not self.recurrent:
            # We use all history frame if we dont use recurrent mechanism by default
            # the recurrent approach should keep the history frames in the hidden (latest) state
            self.use_latest_history_only = False
        self.spatial_width = kwargs['width']
        self.spatial_height = kwargs['height']
        self.channel = kwargs['channel']

        self.num_blocks = kwargs['num_blocks']
        if self.use_latest_history_only:
            kwargs['frames'] = 2
            kwargs['cross_attention']['num_levels'] = 2

        self.hist_frames = kwargs['frames']

        # at least 10% of the image area as object points (ref points)
        min_perc = 0.1
        self.min_obj_points = int(self.spatial_width * self.spatial_height * min_perc)

        self.conf_threshold = kwargs['confidence_threshold']

        self.pos_embedding = nn.Linear(2, self.channel)

        self.blocks = nn.ModuleList([
            DeformableBlock(**kwargs) for _ in range(self.num_blocks)
        ])
    
    def _prepare_history_bevs(self, prev_output: torch.Tensor, history_bevs: torch.Tensor) -> torch.Tensor:
        """
        :param prev_output: output of the previous iteration. Shape: (B, C, H, W)
        :param history_bevs: history bev frames. Shape: (B, T, C, H, W)
        """
        num_histories = history_bevs.shape[1]
        # stack them on the feature dimension
        B, C, H, W = prev_output.shape
        prev_output = prev_output.reshape(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
        history_bevs = history_bevs.reshape(B, num_histories, C, H*W).permute(0, 1, 3, 2)  # (B, T, H*W, C)
        history_bevs = history_bevs.reshape(B, -1, C)

        # concatenate them
        history_bevs = torch.cat([history_bevs, prev_output], dim=1)  # (B, T*H*W+H*W, C)

        if self.hist_frames > num_histories + 1:
            # use the last history frame as a filler
            filler = history_bevs[:, -H*W:, :]
            filler = filler.repeat(1, self.hist_frames - num_histories - 1, 1)
            history_bevs = torch.cat([history_bevs, filler], dim=1)  # (B, T*H*W+H*W, C)
        
        if num_histories + 1 > self.hist_frames:
            # use only the last self.hist_frames history frames
            history_bevs = history_bevs[:, -self.hist_frames*(H*W):, :]

        return history_bevs


    def _calculate_confidence(
            self,
            conf_current_bev: torch.Tensor,
            conf_history_bevs: torch.Tensor,
    ):
        # reshape history bevs from [B, N, 2, H, W] to [B, N, H*W]
        B, N, _, H, W = conf_history_bevs.shape
        conf_history_bevs = conf_history_bevs.reshape(B, N, 2, -1)

        # reshape current bev from [B, 2, H, W] to [B, 1, H*W]
        conf_current_bev = conf_current_bev.reshape(B, 2, -1).unsqueeze(1)

        # concatenate them
        cat = torch.cat([conf_history_bevs, conf_current_bev], dim=1)  # (B, N+1, H*W)

        cat = cat.reshape(B, -1, H*W)
        scores, labels = torch.max(cat, dim=1)  # (B, H*W)

        sort = scores.sort(1, descending=True)
        score_value = sort[0]
        order = sort[1]

        batched_order = []
        for b in range(B):
            obj_num = len(torch.where(score_value[b] > self.conf_threshold)[0])
            obj_num = max(obj_num, self.min_obj_points)

            batched_order.append(order[b][:obj_num])

        return batched_order

    def _create_advanced_query(
            self, data_current_proposal: tuple, data_temporal: tuple):
        # use the confidence of the current bev and the latest history bev to create a query
        current_bev_proposal, current_confidence = data_current_proposal
        temporal_bev_proposal, temporal_confidence = data_temporal

        B, C, H, W = current_bev_proposal.shape
        fusion_list = []
        for b in range(B):
            confidence_current_proposal = current_confidence[b:b+1, :].sigmoid().max(dim=1)[0].unsqueeze(1)
            confidence_temporal_proposal = temporal_confidence[b:b+1, :].sigmoid().max(dim=1)[0].unsqueeze(1)

            total_confidence = torch.cat([confidence_current_proposal, confidence_temporal_proposal], dim=1)
            total_confidence = torch.softmax(total_confidence, dim=1)
            ft_current_proposal = current_bev_proposal[b:b+1, :, :, :] * total_confidence[:,0:1,:,:]
            ft_temporal_proposal = temporal_bev_proposal[b:b+1, :, :, :] * total_confidence[:,1:2,:,:]
            fusion_list.append(ft_current_proposal + ft_temporal_proposal)
        
        fusion = torch.cat(fusion_list, dim=0)
        return fusion


    def forward(
            self,
            current_bev_proposal: torch.Tensor,
            history_bevs: torch.Tensor,
            conf_current_bev: torch.Tensor,
            conf_history_bevs: torch.Tensor,
            **kwargs
        ):
        block_output = current_bev_proposal.clone()
        if self.use_advanced_query:
            block_output = self._create_advanced_query(
                (current_bev_proposal, conf_current_bev),
                (history_bevs[:, -1, :, :, :], conf_history_bevs[:, -1, :, :, :])
            )
        # TODO
        # build mask with confidences where to take the reference points from
        # and where to focus the attention on
        # build mask from confidence of current bev and history bevs
        # batched_order = self._calculate_confidence(conf_current_bev, conf_history_bevs)
        B, C, H, W = block_output.shape
        device = block_output.device

        reference_points = get_reference_points(
            self.spatial_height, self.spatial_width, dim='2d', bs=B, device=device, dtype=torch.float32
        )

        # # Create a mask for the reference points to be set to -1
        # mask = torch.zeros(B, reference_points.shape[1], 1, 2, dtype=torch.float32, device=reference_points.device)
        # for b in range(B):
        #     indices = batched_order[b].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
        #     mask[b].scatter_(0, indices, reference_points[b])

        # reference_points = mask

        pos_embedding = self.pos_embedding(reference_points.squeeze(2))

        # create history_bevs (temporal input)
        history_bevs = self._prepare_history_bevs(current_bev_proposal, history_bevs)

        spatial_shapes = torch.as_tensor(
            [(H, W)] * self.hist_frames, dtype=torch.long, device=device
        )

        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        block_output = block_output.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        for block in self.blocks:
            block_output = block(
                query=block_output,
                history_bevs=history_bevs,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                pos_embedding=pos_embedding,
                **kwargs
            )


        # # scatter back the old values of current bev proposal into the prediction
        # current_bev_proposal = current_bev_proposal.reshape(B, C, -1).permute(0, 2, 1)
        # for b in range(B):
        #     indices = batched_order[b]
        #     # find indices of unattened positions
        #     all_indices = torch.arange(0, H*W, device=device)
        #     unattended_indices = torch.as_tensor(list(set(all_indices.tolist()) - set(indices.tolist())), device=device)

        #     block_output[b].scatter_(0, unattended_indices.unsqueeze(-1).expand(-1, C), current_bev_proposal[b])
        
        # reshape back
        block_output = block_output.permute(0, 2, 1).reshape(B, C, H, W)

        return block_output


class DeformableBlock(nn.Module):
    def __init__(self, **kwargs):
        super(DeformableBlock, self).__init__()
        sequence_length = kwargs['channel']

        self.self_attn = PreNorm(
            sequence_length,
            Attention(
                dim=sequence_length,
                dim_head=sequence_length // kwargs['self_attention']['num_heads'],
                num_heads=kwargs['self_attention']['num_heads'],
                dropout=kwargs['self_attention']['dropout'],
            )
        )

        self.cross_attn = PreNorm_CA(
            sequence_length,
            DeformableCrossAttention(
                dim=sequence_length,
                num_heads=kwargs['cross_attention']['num_heads'],
                dropout=kwargs['cross_attention']['dropout'],
                num_levels=kwargs['cross_attention']['num_levels'],
                num_points=kwargs['cross_attention']['num_points'],
                im2col_step=kwargs['cross_attention']['im2col_step'],
            )
        )

        self.ff = PreNorm(
            sequence_length,
            FeedForward(
                dim=sequence_length,
                widening_factor=kwargs['feed_forward']['widening_factor'],
                dropout=kwargs['feed_forward']['dropout'],
            )
        )

    def forward(
            self,
            query: torch.Tensor,
            history_bevs: torch.Tensor,
            reference_points: torch.Tensor,
            spatial_shapes: torch.Tensor,
            level_start_index: torch.Tensor,
            pos_embedding: torch.Tensor = None,
            **kwargs):
        
        H, W = spatial_shapes[-1]

        # self attention
        if pos_embedding is not None:
            x = self.self_attn(query + pos_embedding)
        else:
            x = self.self_attn(query)
        
        # residual
        x = x + query

        
        # cross attention
        if pos_embedding is not None:
            cross_res = self.cross_attn(
                x + pos_embedding, history_bevs,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
        else:
            cross_res = self.cross_attn(
                x, history_bevs,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
        
        # residual
        x = cross_res + x

        
        # feed forward
        ff = self.ff(x)

        # residual
        x = ff + x

        return x
