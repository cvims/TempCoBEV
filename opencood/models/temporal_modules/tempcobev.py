from typing import List
import torch
import torch.nn as nn

from opencood.models.temporal_modules.utils import shift_bev_with_ego_motion
from opencood.tools.evaluation_temporal.utils import plot_embeddings

import numpy as np
from opencood.models.temporal_modules.temporal_transformer import TemporalTransformer
from opencood.models.temporal_modules.temporal_deformable_attention import TemporalDeformableAttention


def prepare_history_bevs(history_bevs: torch.Tensor, vehicle_offsets: torch.Tensor) -> torch.Tensor:
        # shifted_reference_points = None
        if len(history_bevs) > 0:
            # iterate through histories and warp them to ego motion
            # start from the latest history and go back in time
            # shifted_reference_points = []
            for i in range(len(history_bevs) - 1, -1, -1):
                history_bev = history_bevs[i]
                # hist_ref_points_batched = []
                for batch_id, b_history_bev in enumerate(history_bev):
                    # first key of vehicle offsets is ego
                    ego_id = list(vehicle_offsets[i][batch_id].keys())[0]
                    ego_offset_t_matrix = vehicle_offsets[i][batch_id][ego_id]

                    _i = i
                    while (len(history_bevs) - 1) > _i:
                        # future ego offset
                        future_ego_offset_t_matrix = vehicle_offsets[_i+1][batch_id][ego_id]
                        # from t-n to t-n+1
                        ego_offset_t_matrix = torch.matmul(future_ego_offset_t_matrix, ego_offset_t_matrix)
                        _i += 1

                    shift, latest_history = shift_bev_with_ego_motion(
                        bev=b_history_bev,
                        ego_offset_t_matrix=ego_offset_t_matrix
                    )

                    # shift reference points
                    # ref_points = reference_points.clone()[batch_id,:,0,:]
                    # hist_ref_points_batched.append((ref_points + shift).unsqueeze(0))

                    history_bevs[i][batch_id] = latest_history

                # shifted_reference_points.append(torch.cat(hist_ref_points_batched))
            
            # shifted_reference_points = torch.stack(shifted_reference_points)
            # shifted_reference_points = shifted_reference_points.to(history_bevs[0].device)
            # shifted_reference_points = shifted_reference_points.permute(1,2,0,3)

        return history_bevs#, shifted_reference_points


def init_gaussian_filter(k_size=5, sigma=1):
    def _gen_gaussian_kernel(k_size=5, sigma=1):
        center = k_size // 2
        x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
        g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
        return g
    gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
    gaussian_filter = nn.Conv2d(1, 1, kernel_size=k_size, stride=1, padding=(k_size-1)//2)
    gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
    gaussian_filter.bias.data.zero_()
    # gaussian_filter.requires_grad_ = False

    return gaussian_filter



class FusionMask(nn.Module):
    def __init__(self, confidence_threshold: float, gaussian_smooth: dict, **kwargs) -> None:
        super().__init__()
        self.threshold = confidence_threshold
        self.smooth = False
        if gaussian_smooth is not None:
            self.smooth = True
            learnable = gaussian_smooth['learnable']
            if learnable:
                raise NotImplementedError
                # self.gaussian_filter = LearnableGaussianFilter(
                #     k_size=gaussian_smooth['k_size'],
                #     sigma=gaussian_smooth['c_sigma']
                # )
            else:
                kernel_size = gaussian_smooth['k_size']
                c_sigma = gaussian_smooth['c_sigma']
                self.gaussian_filter = init_gaussian_filter(kernel_size, c_sigma)


    def forward(self, current_confidence_map: torch.Tensor, history_confidence_maps: torch.Tensor, **kwargs):
        """
        :param current_confidence_map: (B, 1, H, W)
        :param history_confidence_maps: (B, N, 1, H, W)
        """
        # fuse current confidence map with history confidence maps
        current_confidence_map = current_confidence_map.unsqueeze(1)
        confidence_maps = torch.cat([current_confidence_map, history_confidence_maps], dim=1)

        B, N, _, H, W = confidence_maps.shape

        fusion_masks = []
        for b in range(B):
            confidence_map = confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1)
            if self.smooth:
                confidence_map = self.gaussian_filter(confidence_map)
            ones_mask = torch.ones_like(confidence_map).to(confidence_map.device)
            zeros_mask = torch.zeros_like(confidence_map).to(confidence_map.device)
            communication_mask = torch.where(confidence_map > self.threshold, ones_mask, zeros_mask)
            fusion_masks.append(communication_mask)
        
        return torch.cat(fusion_masks, dim=0)


class LateFusion(nn.Module):
    def __init__(self, confidence_threshold: float, gaussian_smooth: dict, **kwargs) -> None:
        super().__init__()
        self.threshold = confidence_threshold
        self.smooth = False
        if gaussian_smooth is not None:
            self.smooth = True
            kernel_size = gaussian_smooth['k_size']
            c_sigma = gaussian_smooth['c_sigma']
            self.gaussian_filter = init_gaussian_filter(kernel_size, c_sigma)
    

    def forward(self, data_current_proposal: tuple, data_temporal: tuple, **kwargs):
        current_bev_proposal, current_confidence = data_current_proposal
        temporal_bev_proposal, temporal_confidence = data_temporal

        B, C, H, W = current_bev_proposal.shape
        fusion_list = []
        for b in range(B):
            confidence_current_proposal = current_confidence[b:b+1, :].sigmoid().max(dim=1)[0].unsqueeze(1)
            confidence_temporal_proposal = temporal_confidence[b:b+1, :].sigmoid().max(dim=1)[0].unsqueeze(1)
            if self.smooth:
                confidence_current_proposal = self.gaussian_filter(confidence_current_proposal)
                confidence_temporal_proposal = self.gaussian_filter(confidence_temporal_proposal)
            total_confidence = torch.cat([confidence_current_proposal, confidence_temporal_proposal], dim=1)
            total_confidence = torch.softmax(total_confidence, dim=1)
            ft_current_proposal = current_bev_proposal[b:b+1, :, :, :] * total_confidence[:,0:1,:,:]
            ft_temporal_proposal = temporal_bev_proposal[b:b+1, :, :, :] * total_confidence[:,1:2,:,:]
            fusion_list.append(ft_current_proposal + ft_temporal_proposal)

        fusion = torch.cat(fusion_list, dim=0)
        return fusion


class TemporalFusion(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, current_bev_proposal: torch.Tensor, history_bevs: torch.Tensor, fusion_mask: torch.Tensor, **kwargs):
        # # project histories with ego motion
        # history_bevs = prepare_history_bevs(
        #     history_bevs=history_bevs,
        #     vehicle_offsets=kwargs['vehicle_offsets'],
        #     # reference_points=reference_points
        # )

        # TODO

        return current_bev_proposal


class ConfidenceEstimator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()

        self.confidence_coarse = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

        self.confidence_fine = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )


    def forward(self, data: torch.tensor, **kwargs):
        # coarse_confidence_out = self.confidence_coarse(data)
        fine_confidence_out = self.confidence_fine(data)

        # max over classes
        # coarse_confidence = coarse_confidence_out.sigmoid().max(dim=1)[0].unsqueeze(1)
        # fine_confidence = fine_confidence_out.sigmoid().max(dim=1)[0].unsqueeze(1)

        # mean over channels
        # confidence = torch.cat([coarse_confidence, fine_confidence], dim=1)
        # confidence = confidence.mean(dim=1).unsqueeze(1)

        # confidence = fine_confidence

        return fine_confidence_out #confidence, coarse_confidence_out, fine_confidence_out


class TempCoBEV(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.recurrent = kwargs['recurrent']
        # len(histories) + current frame == frames
        self.max_histories = kwargs['frames'] - 1

        self.spatial_w = kwargs['spatial_width']
        self.spatial_h = kwargs['spatial_height']
        embed_dim = kwargs['sequence_length']

        self.confidence_estimator = ConfidenceEstimator(
            in_channels=embed_dim,
            out_channels=2,
            **kwargs
        )

        self.fusion_mask = FusionMask(
            confidence_threshold=kwargs['confidence_threshold'],
            gaussian_smooth=kwargs['gaussian_smooth'] if 'gaussian_smooth' in kwargs else None,
        )

        self.temporal_fusion = self._load_temporal_module(
            name=kwargs['temporal_module']['name'],
            **kwargs['temporal_module']['args'],
        )

        # self.temporal_fusion = TemporalFusion(
        #     **kwargs
        # )

        self.use_late_fusion = kwargs['use_late_fusion'] if 'use_late_fusion' in kwargs else True

        if self.use_late_fusion:
            self.late_fusion = LateFusion(
                confidence_threshold=kwargs['confidence_threshold'],
                gaussian_smooth=kwargs['gaussian_smooth'] if 'gaussian_smooth' in kwargs else None,
            )

    def _load_temporal_module(
        self, name: str, **kwargs
    ):
        if name == 'temporal_transformer':
            return TemporalTransformer(
                feature_size=self.spatial_h*self.spatial_w,
                **kwargs
            )
        elif name == 'deformable_attention':
            return TemporalDeformableAttention(
                **kwargs
            )
        else:
            raise NotImplementedError

    
    def forward(self, current_bev_proposal: torch.Tensor, history_bevs: List[torch.Tensor], **kwargs):
        """
        :param current_bev_proposal: (B, C, H, W)
        :param history_bevs: (B, N, C, H, W) or None. N for number of histories
        """
        _, C, H, W = current_bev_proposal.shape
        len_H = 0

        # without history, we always use the original output
        if not history_bevs:
            return current_bev_proposal
        else:
            # resume with current bev proposal as output if recurrent is false
            # and history bev is not filled up (self.frames)
            if not self.recurrent and len(history_bevs) < self.max_histories:
                return current_bev_proposal
            history_bevs = [history_bev.clone() for history_bev in history_bevs]
            len_H = len(history_bevs)
            history_bevs = torch.stack(history_bevs, dim=1).contiguous().view(-1, C, H, W)

        # confidence estimation for current bev proposal
        current_cls = self.confidence_estimator(current_bev_proposal)

        # confidence estimation for history bev proposals
        history_cls = self.confidence_estimator(history_bevs)
        history_cls = history_cls.view(-1, len_H, 2, H, W)
        history_bevs = history_bevs.view(-1, len_H, C, H, W)

        # communication mask (dynamic scene masking)
        # fusion_mask = self.fusion_mask(current_cls, history_cls)

        # perform temporal fusion
        temporal_output = self.temporal_fusion(
            current_bev_proposal, history_bevs,
            conf_current_bev=current_cls,
            conf_history_bevs=history_cls,
            **kwargs)


        if self.use_late_fusion:
            # confidence estimation for temporal output
            temporal_confidence = self.confidence_estimator(temporal_output)
            
            # late fusion
            fusion = self.late_fusion(
                data_current_proposal=(current_bev_proposal, current_cls),
                data_temporal=(temporal_output, temporal_confidence),
                **kwargs
            )
        else:
            fusion = temporal_output

        # fusion = temporal_output
        return fusion
