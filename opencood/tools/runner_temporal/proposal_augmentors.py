import torch
from opencood.tools.runner_temporal import BaseProposalAugmentor
import torch.nn.functional as F

from opencood.models.sub_modules.torch_transformation_utils import get_transformation_matrix, warp_affine
from opencood.tools.evaluation_temporal.utils import plot_embeddings


class ProposalBEVFullZeroOutAugmentor(BaseProposalAugmentor):
    def __init__(self, apply_probability: float = 1.0) -> None:
        super().__init__(apply_probability)
    
    def _augment_fusion(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros_like(data)


class ProposalBEVRandomZeroOutAugmentor(BaseProposalAugmentor):
    def __init__(self, apply_probability: float = 1.0, zero_probability: float = 0.1) -> None:
        super().__init__(apply_probability)
        self.zero_probability = zero_probability
    
    def _random_zero_out(self, bev_embedding: torch.Tensor):
        # shape of bev_embedding: (embedding_size, height, width)
        # zero out random pixels
        if torch.rand(1) < self.apply_probability:
            b, ch, height, width = bev_embedding.shape
            zero_mask = torch.rand(b, ch, height, width) < self.zero_probability
            bev_embedding[zero_mask] = 0.0

            return bev_embedding
        
        return bev_embedding

    def _augment_fusion(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._random_zero_out(data)


# Only for EmbeddingsDataset
class ProposalBEVEmbeddingsOnlyEgoAugmentor(BaseProposalAugmentor):
    def __init__(self, apply_probability: float = 1.0) -> None:
        super().__init__(apply_probability)
    
    def _augment_fusion(self, data: dict, **kwargs) -> dict:
        # data shape: (T, embedding_size, height, width)
        # with T as the number of frames

        for t in range(data['inputs'].shape[0]):
            if torch.rand(1) < self.apply_probability:
                data['inputs'][t] = data['inputs_ego'][t]

        return data


class ProposalBEVEmbeddingsOnlyEgoAugmentorXFrames(BaseProposalAugmentor):
    def __init__(self, every_x_frames: int = 0) -> None:
        super().__init__(apply_probability=1.0)
        self.every_x_frames = every_x_frames
        self.internal_counter = 1

    def _augment_fusion(self, data: torch.Tensor, **kwargs) -> torch.Tensor:       
        if self.internal_counter % self.every_x_frames == 0:
            for t in range(data['inputs'].shape[0]):
                data['inputs'][t] = data['inputs_ego'][t]
            self.internal_counter = 1

        self.internal_counter += 1

        return data
    
    def reset(self):
        self.internal_counter = 1


# Only for EmbeddingsDataset
class ProposalBEVEmbeddingsFlipAugmentor(BaseProposalAugmentor):
    def __init__(self, apply_probability: float = 1.0, is_lidar: bool = False) -> None:
        super().__init__(apply_probability)
        self.is_lidar = is_lidar

    def _augment_fusion_lidar(self, data: dict, **kwargs) -> dict:
        # flip vertically and/or horizontally
        if torch.rand(1) > self.apply_probability:
            return data

        # shape [N, C, H_bev, W_bev]; N as the number of frames
        inputs = torch.as_tensor(data['inputs'])
        # shape [N, C, H_bev, W_bev]; N as the number of frames
        inputs_ego = torch.as_tensor(data['inputs_ego'])

        # we flip gt_dynamic, gt_static, inputs, inputs_ego
        # choose randomly if we start with vertical or horizontal flip
        inp_r_dims = [2, 3]
        tgt_r_dims = [0, 1]
        # shuffle dim list
        dim_idx = torch.randperm(len(inp_r_dims)).tolist()
        inp_r_dims = [inp_r_dims[i] for i in dim_idx]
        tgt_r_dims = [tgt_r_dims[i] for i in dim_idx]

        label_dict = []
        # first flip (iterate scenarios)
        for scenario in data['label_dict']:
            _new_dict = {}
            for key in scenario:
                _new_dict[key] = torch.flip(torch.as_tensor(scenario[key]), dims=(tgt_r_dims[0],))
            label_dict.append(_new_dict)
        inputs = torch.flip(inputs, dims=(inp_r_dims[0],))
        inputs_ego = torch.flip(inputs_ego, dims=(inp_r_dims[0],))

        # optional second flip
        if torch.rand(1) < 0.5:
            label_dict = []
            # first flip (iterate scenarios)
            for scenario in data['label_dict']:
                _new_dict = {}
                for key in scenario:
                    _new_dict[key] = torch.flip(torch.as_tensor(scenario[key]), dims=(tgt_r_dims[1],))
                label_dict.append(_new_dict)
            inputs = torch.flip(inputs, dims=(inp_r_dims[1],))
            inputs_ego = torch.flip(inputs_ego, dims=(inp_r_dims[1],))
        
        # new values and the ones not touched from scenario
        data['label_dict'] = label_dict
        data['inputs'] = inputs
        data['inputs_ego'] = inputs_ego

        return data

    def _augment_fusion_camera(self, data: dict, **kwargs) -> dict:
        # flip vertically and/or horizontally
        if torch.rand(1) > self.apply_probability:
            return data

        # shape [N, C, H_gt, W_gt]; N as the number of frames
        gt_dynamic = torch.as_tensor(data['gt_dynamic'])
        # shape [N, C, H_gt, W_gt]; N as the number of frames
        gt_static = torch.as_tensor(data['gt_static'])
        # shape [N, C, H_bev, W_bev]; N as the number of frames
        inputs = torch.as_tensor(data['inputs'])
        # shape [N, C, H_bev, W_bev]; N as the number of frames
        inputs_ego = torch.as_tensor(data['inputs_ego'])

        # we flip gt_dynamic, gt_static, inputs, inputs_ego
        # choose randomly if we start with vertical or horizontal flip
        dim = [2, 3]
        # shuffle dim list
        dim_idx = torch.randperm(len(dim)).tolist()
        dim = [dim[i] for i in dim_idx]

        # first flip
        gt_dynamic = torch.flip(gt_dynamic, dims=(dim[0],))
        gt_static = torch.flip(gt_static, dims=(dim[0],))
        inputs = torch.flip(inputs, dims=(dim[0],))
        inputs_ego = torch.flip(inputs_ego, dims=(dim[0],))

        # optional second flip
        if torch.rand(1) < 0.5:
            gt_dynamic = torch.flip(gt_dynamic, dims=(dim[1],))
            gt_static = torch.flip(gt_static, dims=(dim[1],))
            inputs = torch.flip(inputs, dims=(dim[1],))
            inputs_ego = torch.flip(inputs_ego, dims=(dim[1],))
        
        # new values and the ones not touched from scenario
        data['gt_dynamic'] = gt_dynamic
        data['gt_static'] = gt_static
        data['inputs'] = inputs
        data['inputs_ego'] = inputs_ego

        return data

    def _augment_fusion(self, data: dict, **kwargs) -> dict:
        if self.is_lidar:
            return self._augment_fusion_lidar(data, **kwargs)
        else:
            return self._augment_fusion_camera(data, **kwargs)


# Only for EmbeddingsDataset
class ProposalBEVEmbeddingsRotationAugmentor(BaseProposalAugmentor):
    def __init__(self, apply_probability: float = 1.0) -> None:
        super().__init__(apply_probability)
    
    def _augment_fusion(self, data: dict, **kwargs) -> dict:
        # data shape: (N, embedding_size, height, width)
        # with N as the number of frames
        
        # we rotate gt_dynamic gt_static, inputs, inputs_ego, vehicle_offsets
        if torch.rand(1) > self.apply_probability:
            return data

        # shape: (N, 1, H, W)
        N, _, H_gt, W_gt = data['gt_dynamic'].shape
        gt_dynamic = torch.as_tensor(data['gt_dynamic'])
        gt_static = torch.as_tensor(data['gt_static'])
        gt_dtype = gt_dynamic.dtype
        inputs = torch.as_tensor(data['inputs'])
        inputs_ego = torch.as_tensor(data['inputs_ego'])
        # shape: (N, C, H*scale, W*scale) with scale < 1
        _, _, H_bev, W_bev = inputs.shape

        # create random rotation matrix
        rotation_matrix = torch.zeros((gt_dynamic.shape[0], 2, 3))
        # same angles for all N
        cos = torch.cos(torch.rand(1) * 2 * torch.pi).repeat(gt_dynamic.shape[0])
        sin = torch.sin(torch.rand(1) * 2 * torch.pi).repeat(gt_dynamic.shape[0])
        rotation_matrix[:, 0, 0] = cos
        rotation_matrix[:, 0, 1] = -sin
        rotation_matrix[:, 1, 0] = sin
        rotation_matrix[:, 1, 1] = cos
        rotation_matrix[:, 0, 2] = 0.0  # Set translation parameters to 0
        rotation_matrix[:, 1, 2] = 0.0

        T_gt = get_transformation_matrix(rotation_matrix, (H_gt, W_gt))
        gt_dynamic_rot = warp_affine(
            gt_dynamic.to(dtype=torch.float),
            T_gt.to(dtype=torch.float),
            (H_gt, W_gt)
        )

        #gt_dynamic_rot[gt_dynamic_rot < 0.95] = 0.0
        # others to 1.0
        gt_dynamic_rot[gt_dynamic_rot > 0.0] = 1.0
        # convert back to int
        gt_dynamic_rot = gt_dynamic_rot.to(dtype=gt_dtype)

        gt_static_rot = warp_affine(
            gt_static.to(dtype=torch.float),
            T_gt.to(dtype=torch.float),
            (H_gt, W_gt)
        )

        gt_static_rot[gt_static_rot < 0.95] = 0.0
        # others to 1.0
        gt_static_rot[gt_static_rot > 0.0] = 1.0
        # convert back to int
        gt_static_rot = gt_static_rot.to(dtype=gt_dtype)

        # we rotate inputs and inputs_ego
        T_bev = get_transformation_matrix(rotation_matrix, (H_bev, W_bev))
        inputs_rot = warp_affine(
            inputs,
            T_bev.to(dtype=torch.float),
            (H_bev, W_bev)
        )

        # we rotate inputs_ego
        inputs_ego_rot = warp_affine(
            inputs_ego,
            T_bev.to(dtype=torch.float),
            (H_bev, W_bev)
        )

        # new values and the ones not touched from scenario
        data['gt_dynamic'] = gt_dynamic_rot
        data['gt_static'] = gt_static_rot
        data['inputs'] = inputs_rot
        data['inputs_ego'] = inputs_ego_rot

        return data


class ProposalBEVVehicleToBackgroundAugmentor(BaseProposalAugmentor):
    def __init__(self, apply_probability: float = 1.0, rect_size: list = [0.5, 0.5]) -> None:
        super().__init__(apply_probability)
        assert len(rect_size) == 2, 'rect_size must be a list of length 2'
        assert rect_size[0] > 0.0 and rect_size[0] <= 1.0, 'rect_size[0] must be in range (0.0, 1.0]'
        assert rect_size[1] > 0.0 and rect_size[1] <= 1.0, 'rect_size[1] must be in range (0.0, 1.0]'

        self.rect_size_percentage = rect_size
    
    def _augment_fusion(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        # we only augment the last frame
        # if not ('grad_mode' in kwargs and kwargs['grad_mode']):
        #     return data

        scenario = kwargs['scenario']
        dynamic_gt = scenario['gt_dynamic']

        C_emb, H_emb, W_emb = data.shape[-3:]
        scaled_dynamic_gt = F.interpolate(dynamic_gt.to(torch.float), size=(H_emb, W_emb), mode='bilinear')

        background_mask = scaled_dynamic_gt == 0
 
        # Step 1: Use scaled_gt_map as a mask over all channels of data
        # shape of data and scaled_gt_map is [BS, C, H, W]
        masked_data = data * background_mask

        filter_size = (int(H_emb * self.rect_size_percentage[0]), int(W_emb * self.rect_size_percentage[1]))
        # Step 2: Only keep values unequal to zero (not a boolean mask)
        for bs in range(masked_data.shape[0]):
            nonzero_values = masked_data[bs, :, background_mask[bs,0]]
            # mean for each channel
            mean = nonzero_values.mean(dim=1)
            std = nonzero_values.std(dim=1)

            # Create a normal distribution from the mean and std of masked_data (keep channel dim)
            # the normal distribution should also keep the batch dimension and the channel dimension
            normal_distribution = torch.distributions.normal.Normal(mean, std)

            # Create a filter of size (H_emb * rect_size_percentage[0], W_emb * rect_size_percentage[1])
            h_idx = torch.randint(0, H_emb - filter_size[0] + 1, (data.shape[0],))
            w_idx = torch.randint(0, W_emb - filter_size[1] + 1, (data.shape[0],))

            # Replace masked_data with sampled values at a random location
            data[bs, :, h_idx[bs]:h_idx[bs] + filter_size[0], w_idx[bs]:w_idx[bs] + filter_size[1]] = normal_distribution.sample((1,) + filter_size).permute(0, 3, 1, 2)
            # for c in range(masked_data.shape[1]):
            #     # find all nonzero values in the spatial dimensions
            #     nonzero_values = masked_data[bs, c, background_mask[bs,0]]
                
            #     # Create a normal distribution from the mean and std of masked_data (keep channel dim)
            #     # the normal distribution should also keep the batch dimension and the channel dimension
            #     mean = nonzero_values.mean()
            #     std = nonzero_values.std()
            #     normal_distribution = torch.distributions.normal.Normal(mean, std)

            #     # Create a filter of size (H_emb * rect_size_percentage[0], W_emb * rect_size_percentage[1])
            #     h_idx = torch.randint(0, H_emb - filter_size[0] + 1, (data.shape[0],))
            #     w_idx = torch.randint(0, W_emb - filter_size[1] + 1, (data.shape[0],))

            #     # Replace masked_data with sampled values at a random location
            #     data[bs, c, h_idx[bs]:h_idx[bs] + filter_size[0], w_idx[bs]:w_idx[bs] + filter_size[1]] = normal_distribution.sample((1, 1) + filter_size)

        return data
