import os
import pickle

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from opencood.data_utils.datasets.bev_seg_datasets.camera_only_embeddings import BaseScenarioEmbeddingsDataset


class ScenarioEmbeddingsDataset(BaseScenarioEmbeddingsDataset):
    def __init__(self, params, train=True, validate=False, **kwargs):
        super().__init__(params, train, validate, **kwargs)

        if 'use_ego_only' in params['fusion']['args'] and params['fusion']['args']['use_ego_only']:
            print('Attention: You are in ego only mode for pre-loading embeddings!')
            self._preload_data(self.root_dir, ego_only=True)
        else:
            self._preload_data(self.root_dir, ego_only=False)

    
    def _preload_data(self, path: str, ego_only: bool = False):
        all_files = os.listdir(path)
        # order files by number
        all_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for file in all_files:
            if file.endswith(".pkl"):
                with open(os.path.join(path, file), 'rb') as f:
                    data = pickle.load(f)
                    # id from file name
                    idx = int(''.join(filter(str.isdigit, file)))
                    # check which scenario it belongs to
                    for i, ele in enumerate(self.len_record):
                        if idx < ele:
                            if ego_only:
                                self.data[i].append(data['bev_embedding_ego'])
                            else:
                                self.data[i].append(data['bev_embedding'])                                
                            if 'bev_embedding_ego' in data:
                                self.data_ego[i].append(data['bev_embedding_ego'])
                            else:
                                self.data_ego[i].append(self.data[i][-1])
                            if self.use_last_frame_full_view:
                                self.dynamic_gts[i].append(data['gt_dynamic'])
                            else:
                                self.dynamic_gts[i].append(data['gt_nofull'])
                            if isinstance(data['gt_static'], torch.Tensor):
                                self.static_gts[i].append(np.asarray(data['gt_static'].cpu()))
                            else:
                                self.static_gts[i].append(data['gt_static'])
                            v_offsets = {}
                            for cav_id in data['vehicle_offsets']:
                                if isinstance(data['vehicle_offsets'][0][cav_id], torch.Tensor):
                                    v_offsets.update(
                                        {cav_id: np.asarray(data['vehicle_offsets'][cav_id].cpu())}
                                    )
                                else:
                                    v_offsets.update(
                                        {cav_id: np.asarray(data['vehicle_offsets'][cav_id])}
                                    )
                            self.vehicle_offsets[i].append(v_offsets)
                            break


    def __getitem__(self, idx):
        inputs_scenarios = []
        inputs_ego_scenarios = []
        vehicle_offsets_scenarios = []
        gt_dynamic_scenarios = []
        gt_static_scenarios = []

        scenario_index, timestamp_indices = self._retrieve_timestamps(idx)

        data = self.data[scenario_index]
        data_ego = self.data_ego[scenario_index]
        dynamic_gts = self.dynamic_gts[scenario_index]
        static_gts = self.static_gts[scenario_index]
        
        for timestamp_index in timestamp_indices:
            inputs_scenarios.append(data[timestamp_index])
            inputs_ego_scenarios.append(data_ego[timestamp_index])
            gt_dynamic_scenarios.append(dynamic_gts[timestamp_index])
            gt_static_scenarios.append(static_gts[timestamp_index])
        
        vehicle_offsets_scenarios = self.correct_vehicle_offsets(scenario_index, timestamp_indices)

        ret_dict = dict(
            inputs=np.stack(inputs_scenarios),
            inputs_ego=np.stack(inputs_ego_scenarios),
            vehicle_offsets=vehicle_offsets_scenarios,
            gt_dynamic=np.expand_dims(np.stack(gt_dynamic_scenarios).astype(np.int64), axis=1),
            gt_static=np.expand_dims(np.stack(gt_static_scenarios).astype(np.int64), axis=1)
        )

        if self.proposal_augmentor is not None:
            inputs = self.proposal_augmentor(ret_dict['inputs'], scenario=ret_dict)
            ret_dict['inputs'] = inputs
        
        return ret_dict




if __name__ == "__main__":
    params = dict(
        root_dir=r'/data/public_datasets/OPV2V/embeddings/cobevt/test',
        validate_dir=r'/data/public_datasets/OPV2V/embeddings/cobevt/test',
        fusion=dict(
            args=dict(
                queue_length=4,
                timestamp_offset=1,
                timestamp_offset_mean=0,
                timestamp_offset_std=0
            )
        ),
        train_params=dict(
            use_last_frame_full_view=False
        ),
        model=dict(
            args=dict()
        )
    )

    # Create an instance of the ScenarioEmbeddingsDataset
    custom_data = ScenarioEmbeddingsDataset(params=params, train=True, validate=False)

    # Create a PyTorch DataLoader
    batch_size = 8
    data_loader = DataLoader(
        custom_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True,
        collate_fn=custom_data.collate_batch)

    # Accessing the data
    for _dict in tqdm(data_loader):
        # 'data' contains a batch of random tensors of shape (batch_size, 128, 32, 32)
        # 'offset' contains corresponding vehicle offsets of shape (batch_size, 1, 4, 4)
        # Perform operations using the data and offset here
        # data = _dict['inputs']
        # offset = _dict['vehicle_offsets']
        # gt_dynamic = _dict['gt_dynamic']
        pass
