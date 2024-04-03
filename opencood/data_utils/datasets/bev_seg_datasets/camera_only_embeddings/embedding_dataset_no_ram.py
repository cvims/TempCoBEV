import os
import pickle

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from opencood.data_utils.datasets.bev_seg_datasets.camera_only_embeddings import BaseScenarioEmbeddingsDataset


class ScenarioEmbeddingsDatasetNoRAM(BaseScenarioEmbeddingsDataset):
    def __init__(self, params, train=True, validate=False, **kwargs):
        super().__init__(params, train, validate, **kwargs)

        all_files = os.listdir(self.root_dir)
        # order files by number
        all_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # filter those that are not pkl files; o dict with id as key
        self.all_files = {int(''.join(filter(str.isdigit, f))): f for f in all_files if f.endswith(".pkl")}


    def _load_data(self, scenario_index, timestamp_indices):
        # get the start idx from the len_record
        start_idx = self.len_record[scenario_index - 1] if scenario_index > 0 else 0
        # get min and span of timestamp indices
        min_idx = min(timestamp_indices)
        span = max(timestamp_indices) - min_idx + 1

        # load data from start_idx + min to start_idx + min + span
        data = []
        data_ego = []
        dynamic_gts = []
        static_gts = []
        vehicle_offsets = []
        
        for i in range(min_idx, min_idx + span):
            correct_idx = start_idx + i
            file_name = self.all_files[correct_idx]
            with open(os.path.join(self.root_dir, file_name), 'rb') as f:
                pkl_data = pickle.load(f)
                if self.use_ego_only:
                    data.append(pkl_data['bev_embedding_ego'])
                else:
                    data.append(pkl_data['bev_embedding'])
                if 'bev_embedding_ego' in pkl_data:
                    data_ego.append(pkl_data['bev_embedding_ego'])
                else:
                    data_ego.append(data[-1])
                if self.use_last_frame_full_view:
                    dynamic_gts.append(pkl_data['gt_dynamic'])
                else:
                    dynamic_gts.append(pkl_data['gt_nofull'])
                if isinstance(pkl_data['gt_static'], torch.Tensor):
                    static_gts.append(pkl_data['gt_static'].cpu().numpy())
                else:
                    static_gts.append(pkl_data['gt_static'])
                v_offsets = {}
                for cav_id in pkl_data['vehicle_offsets']:
                    if isinstance(pkl_data['vehicle_offsets'][cav_id], torch.Tensor):
                        v_offsets.update(
                            {cav_id: np.asarray(pkl_data['vehicle_offsets'][cav_id].cpu())}
                        )
                    else:
                        v_offsets.update(
                            {cav_id: np.asarray(pkl_data['vehicle_offsets'][cav_id])}
                        )
                vehicle_offsets.append(v_offsets)
        
        return dict(
            inputs=data,
            inputs_ego=data_ego,
            gt_dynamic=dynamic_gts,
            gt_static=static_gts,
            vehicle_offsets=vehicle_offsets
        )



    def __getitem__(self, idx):
        inputs_scenarios = []
        inputs_ego_scenarios = []
        vehicle_offsets_scenarios = []
        gt_dynamic_scenarios = []
        gt_static_scenarios = []

        scenario_index, timestamp_indices = self._retrieve_timestamps(idx)
        data = self._load_data(scenario_index, timestamp_indices)
        # rescale timestamps indices, starting from 0
        timestamp_indices = [idx - min(timestamp_indices) for idx in timestamp_indices]

        _inputs = data['inputs']
        _inputs_ego = data['inputs_ego']
        dynamic_gts = data['gt_dynamic']
        static_gts = data['gt_static']
        vehicle_offsets = data['vehicle_offsets']

        for timestamp_index in timestamp_indices:
            inputs_scenarios.append(_inputs[timestamp_index])
            inputs_ego_scenarios.append(_inputs_ego[timestamp_index])
            gt_dynamic_scenarios.append(dynamic_gts[timestamp_index])
            gt_static_scenarios.append(static_gts[timestamp_index])
        
        vehicle_offsets_scenarios = self.correct_vehicle_offsets(scenario_index, timestamp_indices, vehicle_offsets)
        
        ret_dict = dict(
            inputs=np.stack(inputs_scenarios),
            inputs_ego=np.stack(inputs_ego_scenarios),
            vehicle_offsets=vehicle_offsets_scenarios,
            gt_dynamic=np.expand_dims(np.stack(gt_dynamic_scenarios).astype(np.int64), axis=1),
            gt_static=np.expand_dims(np.stack(gt_static_scenarios).astype(np.int64), axis=1)
        )

        if self.proposal_augmentor is not None:
            ret_dict.update(self.proposal_augmentor(ret_dict))
        
        return ret_dict



if __name__ == "__main__":
    mp.set_start_method('spawn')

    params = dict(
        root_dir=r'/data/public_datasets/OPV2V/embeddings/cobevt/test',
        validate_dir=r'/data/public_datasets/OPV2V/embeddings/cobevt/test',
        fusion=dict(
            args=dict(
                queue_length=4,
                timestamp_offset=0,
                timestamp_offset_mean=-1,
                timestamp_offset_std=-1
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
    custom_data = ScenarioEmbeddingsDatasetNoRAM(params=params, train=True, validate=False)

    # Create a PyTorch DataLoader
    batch_size = 8
    data_loader = DataLoader(
        custom_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True,
        num_workers=0, collate_fn=custom_data.collate_batch)

    # Accessing the data
    for _dict in tqdm(data_loader):
        # 'data' contains a batch of random tensors of shape (batch_size, 128, 32, 32)
        # 'offset' contains corresponding vehicle offsets of shape (batch_size, 1, 4, 4)
        # Perform operations using the data and offset here
        # data = _dict['inputs']
        # offset = _dict['vehicle_offsets']
        # gt_dynamic = _dict['gt_dynamic']
        pass
