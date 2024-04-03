import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.utils.data import Dataset
from opencood.data_utils.post_processor import build_postprocessor

import torch
import numpy as np


class BaseScenarioEmbeddingsDataset(Dataset):
    def __init__(self, params, train=True, validate=False, **kwargs) -> None:
        super().__init__()
        if 'postprocess' in params:
            self.post_processor = build_postprocessor(params['postprocess'], train)
        else:
            self.post_processor = None

        if 'queue_length' in params['fusion']['args']:
            self.queue_length = max(1, params['fusion']['args']['queue_length'])
        else:
            self.queue_length = 1
        
        if 'timestamp_offset' in params['fusion']['args']:
            self.timestamp_offset = max(0, params['fusion']['args']['timestamp_offset'])
        else:
            self.timestamp_offset = 0
        
        if 'timestamp_offset_mean' in params['fusion']['args']:
            self.timestamp_offset_mean = params['fusion']['args']['timestamp_offset_mean']
            self.timestamp_offset_std = params['fusion']['args']['timestamp_offset_std']
        else:
            self.timestamp_offset_mean = 0
            self.timestamp_offset_std = 0

        self.use_last_frame_full_view = params['train_params']['use_last_frame_full_view'] if 'use_last_frame_full_view' in params['train_params'] else False

        if train and not validate:
            root_dir = params['root_dir']
            self.is_train = True
        else:
            root_dir = params['validate_dir']
            self.is_train = False          

        if root_dir.endswith('train'):
            self.len_record = self._get_train_record_list()
        elif root_dir.endswith('validate'):
            self.len_record = self._get_validate_record_list()
        elif root_dir.endswith('test'):
            self.len_record = self._get_test_record_list()
        else:
            raise ValueError(f'Unknown root dir: {root_dir}')

        self.data = [[] for _ in range(len(self.len_record))]
        self.data_ego = [[] for _ in range(len(self.len_record))]
        self.dynamic_gts = [[] for _ in range(len(self.len_record))]
        self.static_gts = [[] for _ in range(len(self.len_record))]
        self.vehicle_offsets = [[] for _ in range(len(self.len_record))]

        # build proposal augmentor
        from opencood.tools.training_temporal.train_utils import setup_augmentation
        augmentors = setup_augmentation(params)
        self.proposal_augmentor = augmentors['proposal_augmentors'] if 'proposal_augmentors' in augmentors and augmentors['proposal_augmentors'] else None

        self.use_ego_only = False

        if 'use_ego_only' in params['fusion']['args'] and params['fusion']['args']['use_ego_only']:
            print('Attention: You are in ego only mode for pre-loading embeddings!')
            self.use_ego_only = True
    
        self.root_dir = root_dir

    def __len__(self):
        return self.len_record[-1]

    def correct_vehicle_offsets(self, scenario_index: int, timestamp_indices: list, vehicle_offsets = None):
        if vehicle_offsets is None:
            vehicle_offsets_scenario = self.vehicle_offsets[scenario_index]

            # Get vehicle offsets from min to max timestamp index
            min_index = min(timestamp_indices)
            max_index = max(timestamp_indices)
            # copy the vehicle offsets scenario to not change the original one
            vehicle_offsets = [{cav_id: np.copy(vehicle_offsets_scenario[i][cav_id]) for cav_id in vehicle_offsets_scenario[i]} for i in range(min_index, max_index + 1)]

        if not self.is_train:
            return vehicle_offsets

        _cav_ids = list(vehicle_offsets[0].keys())

        # Set the first vehicle offsets of index 0 to the identity matrix
        vehicle_offsets[0] = {_id: np.eye(4) for _id in vehicle_offsets[0].keys()}

        # Calculate the offset between all timestamps from min to max
        for i in range(1, len(vehicle_offsets)):
            current_cav_ids = list(vehicle_offsets[i].keys())
            # check if all current cav ids are in the previous vehicle offsets
            for cav_id in current_cav_ids:
                if cav_id not in vehicle_offsets[i - 1]:
                    vehicle_offsets[i - 1][cav_id] = np.eye(4)
                if cav_id not in _cav_ids:
                    _cav_ids.append(cav_id)

            for cav_id in _cav_ids:
                if cav_id not in current_cav_ids:
                    vehicle_offsets[i][cav_id] = np.eye(4)
                else:
                    vehicle_offsets[i][cav_id] = vehicle_offsets[i][cav_id] @ np.linalg.pinv(vehicle_offsets[i - 1][cav_id])

        # Pick the vehicle offsets defined by timestamp indices
        corrected_offsets = [vehicle_offsets[i] for i in range(len(timestamp_indices))]

        return corrected_offsets


    def _retrieve_timestamps(self, idx: int):
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                _len_record = self.len_record[i - 1] if i > 0 else 0
                scenario_index = i
                timestamp_index = idx if i == 0 else idx - _len_record
                max_idx = self.len_record[i] - _len_record - 1
                break
        
        if self.timestamp_offset_mean > 0 and self.timestamp_offset_std > 0:
            timestamp_offset = max(0, int(np.random.normal(self.timestamp_offset_mean, self.timestamp_offset_std)))
        else:
            timestamp_offset = self.timestamp_offset
        
        # keep the timestamp indices between min_idx and max_idx
        span = (self.queue_length - 1) * timestamp_offset + self.queue_length

        if span > max_idx:
            timestamp_index = 0
            timestamp_offset = (max_idx - self.queue_length) // (self.queue_length - 1)
            span = (self.queue_length - 1) * timestamp_offset + self.queue_length

        # check if its in between min and max idx
        if span > 1:
            if (timestamp_index + span) > max_idx:
                timestamp_index = max(0, timestamp_index - span)

        timestamp_indices = np.array(
            [timestamp_index + i * (timestamp_offset + 1) for i in range(self.queue_length)]
        )

        return scenario_index, timestamp_indices


    def reinitialize(self):
        pass

    def has_future_frame(self, idx: int):
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                max_idx = self.len_record[i] - 1
                if idx < max_idx:
                    return True
                else:
                    return False


    def _get_train_record_list(self):
        data = {
            "00": 419,
            "01": 493,
            "02": 572,
            "03": 694,
            "04": 936,
            "05": 1203,
            "06": 1302,
            "07": 1350,
            "08": 1477,
            "09": 1587,
            "10": 1691,
            "11": 1865,
            "12": 2027,
            "13": 2221,
            "14": 2366,
            "15": 2491,
            "16": 2625,
            "17": 2806,
            "18": 2996,
            "19": 3156,
            "20": 3244,
            "21": 3346,
            "22": 3473,
            "23": 3564,
            "24": 3660,
            "25": 3811,
            "26": 3875,
            "27": 4031,
            "28": 4113,
            "29": 4181,
            "30": 4272,
            "31": 4358,
            "32": 4427,
            "33": 4515,
            "34": 4628,
            "35": 4731,
            "36": 4805,
            "37": 4925,
            "38": 5255,
            "39": 5386,
            "40": 5540,
            "41": 5991,
            "42": 6371
        }

        values_list = list(data.values())
        
        return values_list


    def _get_validate_record_list(self):
        data = {
            "00": 112,
            "01": 269,
            "02": 404,
            "03": 606,
            "04": 670,
            "05": 718,
            "06": 775,
            "07": 1234,
            "08": 1980
        }

        values_list = list(data.values())
        return values_list
    

    def _get_test_record_list(self):
        data = {
            "00": 178,
            "01": 276,
            "02": 423,
            "03": 590,
            "04": 741,
            "05": 812,
            "06": 916,
            "07": 1083,
            "08": 1217,
            "09": 1319,
            "10": 1460,
            "11": 1537,
            "12": 1672,
            "13": 1776,
            "14": 1996,
            "15": 2170
        }

        values_list = list(data.values())
        return values_list


    def post_process(self, batch_dict, output_dict):
        if self.post_processor is not None:
            output_dict = self.post_processor.post_process(
                batch_dict, output_dict)

        return output_dict


    def collate_batch(self, batch):
        bs = len(batch)
        scneario_length = len(batch[0]['inputs'])

        b_inputs = []
        b_inputs_ego = []
        b_vehicle_offsets = []
        b_gt_dynamic = []
        b_gt_static = []
        # b_record_len = []

        for b in range(bs):
            s_inputs = []
            s_inputs_ego = []
            s_vehicle_offsets = []
            s_gt_dynamic = []
            s_gt_static = []
            # s_record_len = []

            for s in range(scneario_length):
                s_inputs.append(batch[b]['inputs'][s])
                s_inputs_ego.append(batch[b]['inputs_ego'][s])
                s_vehicle_offsets.append(batch[b]['vehicle_offsets'][s])
                s_gt_dynamic.append(batch[b]['gt_dynamic'][s])
                s_gt_static.append(batch[b]['gt_static'][s])
                # s_record_len.append(torch.as_tensor(len(batch[b]['vehicle_offsets'][s].keys())))

            b_inputs.append(torch.from_numpy(np.stack(s_inputs)))
            b_inputs_ego.append(torch.from_numpy(np.stack(s_inputs_ego)))
            b_vehicle_offsets.append(s_vehicle_offsets)
            b_gt_dynamic.append(torch.from_numpy(np.stack(s_gt_dynamic)))
            b_gt_static.append(torch.from_numpy(np.stack(s_gt_static)))
            # b_record_len.append(s_record_len)
        
        # reformat everything such that the batch size is the second dimension and scneario length is the first
        # now we have lists of [BS, Scenario_length, ...]
        # reformat them such that we have [Scenario_length, BS, tensors]
        b_inputs = list(map(list, zip(*b_inputs)))
        # now combine the batch size with the first dimension of tensors
        b_inputs = [torch.stack(_input, dim=0) for _input in b_inputs]

        b_inputs_ego = list(map(list, zip(*b_inputs_ego)))
        b_inputs_ego = [torch.stack(_input, dim=0) for _input in b_inputs_ego]

        b_gt_dynamic = list(map(list, zip(*b_gt_dynamic)))
        b_gt_dynamic = [torch.stack(_gt) for _gt in b_gt_dynamic]

        b_gt_static = list(map(list, zip(*b_gt_static)))
        b_gt_static = [torch.stack(_gt) for _gt in b_gt_static]

        b_vehicle_offsets = list(map(list, zip(*b_vehicle_offsets)))
        
        return dict(
            inputs=b_inputs,
            inputs_ego=b_inputs_ego,
            vehicle_offsets=b_vehicle_offsets,
            gt_dynamic=b_gt_dynamic,
            gt_static=b_gt_static,
            # record_len=b_record_len
        )
