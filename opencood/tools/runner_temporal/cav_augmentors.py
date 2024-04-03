import torch
from opencood.tools.runner_temporal import BaseCAVAugmentor

from typing import Any


class TemporalCameraInputDropAllButEgoVehicleAugmentor(BaseCAVAugmentor):
    def __init__(
            self,
            apply_probability: float = 0.0,
            drop_every_x: int = 0,
            drop_mode: str = 'static'
    ) -> None:
        super().__init__(apply_probability)
        self.use_probability_application = False
        self.drop_mode = drop_mode

        if apply_probability > 0.0:
            self.use_probability_application = True
            drop_every_x = 0
        
        if drop_every_x > 0:
            self.use_probability_application = False
            apply_probability = 0.0

        self.drop_every_x = drop_every_x

        self.internal_counter = 0
    
    def forward(self, data: dict) -> Any:
        if self.drop_every_x > 0 or self.apply_probability > 0:
            return self._augment_data(data)
        else:
            return data

    def _augment_data(self, data: dict) -> dict:
        if self.use_probability_application:
            if torch.rand(1) < self.apply_probability:
                data = self._drop_cavs(data)
        else:
            if self.internal_counter % self.drop_every_x == 0 and self.internal_counter > 0:
                data = self._drop_cavs(data)
        
        self.internal_counter += 1
        
        return data

    def _drop_cavs(self, data: dict) -> dict:
        return drop_cavs_static(data, drop_count=999, keep_at_least=1, mode=self.drop_mode)


class TemporalCameraInputDropAllButEgoForXFramesAugmentor(BaseCAVAugmentor):
    def __init__(
            self, 
            drop_for_x_frames: int,
            neutral_frames: int,
            drop_mode: str = 'static'
    ) -> None:
        """
        Args:
            drop_for_x_frames: int
                drop all but ego vehicle for x frames
            neutral_frames: int
                number of frames to keep before dropping
        """
        super().__init__()
        self.drop_for_x_frames = drop_for_x_frames
        self.neutral_frames = neutral_frames
        self.drop_mode = drop_mode

        self._drop_count = self.drop_for_x_frames
        self._neutral_count = self.neutral_frames
    
    def reset_dropping(self):
        self._drop_count = self.drop_for_x_frames
        self._neutral_count = self.neutral_frames
    
    def _augment_data(self, data: dict) -> dict:
        if self._neutral_count > 0:
            # do not drop
            self._neutral_count -= 1
        elif self._drop_count > 0:
            # drop
            data = self._drop_cavs(data)
            self._drop_count -= 1
        else:
            # reset
            self.reset_dropping()
        
        return data

    def _drop_cavs(self, data: dict) -> dict:
        return drop_cavs_static(data, drop_count=999, keep_at_least=1, mode=self.drop_mode)


class TemporalCameraInputDropRandomButEgoVehicleAugmentor(BaseCAVAugmentor):
    """
    Drops a random count of cavs but keeps the ego vehicle
    """
    def __init__(
            self,
            apply_probability: float = 0.0,
            drop_every_x: int = 0,
            drop_mode: str = 'static'
    ) -> None:
        super().__init__(apply_probability)
        self.use_probability_application = False
        self.drop_mode = drop_mode

        if apply_probability > 0.0:
            self.use_probability_application = True
            drop_every_x = 0
        
        if drop_every_x > 0:
            self.use_probability_application = False
            apply_probability = 0.0

        self.drop_every_x = drop_every_x

        self.internal_counter = 0
    
    def forward(self, data: dict) -> Any:
        if self.drop_every_x > 0 or self.apply_probability > 0:
            return self._augment_data(data)
        else:
            return data

    def _augment_data(self, data: dict) -> dict:
        if self.use_probability_application:
            if torch.rand(1) < self.apply_probability:
                data = self._drop_cavs(data)
        else:
            if self.internal_counter % self.drop_every_x == 0 and self.internal_counter > 0:
                data = self._drop_cavs(data)
        
        self.internal_counter += 1
        
        return data

    def _drop_cavs(self, data: dict) -> dict:
        # shape: Scenario, CAVs, ...
        # We use the first batch and scenario and the number of cavs involved there
        max_drop_count = max(1, len(data['inputs'][0]) - 1)
        # random number between 1 and max_drop_count
        if max_drop_count > 1:
            drop_count = torch.randint(1, max_drop_count + 1, (1,)).item()
        else:
            drop_count = 1

        return drop_cavs_static(data, drop_count=drop_count, keep_at_least=1, mode=self.drop_mode)


def drop_cavs_static(scenarios, drop_count=1, keep_at_least=1, mode='static'):
    """
    Only for the last scenario
    mode: static or random
    """
    assert mode in ['static', 'random'], 'mode must be static or random'

    if mode == 'random':
        raise NotImplementedError('random mode not implemented yet')

    keep_at_least = max(1, keep_at_least)

    # scenarios of shape dict of list of tensors
    # [scenario1, scenario2, ...]
    # scenario1: dict of tensors
    # dict of tensors of shape [batch_size, tensor_shape]

    new_scenarios = dict(
        inputs = [],
        extrinsic = [],
        intrinsic = [],
        gt_static = [],
        gt_dynamic = [],
        gt_dynamic_non_corp = [],
        cav_ids = [],
        record_len = [],
        transformation_matrix = [],
        pairwise_t_matrix = [],
        vehicle_offsets = [],
    )
 
    # the dict contains keys with shape [Scenarios, CAVs, ...]
    for s in range(len(scenarios['inputs'])):  # scenarios per batch
        scenario = {key: scenarios[key][s] for key in scenarios.keys()}
        batch_size = len(scenario['record_len'])

        new_batches = dict(
            inputs = [],
            extrinsic = [],
            intrinsic = [],
            gt_static = [],
            gt_dynamic = [],
            gt_dynamic_non_corp = [],
            cav_ids = [],
            record_len = [],
            transformation_matrix = [],
            pairwise_t_matrix = [],
            vehicle_offsets = [],
        )

        for b in range(batch_size):
            # batch size is combined at first dimension
            record_len = scenario['record_len'][b]
            data = dict(
                inputs = scenario['inputs'][:record_len],
                extrinsic = scenario['extrinsic'][:record_len],
                intrinsic = scenario['intrinsic'][:record_len],
                gt_static = scenario['gt_static'][b:b+1],
                gt_dynamic = scenario['gt_dynamic'][b:b+1],
                gt_dynamic_non_corp = scenario['gt_dynamic_non_corp'][b:b+1],
                cav_ids = scenario['cav_ids'][b:b+1],
                record_len = scenario['record_len'][b:b+1],
                transformation_matrix = scenario['transformation_matrix'][b:b+1],
                pairwise_t_matrix = scenario['pairwise_t_matrix'][b:b+1],
                vehicle_offsets = scenario['vehicle_offsets'][b],
            )

            # we always keep the index 0 (ego vehicle)
            max_drop = record_len - keep_at_least
            if drop_count > max_drop:
                drop_count = max_drop
            
            if drop_count > 0:
                if mode == 'static':
                    new_batch_incl_entries = {
                        key: data[key][:-drop_count] if len(data[key]) > 1 else data[key][:1]
                        for key in ['inputs', 'extrinsic', 'intrinsic']
                    }
                    new_batch_excl_entries = {
                        key: data[key]
                        for key in ['gt_static', 'gt_dynamic', 'gt_dynamic_non_corp', 'transformation_matrix', 'pairwise_t_matrix']
                    }

                    # cav ids
                    new_batch_excl_entries['cav_ids'] = data['cav_ids'][0][:-drop_count] if len(data['cav_ids'][0]) > 1 else data['cav_ids'][0][:1]

                    # record length
                    new_batch_excl_entries['record_len'] = data['record_len'] - drop_count
            
                    # vehicle offsets
                    # drop the last cavs from vehicle offsets dict
                    v_keys = list(data['vehicle_offsets'].keys())
                    for key in v_keys[-drop_count:]:
                        del data['vehicle_offsets'][key]
        
                elif mode == 'random':
                    pass
            
                for key in new_batch_incl_entries.keys():
                    data[key] = new_batch_incl_entries[key]
                
                for key in new_batch_excl_entries.keys():
                    data[key] = new_batch_excl_entries[key]
            


            # put back everything into the new_scenarios dict
            for key in new_scenarios.keys():
                new_batches[key].append(data[key])


        # cat inputs, extrinsic, intrinsic
        for key in ['inputs', 'extrinsic', 'intrinsic']:
            new_scenarios[key].append(torch.cat(new_batches[key], dim=0))
        
        # stack gt_static, gt_dynamic, gt_dynamic_non_corp, cav_ids, transformation_matrix, pairwise_t_matrix
        for key in ['gt_static', 'gt_dynamic', 'gt_dynamic_non_corp', 'transformation_matrix', 'pairwise_t_matrix']:
            new_scenarios[key].append(torch.cat(new_batches[key], dim=0))

        # cav ids
        new_scenarios['cav_ids'].append(new_batches['cav_ids'])

        # record len
        new_scenarios['record_len'].append(torch.cat(new_batches['record_len']))
        
        # vehicle offsets
        new_scenarios['vehicle_offsets'].append(new_batches['vehicle_offsets'])



        # data = dict(
        #     inputs = scenario['inputs'],
        #     extrinsic = scenario['extrinsic'] if 'extrinsic' in scenario.keys() else None,
        #     intrinsic = scenario['intrinsic'] if 'intrinsic' in scenario.keys() else None,
        #     # vehicle_offsets = scenario['vehicle_offsets'][b][s],
        #     gt_static = scenario['gt_static'],
        #     gt_dynamic = scenario['gt_dynamic'],
        #     gt_dynamic_non_corp = scenario['gt_dynamic_non_corp'] if 'gt_dynamic_non_corp' in scenario.keys() else None,
        #     # transformation_matrix = scenario['transformation_matrix'][s],
        #     # pairwise_t_matrix = scenario['pairwise_t_matrix'][s],
        #     # record_len = scenario['record_len'][s]
        #     cav_ids = scenario['cav_ids'] if 'cav_ids' in scenario.keys() else None,
        # )

        # # we always keep the index 0 (ego vehicle)
        # max_drop = len_cavs - keep_at_least
        # if drop_count > max_drop:
        #     drop_count = max_drop
        
        # # delete data entries with None values
        # for key in list(data.keys()):
        #     if data[key] is None:
        #         del data[key]        

        # # for now we only drop the last cavs
        # if drop_count > 0:
        #     if mode == 'static':
        #         new_entries = {
        #             key: scenario[key][:-drop_count] if len(scenario[key]) > 1 else scenario[key][:1]
        #             for key in data.keys()
        #         }
        #     elif mode == 'random':
        #         pass

        #     for key in new_entries.keys():
        #         scenario[key] = new_entries[key]

        #     # record length
        #     scenario['record_len'] = scenario['record_len'] - drop_count
        
        # # update batch
        # for key in list(scenarios.keys()):
        #     scenarios[key][s] = scenario[key]
    
    return new_scenarios
