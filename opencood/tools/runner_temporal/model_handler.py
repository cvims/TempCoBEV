from opencood.tools.runner_temporal import ScenarioCameraModelHandler, ScenarioLidarModelHandler, \
    TemporalModelHandler, BaseCAVAugmentor, BaseProposalAugmentor, BaseHistoryBEVEmbeddingAugmentor, \
    get_camera_data_batch_size, get_camera_data_scenario_length, \
    get_lidar_data_batch_size, get_lidar_data_scenario_length
from typing import Tuple, Any
from opencood.tools.training_temporal.train_utils import to_device
from opencood.models import TemporalFusionModel


class CooperativeCameraModelHandler(ScenarioCameraModelHandler):
    def __init__(self, hypes: dict, data_record_len: list, max_frames: int, cav_augmentor: BaseCAVAugmentor) -> None:
        super().__init__(hypes, data_record_len, max_frames, cav_augmentor)
    
    def run(self, model, data, device, iteration, dataset_postprocessor, **kwargs) -> Tuple[Any, Any]:
        return self.default_eval_run_wrapper(
            model=model, data=data, device=device, iteration=iteration,
            dataset_postprocessor=dataset_postprocessor, **kwargs
        )


class CooperativeLidarModelHandler(ScenarioLidarModelHandler):
    def __init__(self, hypes: dict, data_record_len: list, max_frames: int, cav_augmentor: BaseCAVAugmentor = None) -> None:
        super().__init__(hypes, data_record_len, max_frames, cav_augmentor)
    
    def run(self, model, data, device, iteration, dataset_postprocessor, **kwargs) -> Tuple[Any, Any]:
        return self.default_eval_run_wrapper(
            model=model, data=data, device=device, iteration=iteration,
            dataset_postprocessor=dataset_postprocessor, **kwargs
        )


class TemporalCameraModelHandler(TemporalModelHandler):
    def __init__(self, hypes: dict, data_record_len: list, max_frames: int = 5,
                bev_augmentor: BaseHistoryBEVEmbeddingAugmentor = None,
                cav_augmentor: BaseCAVAugmentor = None,
                proposal_augmentor: BaseProposalAugmentor = None) -> None:
        super().__init__(hypes, data_record_len, max_frames, bev_augmentor, cav_augmentor, proposal_augmentor)
    
    def _temporal_model_exec_fn(self, model: TemporalFusionModel, scenario, **kwargs):
        device = kwargs['device'] if 'device' in kwargs else 'cuda'
        cooperative_bev_embeddings = self.previous_cooperative_bev_embeddings
        cooperative_bev_embeddings = to_device(cooperative_bev_embeddings, device)

        if self.previous_cooperative_bev_embeddings is not None and len(self.previous_cooperative_bev_embeddings) > 0:
            self.previous_vehicle_offsets.append(scenario['vehicle_offsets'])

        fusion_output, _kwargs = model.until_fusion(
            scenario,
            **kwargs
        )

        kwargs.update(_kwargs)

        # augment fusion output (this is the proposal from the current frame)
        if self.bev_proposal_augmentor:
            if 'ignore_bev_proposal_augmentor' in kwargs and kwargs['ignore_bev_proposal_augmentor']:
                pass
            else:
                fusion_output = self.bev_proposal_augmentor._augment(
                    fusion_output, scenario=scenario, **kwargs
                )
        
        kwargs['vehicle_offsets'] = self.previous_vehicle_offsets

        temporal_fusion_output = model.temporal_fusion(
            fusion_output,
            history_cooperative_bev_embeddings=cooperative_bev_embeddings,
            **kwargs
        )

        prediction = model.from_fusion(
            temporal_fusion_output,
            **kwargs
        )

        return prediction

    def default_eval_run_wrapper(self, model, data, device, iteration, dataset_postprocessor, **kwargs) -> Tuple[Any, Any]:
        bs = get_camera_data_batch_size(data)
        scenario_length = get_camera_data_scenario_length(data)
        
        return super().default_eval_run_wrapper(
            batch_size=bs, scenario_length=scenario_length,
            model=model, data=data, device=device, iteration=iteration,
            dataset_postprocessor=dataset_postprocessor, model_exec_fn=self._temporal_model_exec_fn,
            **kwargs
        )

    def default_train_run_wrapper(self, model, data, device, iteration, dataset_postprocessor, criterion, optimizer, **kwargs) -> Tuple[Any, Any]:
        bs = get_camera_data_batch_size(data)
        scenario_length = get_camera_data_scenario_length(data)

        return super().default_train_run_wrapper(
            batch_size=bs, scenario_length=scenario_length,
            model=model, data=data, device=device, iteration=iteration,
            dataset_postprocessor=dataset_postprocessor, criterion=criterion, optimizer=optimizer,
            model_exec_fn=self._temporal_model_exec_fn, **kwargs
        )


class TemporalLidarModelHandler(TemporalModelHandler):
    def __init__(self, hypes: dict, data_record_len: list, max_frames: int = 5,
                 bev_augmentor: BaseHistoryBEVEmbeddingAugmentor = None,
                 cav_augmentor: BaseCAVAugmentor = None,
                 proposal_augmentor: BaseProposalAugmentor = None) -> None:
        super().__init__(hypes, data_record_len, max_frames, bev_augmentor, cav_augmentor, proposal_augmentor)
    
    def _temporal_model_exec_fn(self, model: TemporalFusionModel, scenario, **kwargs):
        device = kwargs['device'] if 'device' in kwargs else 'cuda'
        cooperative_bev_embeddings = self.previous_cooperative_bev_embeddings
        cooperative_bev_embeddings = to_device(cooperative_bev_embeddings, device)

        if self.previous_cooperative_bev_embeddings is not None and len(self.previous_cooperative_bev_embeddings) > 0:
            if 'vehicle_offsets' in scenario:
                self.previous_vehicle_offsets.append(scenario['vehicle_offsets'])
            else:
                self.previous_vehicle_offsets.append(None)

        fusion_output, _kwargs = model.until_fusion(
            scenario,
            **kwargs
        )

        kwargs.update(_kwargs)

        # augment fusion output (this is the proposal from the current frame)
        if self.bev_proposal_augmentor:
            if 'ignore_bev_proposal_augmentor' in kwargs and kwargs['ignore_bev_proposal_augmentor']:
                pass
            else:
                fusion_output = self.bev_proposal_augmentor._augment(
                    fusion_output, scenario=scenario, **kwargs
                )

        kwargs['vehicle_offsets'] = self.previous_vehicle_offsets

        temporal_fusion_output = model.temporal_fusion(
            fusion_output,
            history_cooperative_bev_embeddings=cooperative_bev_embeddings,
            **kwargs
        )

        prediction = model.from_fusion(
            temporal_fusion_output,
            **kwargs
        )

        return prediction

    def default_eval_run_wrapper(self, model, data, device, iteration, dataset_postprocessor, **kwargs) -> Tuple[Any, Any]:
        bs = get_lidar_data_batch_size(data)
        scenario_length = get_lidar_data_scenario_length(data)

        return super().default_eval_run_wrapper(
            batch_size=bs, scenario_length=scenario_length,
            model=model, data=data, device=device, iteration=iteration,
            dataset_postprocessor=dataset_postprocessor, model_exec_fn=self._temporal_model_exec_fn,
            **kwargs
        )

    def default_train_run_wrapper(self, model, data, device, iteration, dataset_postprocessor, criterion, optimizer, **kwargs) -> Tuple[Any, Any]:
        bs = get_lidar_data_batch_size(data)
        scenario_length = get_lidar_data_scenario_length(data)

        return super().default_train_run_wrapper(
            batch_size=bs, scenario_length=scenario_length,
            model=model, data=data, device=device, iteration=iteration,
            dataset_postprocessor=dataset_postprocessor, criterion=criterion, optimizer=optimizer,
            model_exec_fn=self._temporal_model_exec_fn, **kwargs
        )
