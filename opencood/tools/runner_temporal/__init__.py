from abc import ABC, abstractmethod
import os
import io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Tuple, List, Union
from opencood.tools.training_temporal.train_utils import to_device
from opencood.visualization.vis_utils import matplot_to_numpy


class BaseAugmentor(nn.Module):
    def __init__(self, apply_probability: float = 1.0) -> None:
        super().__init__()
        self.apply_probability = apply_probability

    def forward(self, data: dict or torch.Tensor, **kwargs) -> Any:
        if torch.rand(1) < self.apply_probability:
            return self._augment(data, **kwargs)
        else:
            return data

    @abstractmethod
    def _augment(self, data: dict, **kwargs) -> dict:
        raise NotImplementedError


    def _reset(self, **kwargs) -> None:
        pass


class BaseCAVAugmentor(BaseAugmentor):
    def __init__(self, apply_probability: float = 1.0) -> None:
        super().__init__(apply_probability)

    def _augment(self, data: dict, **kwargs) -> dict:
        return self._augment_data(data, **kwargs)
    
    @abstractmethod
    def _augment_data(self, data: dict, **kwargs) -> dict:
        raise NotImplementedError


class BaseProposalAugmentor(BaseAugmentor):
    def __init__(self, apply_probability: float = 1.0) -> None:
        super().__init__(apply_probability)

    def _augment(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._augment_fusion(data, **kwargs)
    
    @abstractmethod
    def _augment_fusion(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class BaseHistoryBEVEmbeddingAugmentor(BaseAugmentor):
    def __init__(self, apply_probability: float = 1.0) -> None:
        super().__init__(apply_probability)

    def _augment(self, data: dict, **kwargs) -> dict:
        assert 'cooperative_bev_embeddings' in data, 'Data must contain cooperative_bev_embeddings'
        return self._augment_bev_embeddings(data, **kwargs)
    
    @abstractmethod
    def _augment_bev_embeddings(self, data: dict, **kwargs) -> dict:
        raise NotImplementedError


class AugmentorCombiner(BaseAugmentor):
    def __init__(
        self,
        modules: List[Union[BaseHistoryBEVEmbeddingAugmentor, BaseCAVAugmentor, BaseProposalAugmentor]]
    ) -> None:
        super().__init__(apply_probability=1.0)
        # check if all modules are of the same base type
        base_classes = set([type(module).__bases__[0] for module in modules])
        assert len(base_classes) == 1, 'All modules must be of the same type.'

        self._modules = modules
    
    def _augment(self, data: dict, **kwargs) -> torch.Tensor:
        for module in self._modules:
            data = module(data, **kwargs)
        return data

    def _reset(self) -> None:
        for module in self._modules:
            module._reset()


class BaseMetric(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def calculate(self, data: dict, model_output: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_latest(self, last_x: int = 1) -> dict:
        pass

    @abstractmethod
    def get_mean(self) -> dict:
        pass


class BaseMetricsCombiner(BaseMetric):
    def __init__(self, metrics: List[BaseMetric]) -> None:
        self.metrics = metrics
    
    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()
    
    def calculate(self, data, model_output) -> dict:
        for metric in self.metrics:
            metric.calculate(data, model_output)
    
    def get_latest(self, last_x: int = 1) -> dict:
        return {metric.__class__.__name__: metric.get_latest(last_x) for metric in self.metrics}
    
    def get_mean(self) -> dict:
        return {metric.__class__.__name__: metric.get_mean() for metric in self.metrics}
    

class ModelHandler(ABC):
    def __init__(self, hypes: dict, data_record_len: list, max_frames: int = 5,
                 cav_augmentor: BaseCAVAugmentor = None) -> None:
        super().__init__()
        self.hypes = hypes
        self.cav_augmentor = cav_augmentor
        self._record_lengths = data_record_len

        # used to cut the previous history
        self.max_frames = max_frames

        self.previous_model_outputs = []
        self.previous_data_inputs = []
    
    def run(self, model, data, device, iteration, dataset_postprocessor = None,
            criterion = None, optimizer = None, **kwargs) -> Tuple[Any, Any]:
        exec_fn = None
        if optimizer is None:
            exec_fn = self.default_eval_run_wrapper
        else:
            exec_fn = self.default_train_run_wrapper
        
        return exec_fn(
            model=model, data=data, device=device, iteration=iteration,
            dataset_postprocessor=dataset_postprocessor, criterion=criterion,
            optimizer=optimizer, **kwargs
        )

    def default_eval_run_wrapper(self, model, data, device, iteration, dataset_postprocessor = None, model_exec_fn = None, **kwargs) -> Tuple[Any, Any]:
        raise NotImplementedError

    def default_train_run_wrapper(self, model, data, device,
                                  iteration,
                                  criterion, optimizer,
                                  model_exec_fn = None,
                                  dataset_postprocessor = None, **kwargs) -> Tuple[Any, Any]:
        raise NotImplementedError

    def reset_history(self):
        self.previous_model_outputs = []
        self.previous_data_inputs = []
    
    def _check_record_length(self, data_iteration):
        for record_len in self._record_lengths:
            if record_len == data_iteration:
                # reset history (scenario changes)
                self.reset_history()
    
    def _get_remaining_scenario_length(self, data_iteration):
        for record_len in self._record_lengths:
            if record_len > data_iteration:
                return record_len - data_iteration
    
    def _drop_old_history_values(self):
        if self.previous_model_outputs:
            self.previous_model_outputs = self.previous_model_outputs[-self.max_frames:]
        
        if self.previous_data_inputs:
            self.previous_data_inputs = self.previous_data_inputs[-self.max_frames:]
    
    def _preprocess_data(self, data: dict, iteration: int, train: bool = False,
                         **kwargs) -> Any:        
        # get current data record length
        if not train:
            self._check_record_length(iteration)

        # apply augmentors
        skip_cav_augmentation = kwargs['skip_cav_augmentation'] if 'skip_cav_augmentation' in kwargs else False
        if self.cav_augmentor and not skip_cav_augmentation:
            data = self.cav_augmentor(data=data)

        return data

    def _store_model_output(self, model_output: dict, **kwargs) -> Any:
        # add model outputs to list
        # everything to cpu and detach
        _model_output = {key: model_output[key].cpu().detach() \
                         if isinstance(model_output[key], torch.Tensor) else model_output[key] \
                            for key in model_output}
        
        self.previous_model_outputs.append(_model_output)

    def _store_data(self, data: dict) -> Any:
        # add data inputs to list
        # everything to cpu and detach
        _data = {key: [] for key in data}
        for key in data:
            element = data[key]
            if isinstance(element, torch.Tensor):
                element = element.cpu().detach()
            _data[key] = element
        
        self.previous_data_inputs.append(_data)
    
    def _postprocess_data(self, data: dict, model_output: dict, **kwargs) -> Any:
        # add model outputs
        self._store_model_output(model_output)

        # add data inputs
        self._store_data(data)
        
        # drop old values
        self._drop_old_history_values()


def get_camera_data_batch_size(data: dict):
    return len(data['inputs'][0])


def get_camera_data_scenario_length(data: dict):
    return len(data['inputs'])


def get_lidar_data_batch_size(data: dict):
    if 'processed_lidar' not in data:
        return len(data['inputs'][0])

    return len(data['processed_lidar'][0])


def get_lidar_data_scenario_length(data: dict):
    if 'processed_lidar' not in data:
        return len(data['inputs'])
    return len(data['processed_lidar'])


class ScenarioModelHandler(ModelHandler):
    def __init__(self, hypes: dict, data_record_len: list, max_frames: int = 5, cav_augmentor: BaseCAVAugmentor = None) -> None:
        super().__init__(hypes, data_record_len, max_frames, cav_augmentor)

    def default_eval_run_wrapper(
            self,
            batch_size,
            scenario_length,
            model,
            data,
            device,
            iteration,
            dataset_postprocessor = None,
            model_exec_fn = None,
            **kwargs) -> Tuple[Any, Any]:
        # assert batch_size == 1 and scenario_length == 1, 'Evaluation mode only works with batch size 1 and scenario length 1'

        data = to_device(data, device)

        kwargs.update(dict(
            device=device,
        ))

        # processors that format [scenarios, ...], so we have to [i:i+1]
        scenario_data = {key: data[key][-1:] for key in data}

        scenario_data = self._preprocess_data(
            scenario_data, iteration, train=False, **kwargs
        )

        # pick index i
        scenario_data = {key: scenario_data[key][-1] for key in scenario_data}

        model.eval()

        if model_exec_fn:
            model_output = model_exec_fn(model, scenario_data, **kwargs)
        else:
            model_output = model(scenario_data, **kwargs)
        
        # calculate loss (add batch dimension which must be of shape 1)
        if 'criterion' in kwargs:
            criterion = kwargs['criterion']
            if criterion is not None:
                criterion(model_output, scenario_data)

        if dataset_postprocessor:
            model_output = dataset_postprocessor(scenario_data, model_output)

        self._postprocess_data(scenario_data, model_output)

        return data, model_output

    def default_train_run_wrapper(self,
                                  batch_size, scenario_length,
                                  model, data, device,
                                  iteration,
                                  criterion, optimizer,
                                  dataset_postprocessor = None,
                                  model_exec_fn = None,
                                  **kwargs) -> Tuple[Any, Any]:
        data = to_device(data, device)

        kwargs.update(dict(
            device=device,
        ))

        # we iterate over the scenario length
        # only the last one is used for training (gradient calculation)
        model.zero_grad()
        model.eval()

        # drop all old values
        self.reset_history()
        with torch.no_grad():
            for i in range(scenario_length - 1):
                # processors that format [scenarios, ...], so we have to [i:i+1]
                scenario_data = {key: data[key][i:i+1] for key in data}

                scenario_data = self._preprocess_data(
                    scenario_data, iteration, train=True, **kwargs
                )

                # pick index i (which is also 0 because of previous dict slicing [i:i+1])
                scenario_data = {key: scenario_data[key][0] for key in scenario_data}

                if model_exec_fn:
                    model_output = model_exec_fn(model, scenario_data, only_bev_embeddings=True, grad_mode=False, **kwargs)
                else:
                    model_output = model(scenario_data, only_bev_embeddings=True, **kwargs)

                if dataset_postprocessor:
                    model_output = dataset_postprocessor(scenario_data, model_output)

                self._postprocess_data(scenario_data, model_output)
        
        model.train()
        model.zero_grad()
        # optimizer
        optimizer.zero_grad()
        # train on last scenario
        scenario_data = {key: data[key][-1:] for key in data}
        scenario_data = self._preprocess_data(scenario_data, iteration, train=True)

        # pick last index
        scenario_data = {key: scenario_data[key][-1] for key in scenario_data}

        if model_exec_fn:
            model_output = model_exec_fn(model, scenario_data, grad_mode=True, **kwargs)
        else:
            model_output = model(scenario_data, **kwargs)
        
        if dataset_postprocessor:
            model_output = dataset_postprocessor(scenario_data, model_output)

        # calculate loss (add batch dimension which must be of shape 1)
        loss = criterion(model_output, scenario_data)

        self._postprocess_data(scenario_data, model_output)

        # backprop
        loss.backward()
        optimizer.step()

        return data, model_output


class TemporalModelHandler(ScenarioModelHandler):
    def __init__(self, hypes: dict, data_record_len: list, max_frames: int = 5,
                bev_augmentor: BaseHistoryBEVEmbeddingAugmentor = None,
                cav_augmentor: BaseCAVAugmentor = None,
                proposal_augmentor: BaseProposalAugmentor = None) -> None:
        super().__init__(hypes, data_record_len, max_frames, cav_augmentor)
        self.bev_augmentor = bev_augmentor
        self.bev_proposal_augmentor = proposal_augmentor

        self.previous_cooperative_bev_embeddings = []
        self.previous_vehicle_offsets = []

        # Turn off to predict future steps (zero out current bev embeddings before temporal step)
        self._turn_off_current_proposal_prediction = False
    
    def turn_off_currrent_proposal_prediction(self, turn_off: bool):
        self._turn_off_current_proposal_prediction = turn_off

    def reset_history(self):
        super().reset_history()
        self.previous_cooperative_bev_embeddings = []
        self.previous_vehicle_offsets = []

    def _drop_old_history_values(self):        
        if self.previous_cooperative_bev_embeddings:
            self.previous_cooperative_bev_embeddings = self.previous_cooperative_bev_embeddings[-self.max_frames:]

        if self.previous_vehicle_offsets:
            self.previous_vehicle_offsets = self.previous_vehicle_offsets[-self.max_frames:]

        if self.previous_model_outputs:
            self.previous_model_outputs = self.previous_model_outputs[-self.max_frames:]
        
        if self.previous_data_inputs:
            self.previous_data_inputs = self.previous_data_inputs[-self.max_frames:]
        
        

    def _preprocess_data(
            self, data: dict, iteration: int, train: bool = False,
            **kwargs) -> Any:        
        if not train:
            # get current data record length
            self._check_record_length(iteration)

        # apply augmentors
        skip_cav_augmentation = kwargs['skip_cav_augmentation'] if 'skip_cav_augmentation' in kwargs else False
        skip_bev_augmentation = kwargs['skip_bev_augmentation'] if 'skip_bev_augmentation' in kwargs else False

        if self.cav_augmentor and not skip_cav_augmentation:
            data = self.cav_augmentor(data=data)

        if self.bev_augmentor and not skip_bev_augmentation:
            # only augment the latest of the previous cooperative bev embeddings
            augmented_bev = self.bev_augmentor(
                data=dict(
                    cooperative_bev_embeddings=self.previous_cooperative_bev_embeddings[-1:]
                )
            )

            # replace the last entry with the augmented one
            if augmented_bev['cooperative_bev_embeddings']:
                self.previous_cooperative_bev_embeddings[-1] = augmented_bev['cooperative_bev_embeddings'][-1]

        return data

    def _postprocess_data(self, data: dict, model_output: dict, **kwargs) -> Any:
        use_original_fusion_output = False
        if 'use_original_fusion_embedding_as_history' in kwargs and kwargs['use_original_fusion_embedding_as_history']:
            use_original_fusion_output = True

        # update previous bev embedding
        if use_original_fusion_output:
            cooperative_bev_embedding = model_output['org_fusion_embedding'] if 'org_fusion_embedding' in model_output else None
        else:
            cooperative_bev_embedding = model_output['bev_embedding'] if 'bev_embedding' in model_output else None

        # add cooperative bev embedding
        if cooperative_bev_embedding is not None:
            # to cpu and detach
            cooperative_bev_embedding = cooperative_bev_embedding.cpu().detach()

        self.previous_cooperative_bev_embeddings.append(cooperative_bev_embedding)

        # add model outputs
        self._store_model_output(model_output)

        # add data inputs
        self._store_data(data)
        
        # drop old values
        self._drop_old_history_values()


class ScenarioCameraModelHandler(ScenarioModelHandler):
    def __init__(self, hypes: dict, data_record_len: list, max_frames: int = 5,
                 cav_augmentor: BaseCAVAugmentor = None) -> None:
        super().__init__(hypes=hypes, data_record_len=data_record_len,
                         max_frames=max_frames, cav_augmentor=cav_augmentor)

    def default_eval_run_wrapper(self, model, data, device, iteration, dataset_postprocessor = None, model_exec_fn = None, **kwargs) -> Tuple[Any, Any]:
        bs = get_camera_data_batch_size(data)
        scenario_length = get_camera_data_scenario_length(data)

        return super().default_eval_run_wrapper(
            batch_size=bs, scenario_length=scenario_length,
            model=model, data=data, device=device, iteration=iteration,
            dataset_postprocessor=dataset_postprocessor, model_exec_fn=model_exec_fn,
            **kwargs
        )

    def default_train_run_wrapper(self, model, data, device,
                                  iteration, dataset_postprocessor,
                                  criterion, optimizer,
                                  model_exec_fn = None, **kwargs) -> Tuple[Any, Any]:
        bs = get_camera_data_batch_size(data)
        scenario_length = get_camera_data_scenario_length(data)

        return super().default_train_run_wrapper(
            batch_size=bs, scenario_length=scenario_length,
            model=model, data=data, device=device, iteration=iteration,
            dataset_postprocessor=dataset_postprocessor, criterion=criterion,
            optimizer=optimizer, model_exec_fn=model_exec_fn, **kwargs
        )


class ScenarioLidarModelHandler(ScenarioModelHandler):
    def __init__(self, hypes: dict, data_record_len: list, max_frames: int = 5,
                 cav_augmentor: BaseCAVAugmentor = None) -> None:
        super().__init__(hypes=hypes, data_record_len=data_record_len,
                         max_frames=max_frames, cav_augmentor=cav_augmentor)

    def default_eval_run_wrapper(self, model, data, device, iteration, dataset_postprocessor = None, model_exec_fn = None, **kwargs) -> Tuple[Any, Any]:
        bs = get_lidar_data_batch_size(data)
        scenario_length = get_lidar_data_scenario_length(data)
        
        return super().default_eval_run_wrapper(
            batch_size=bs, scenario_length=scenario_length,
            model=model, data=data, device=device, iteration=iteration,
            dataset_postprocessor=dataset_postprocessor, model_exec_fn=model_exec_fn,
            **kwargs
        )

    def default_train_run_wrapper(self, model, data, device,
                                  iteration, dataset_postprocessor,
                                  criterion, optimizer,
                                  model_exec_fn = None, **kwargs) -> Tuple[Any, Any]:
        bs = get_lidar_data_batch_size(data)
        scenario_length = get_lidar_data_scenario_length(data)

        return super().default_train_run_wrapper(
            batch_size=bs, scenario_length=scenario_length,
            model=model, data=data, device=device, iteration=iteration,
            dataset_postprocessor=dataset_postprocessor, criterion=criterion,
            optimizer=optimizer, model_exec_fn=model_exec_fn, **kwargs
        )


class Visualizer(ABC):
    def __init__(self, save_path: str) -> None:
        self.save_path = save_path
        self.save_path_filename_suffix = None

        self.additional_path = None
        self.warning = False
    
    def update_save_path_filename_suffix(self, suffix: str):
        self.save_path_filename_suffix = suffix
    
    def update_path_suffix(self, additional_path: str):
        """
        Updates the save path and adds an additional path
        """
        self.additional_path = additional_path
    
    def _preprocess_data(self, data: list) -> Any:
        """
        Input of a list of data dicts (history)
        """
        # extract ground truths
        gt_dynamic = []
        for data_dict in data:
            # first entry is always the gt of the ego vehicle
            gt_dynamic.append(data_dict['gt_dynamic'][0].detach().cpu().numpy())

        # todo
        gt_static = []

        _data = dict(
            gt_dynamic=gt_dynamic,
            gt_static=gt_static
        )

        return _data
    
    def _preprocess_model_output(self, model_output: list) -> Any:
        """
        Input of a list of model output dicts (history)
        """
        # extract dynamic probability output
        pred_dynamic = []
        for model_output_dict in model_output:
            # batch size should be 1 [0] and the last index is the prediction [-1]
            pred_dynamic.append(model_output_dict['dynamic_prob'][0][-1].detach().cpu().numpy())

        # todo
        gt_static = []

        _out = dict(
            pred_dynamic=pred_dynamic,
            gt_static=gt_static
        )

        return _out

    def matplot_to_numpy(self, fig):
        return matplot_to_numpy(fig)
    
    def visualize(
            self, epoch: int, iteration: int, save_to_disk: bool = True, **kwargs
    ) -> np.array or dict or plt.Figure:
        full_save_path = self.save_path
        # with suffix if not None
        if self.additional_path:
            if epoch:
                full_save_path = os.path.join(self.save_path, self.additional_path, f'epoch_{epoch}')
            else:
                full_save_path = os.path.join(self.save_path, self.additional_path)
            os.makedirs(full_save_path, exist_ok=True)

        full_save_path = os.path.join(full_save_path, f'iter_{iteration}')
        os.makedirs(self.save_path, exist_ok=True)
        if self.save_path_filename_suffix:
            full_save_path += f'_{self.save_path_filename_suffix}'

        vis_ret = self._visualize(
            **kwargs
        )

        # if it is a dict we assume that it is a point cloud
        if isinstance(vis_ret, dict):
            if save_to_disk:
                full_save_path += '.npy'
                np.save(full_save_path, vis_ret)
        elif isinstance(vis_ret, plt.Figure):
            if save_to_disk:
                full_save_path += '.jpg'
                vis_ret.savefig(full_save_path)
            # close everything related to plt
            plt.close()
        elif isinstance(vis_ret, np.ndarray):
            if save_to_disk:
                full_save_path += '.jpg'
                plt.savefig(full_save_path)
            plt.close()
        else:
            if not self.warning:
                self.warning = True
                print('Cannot save None result from Visualizer to disk. ' + \
                      'This is the only message that will show up even if there' + \
                      'are more None values in the future.')

        return vis_ret
    
    @abstractmethod
    def _visualize(
            self, **kwargs
    ) -> np.array:
        raise NotImplementedError


class TemporalModelIterator(ABC):
    def __init__(self, processor: TemporalModelHandler, data_loader: DataLoader, metrics: BaseMetric or BaseMetricsCombiner = None) -> None:
        self.processor = processor
        self.data_loader = data_loader
        self.metrics = metrics

    @abstractmethod
    def iterate(self, model, device,**kwargs) -> Tuple[Any, Any]:
        raise NotImplementedError


class MultipleModelIterator(ABC):
    def __init__(self, processors: List[ScenarioCameraModelHandler], data_loader: DataLoader, metrics: List[BaseMetric] or List[BaseMetricsCombiner] = None) -> None:
        self.processors = processors
        self.data_loader = data_loader
        self.metrics = metrics

        assert len(self.processors) == len(self.metrics), 'Number of processors and metrics must be the same'
    
    def iterate(self, models: List[nn.Module], device, **kwargs) -> Tuple[Any, Any]:
        raise NotImplementedError


class BaseLogger(ABC):
    def __init__(self) -> None:
        self.log_prefix = None

    def set_log_prefix(self, prefix: str) -> None:
        self.log_prefix = prefix
    
    def log_metrics(self, metrics: dict, epoch: int, iteration: int, batch_len: int, **kwargs) -> None:
        if self.log_prefix:
            metrics = {
                self.log_prefix: metrics
            }

        self._log_metrics(metrics, epoch, iteration, batch_len, **kwargs)

    @abstractmethod
    def _log_metrics(self, metrics: dict, epoch: int, iteration: int, batch_len: int, **kwargs) -> None:
        raise NotImplementedError


    def log_visualization(self, visualization: np.ndarray, epoch: int, iteration: int, batch_len: int, **kwargs) -> None:
        if self.log_prefix:
            kwargs.update(dict(
                caption_prefix=self.log_prefix
            ))

        self._log_visualization(visualization, epoch, iteration, batch_len, **kwargs)

    @abstractmethod
    def _log_visualization(self, visualization: np.ndarray, epoch: int, iteration: int, batch_len: int, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self, **kwargs) -> None:
        raise NotImplementedError
