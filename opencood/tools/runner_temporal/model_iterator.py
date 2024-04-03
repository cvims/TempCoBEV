import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from opencood.loss import BaseLoss
from opencood.tools.training_temporal.train_utils import to_device
from opencood.tools.runner_temporal import TemporalModelIterator, MultipleModelIterator, TemporalModelHandler, BaseMetric, BaseMetricsCombiner, \
    BaseHistoryBEVEmbeddingAugmentor, BaseCAVAugmentor
from opencood.data_utils.datasets.bev_seg_datasets.camera_only_embeddings import BaseScenarioEmbeddingsDataset as CamBaseScenarioEmbeddingsDataset
from typing import Any, Tuple, List


class EvalTemporalModelIterator(TemporalModelIterator):
    def __init__(
            self,
            processor: TemporalModelHandler,
            data_loader: DataLoader,
            metrics: BaseMetric or BaseMetricsCombiner = None) -> None:
        super().__init__(processor, data_loader, metrics)
    
    def iterate(self, model, device, **kwargs) -> Tuple[Any, Any]:
        model = model.to(device)
        model.eval()

        batch_len = len(self.data_loader)
        pbar_eval = tqdm.tqdm(total=batch_len, leave=True)

        with torch.no_grad():
            for iteration, data in tqdm.tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
                data = to_device(data, device)

                dataset_postprocess = self.data_loader.dataset.post_process if hasattr(self.data_loader.dataset, 'post_process') else None
                if 'use_dataset_postprocessor' in kwargs:
                    if not kwargs['use_dataset_postprocessor']:
                        dataset_postprocess = None

                data, output = self.processor.run(
                    model, data, device, iteration, dataset_postprocess, **kwargs
                )

                if self.metrics:
                    self.metrics.calculate(
                        data=data,
                        model_output=output
                    )

                if 'criterion' in kwargs:
                    criterion = kwargs['criterion']
                    epoch = kwargs['epoch'] if 'epoch' in kwargs else 0
                    logger = kwargs['logger'] if 'logger' in kwargs else None
                    criterion.logging(epoch, iteration, batch_len, logger, pbar_eval)

                yield iteration, data, output


class TrainTemporalModelIterator(TemporalModelIterator):
    def __init__(self, processor: TemporalModelHandler, data_loader: DataLoader, metrics: BaseMetric or BaseMetricsCombiner = None) -> None:
        super().__init__(processor, data_loader, metrics)
    
    def iterate(self, model: nn.Module, device: torch.device, criterion: BaseLoss, optimizer, scheduler, epoch, logger=None, **kwargs) -> Tuple[Any, Any]:
        model = model.to(device)
        model.train()

        batch_len = len(self.data_loader)
        pbar_train = tqdm.tqdm(total=batch_len, leave=True)

        for iteration, data in enumerate(self.data_loader):
            data = to_device(data, device)

            dataset_postprocess = self.data_loader.dataset.post_process if hasattr(self.data_loader.dataset, 'post_process') else None
            if 'use_dataset_postprocessor' in kwargs:
                if not kwargs['use_dataset_postprocessor']:
                    dataset_postprocess = None
            
            # Hack
            if isinstance(self.data_loader.dataset, CamBaseScenarioEmbeddingsDataset):
                kwargs['ignore_bev_proposal_augmentor'] = True

            # processor does gradient descent calculation - no need to do it here
            data, output = self.processor.run(
                model, data, device, iteration, dataset_postprocess,
                criterion, optimizer, **kwargs
            )

            if self.metrics:
                self.metrics.calculate(
                    data=data,
                    model_output=output
                )
            
            criterion.logging(epoch, iteration, batch_len, logger, pbar_train)
            if scheduler:
                scheduler.step_update(epoch * batch_len + iteration)
            
            pbar_train.update(1)

            yield iteration, data, output


class ModelComparerIterator(MultipleModelIterator):
    """
    Only possible for evaluation
    """
    def __init__(
            self,
            processors: List[TemporalModelHandler],
            data_loader: DataLoader,
            metrics: List[BaseMetric],
            bev_augmentor: BaseHistoryBEVEmbeddingAugmentor = None,
            cav_augmentor: BaseCAVAugmentor = None) -> None:
        
        # set internal augmentors to None
        for processor in processors:
            # check if bev augmentor is an attribute of processor
            if hasattr(processor, 'bev_augmentor'):
                processor.bev_augmentor = None
            if hasattr(processor, 'cav_augmentor'):
                processor.cav_augmentor = None

        super().__init__(processors, data_loader, metrics)

        # If we dont augment in the iteration process,
        # we might end up with different augmentations
        # for each model. Therefore, we set the augmentors
        # here to the same ones
        self.bev_augmentor = bev_augmentor
        self.cav_augmentor = cav_augmentor
        
    
    def iterate(self, models: List[nn.Module], device, **kwargs) -> Tuple[Any, Any]:
        assert len(models) == len(self.processors), "Number of models and processors must be equal"
        models = [model.to(device) for model in models]
        [model.eval() for model in models]

        with torch.no_grad():
            for iteration, data in tqdm.tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
                data = to_device(data, device)
                batch_size = len(data[list(data.keys())[0]])
                scenario_length = len(data[list(data.keys())[0]][0])

                dataset_postprocess = self.data_loader.dataset.post_process if hasattr(self.data_loader.dataset, 'post_process') else None
                if 'use_dataset_postprocessor' in kwargs:
                    if not kwargs['use_dataset_postprocessor']:
                        dataset_postprocess = None
                
                # use augmentors batched
                for b in range(batch_size):
                    for s in range(scenario_length):
                        if self.bev_augmentor:
                            _augmented = self.bev_augmentor._augment(
                                {key: data[key][b][s:s+1] for key in data}
                            )

                            for key in _augmented:
                                data[key][b][s] = _augmented[key][s]

                        if self.cav_augmentor:
                            _augmented = self.cav_augmentor._augment(
                                {key: data[key][b][s:s+1] for key in data}
                            )

                            for key in _augmented:
                                data[key][b][s] = _augmented[key][s]

                data_used = []
                all_outputs = []
                for processor, model, metric in zip(self.processors, models, self.metrics):
                    used_data, output = processor.run(
                        model, data, device, iteration, dataset_postprocess, **kwargs
                    )
                    data_used.append(used_data)
                    all_outputs.append(output)

                    if metric:
                        metric.calculate(
                            data=used_data,
                            model_output=output
                        )

                yield iteration, data_used, all_outputs
