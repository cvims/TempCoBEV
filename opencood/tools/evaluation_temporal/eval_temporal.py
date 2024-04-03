import argparse
import os
import torch
import opencood.hypes_yaml.yaml_utils as yaml_utils
import opencood.tools.evaluation_temporal.utils as eval_utils
from opencood.tools.training_temporal import train_utils
import numpy as np

import opencood.tools.evaluation_temporal.utils as eval_utils
from opencood.tools.runner_temporal.model_handler import TemporalCameraModelHandler
from opencood.tools.runner_temporal.model_iterator import EvalTemporalModelIterator
from opencood.tools.runner_temporal.visualizers import IoUVisualizer, BEVGTOverlayVisualizer
from opencood.tools.runner_temporal.metrics import CameraIoUMetric



MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def recursive_update(target, update):
    for key, value in update.items():
        if key in target:
            if isinstance(value, dict) and isinstance(target[key], dict):
                # If both the current value and the value in target are dictionaries,
                # recursively update the nested dictionary
                target[key] = recursive_update(target[key], value)
            else:
                # If not a dictionary, simply update the value
                target[key] = value
        else:
            # If the key does not exist in the target, insert it
            target[key] = value
    return target


def eval_parser():
    parser = argparse.ArgumentParser(description="Evaluation of a temporal model")
    parser.add_argument('--temporal_model_dir', type=str,
                        help='Trained temporal model to evaluate')

    opt = parser.parse_args()
    return opt


def run_temporal_model(
        eval_name,
        temporal_model_dir=None,
        parser=None,
        temporal_net_epoch=-1,
        bev_augmentor=None,  # only one entry for temporal. Standard does not have BEV augmentors
        cav_augmentor=None,  # only one entry for standard. Temporal does not have CAV augmentors
        proposal_augmentor=None,
        hypes_update=None,
        dl_kwargs=None,
        save_hypes=False,
        save_metrics=True,
        metrics_file_name=None,
        save_suffix=None,
        **kwargs
):
    assert temporal_model_dir or parser, 'Either model dir or parser must be provided.'

    opt = parser() if parser else None
    if opt is not None:
        temporal_model_dir = opt.temporal_model_dir

    hypes_temporal_model = os.path.join(temporal_model_dir, 'config_eval.yaml')

    hypes_temporal_model = yaml_utils.load_yaml(hypes_temporal_model, None, test=True)

    # delete the training augmentation from the hypes
    if 'training_augmentation' in hypes_temporal_model['model']['args']:
        del hypes_temporal_model['model']['args']['training_augmentation']

    if hypes_update:
        hypes_temporal_model = recursive_update(hypes_temporal_model, hypes_update)

    # folder name of dataset dir
    dataset_type = os.path.basename(os.path.normpath(hypes_temporal_model['validate_dir']))        

    if save_suffix:
        save_path = os.path.join(temporal_model_dir, 'eval', save_suffix, eval_name, dataset_type)
    else:
        save_path = os.path.join(temporal_model_dir, 'eval', eval_name, dataset_type)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = 1  # do not change
    queue_length = 1  # do not change
    max_frames = 5  # frames to capture for visualization

    # set timestamp offset to 0
    hypes_temporal_model['fusion']['args']['timestamp_offset'] = 0
    org_queue_length = hypes_temporal_model['fusion']['args']['queue_length']

    # Change it to 1 for data loading (sample by sample without preloading queues)
    hypes_temporal_model['fusion']['args']['queue_length'] = queue_length

    # save hypes yaml used for evaluation
    if save_hypes:
        train_utils.save_hypes(
            hypes=hypes_temporal_model,
            saved_path=save_path,
            config_file_name='config_eval.yaml'
        )

    ########################################
    ############## LOAD DATA ###############
    ########################################

    dataloader_kwargs = dict(
        batch_size=batch_size,
        num_workers=8,
        pin_memory=False,
        shuffle=False,
        drop_last=False
    )

    if dl_kwargs:
        dataloader_kwargs.update(dl_kwargs)

    data_loader = eval_utils.create_data_loader(
        hypes=hypes_temporal_model,
        **dataloader_kwargs
    )

    ########################################
    ############## LOAD MODEL ##############
    ########################################

    # Temporal model
    temporal_model = train_utils.create_model(hypes_temporal_model)
    _, temporal_model = train_utils.load_saved_model(temporal_model_dir, temporal_model, use_epoch_x=temporal_net_epoch)

    if hasattr(temporal_model, 'load_model_params'):
        temporal_model.load_model_params(hypes_temporal_model['model']['args'])

    temporal_model = temporal_model.to(device)
    temporal_model.eval()
    
    record_lens = data_loader.dataset.len_record

    ########################################
    ############## EVALUATION ##############
    ########################################

    # 1) Define Augmentors
    # Already done by arguments

    # 2) Define ModelHandler
    temporal_processor = TemporalCameraModelHandler(
        hypes_temporal_model, record_lens, max_frames=org_queue_length,
        cav_augmentor=cav_augmentor, bev_augmentor=bev_augmentor, proposal_augmentor=proposal_augmentor
        )
    temporal_processor.reset_history()

    # 3) Define Visualizer (optional)
    iou_visualizer = IoUVisualizer(save_path, max_frames=max_frames)
    gt_overlay_visualizer = BEVGTOverlayVisualizer(save_path, max_frames=max_frames)
    gt_overlay_visualizer.update_save_path_filename_suffix('gt_overlay')

    # 4) Define Metrics (optional)
    temporal_iou_calculator = CameraIoUMetric(
        dynamic_classes=hypes_temporal_model['model']['args']['dynamic_classes'],
        static_classes=hypes_temporal_model['model']['args']['static_classes']
    )

    # 5) Define TemporalModelIterator   
    temporal_eval_model_iterator = EvalTemporalModelIterator(
        processor=temporal_processor,
        data_loader=data_loader,
        metrics=temporal_iou_calculator,
    )

    # run iterations
    for iteration, data, output_processed in temporal_eval_model_iterator.iterate(
        temporal_model, device,
        **kwargs
    ):
        _kwargs = dict(
            bev_augmentors=bev_augmentor,
            iou_visualizer=iou_visualizer,
            gt_overlay_visualizer=gt_overlay_visualizer,
            temporal_iou_calculator=temporal_iou_calculator,
            temporal_processor=temporal_processor,
            visualization_frames=max_frames,
            device=device,
            queue_length=org_queue_length,
            dataset=data_loader.dataset,
            temporal_model=temporal_model,
            save_path=save_path,
        )

        yield iteration, data, output_processed, _kwargs

    if save_metrics:
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(os.path.join(save_path))

        if metrics_file_name is None:
            metrics_file_name = 'metrics'
        # Save metrics to txt
        eval_utils.save_metrics_to_file(
            save_path=os.path.join(save_path, f'{metrics_file_name}.txt'),
            metrics=temporal_iou_calculator.get_mean(),
        )

        eval_utils.save_metrics_to_file(
            save_path=os.path.join(save_path, f'{metrics_file_name}.pkl'),
            metrics=temporal_iou_calculator._subclasses_iou(
                dynamic_ious=temporal_iou_calculator.dynamic_ious,
                static_ious=temporal_iou_calculator.static_ious,
            ),
            use_pickle=True
        )
