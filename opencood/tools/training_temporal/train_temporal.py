import argparse
import os
import copy
from typing import Tuple

import matplotlib
matplotlib.use("Agg")

import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools.training_temporal import train_utils

from opencood.tools.runner_temporal.model_handler import TemporalCameraModelHandler
from opencood.tools.runner_temporal.model_iterator import TrainTemporalModelIterator, EvalTemporalModelIterator
from opencood.tools.runner_temporal.metrics import CameraIoUMetric
from opencood.tools.runner_temporal.visualizers import BEVGTOverlayVisualizer
from opencood.tools.runner_temporal.logger import WandBLogger, NoLogger


WANDB_PROJECT_NAME ='temporal_opv2v_camera'


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str,
                        help='data generation yaml file needed')
    parser.add_argument('--model_dir', type=str,
                        help='Continued training path')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for training')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode')
    opt = parser.parse_args()
    return opt



def run_training(
        hypes_yaml=None,
        model_dir=None,
        seed=42,
        debug=False,
        parser=None,
        dl_kwargs = None,
        **kwargs
) -> Tuple[str, torch.nn.Module, int]:
    # save dir is the folder location of the opencood package
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')

    _dl_kwargs = dict(
        num_workers=8 if not debug else 1,
        pin_memory=False,
    )

    if dl_kwargs:
        _dl_kwargs.update(dl_kwargs)
    
    dl_kwargs_train = dict(
        drop_last=True
    )
    dl_kwargs_train.update(_dl_kwargs)
    if 'shuffle' not in dl_kwargs_train:
        dl_kwargs_train['shuffle'] = True

    dl_kwargs_val = dict(
        drop_last=False
    )
    dl_kwargs_val.update(_dl_kwargs)
    dl_kwargs_val['shuffle'] = False


    assert hypes_yaml or parser, 'Either hypes_yaml or parser must be provided.'

    opt = parser() if parser else None
    if opt is not None:
        hypes_yaml = opt.hypes_yaml
        model_dir = opt.model_dir
        seed = opt.seed
        debug = opt.debug

    hypes = yaml_utils.load_yaml(hypes_yaml, None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = train_utils.set_seeds(seed)
    hypes['train_params']['seed'] = seed

    pretrained_model_weights_dir = hypes['train_params']['pretrained_model_weights'] if 'pretrained_model_weights' in hypes['train_params'] else None

    if pretrained_model_weights_dir and not os.path.exists(pretrained_model_weights_dir):
        pretrained_model_weights_dir = None
        print('No pretrained model used for training.')

    # multi_gpu_utils.init_distributed_mode(opt)

    train_loader = train_utils.create_data_loader(
        hypes, train=True, validate=False, visualize=False,
        batch_size=hypes['train_params']['batch_size'],
        **dl_kwargs_train
    )

    eval_hypes = copy.deepcopy(hypes)
    # We delete the augmentation by default from the evaluation hypes
    if 'training_augmentation' in eval_hypes['model']['args']:
        del eval_hypes['model']['args']['training_augmentation']

    eval_hypes['fusion']['args']['queue_length'] = 1
    val_loader = train_utils.create_data_loader(
        eval_hypes, train=False, validate=True, visualize=False,
        batch_size=1,
        **dl_kwargs_val
    )

    unfreeze_after_epochs = hypes['train_params']['un_freeze_after_epochs'] if 'un_freeze_after_epochs' in hypes['train_params'] else 0

    save_path_prefix = 'opv2v_camera'
    if 'save_path_suffix' in kwargs:
        save_path_prefix = os.path.join(save_path_prefix, kwargs['save_path_suffix'])

    model, saved_path, init_epoch = train_utils.prepare_model(
        hypes=hypes,
        model_dir=model_dir,
        device=device,
        model_weights=pretrained_model_weights_dir,
        freeze_pretrained_layers=unfreeze_after_epochs > 0,
        save_path_prefix=save_path_prefix,
        save_config_file_name=os.path.basename(hypes_yaml),
        save_dir=save_dir
    )

    # save hypes yaml used for training with initial name

    validation_image_save_path = os.path.join(saved_path, 'train_vis')

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)

    # augmentation setup
    augmentors = train_utils.setup_augmentation(hypes)
    bev_augmentor = augmentors['history_bev_augmentors'] if 'history_bev_augmentors' in augmentors and augmentors['history_bev_augmentors'] else None
    cav_augmentors = augmentors['cav_augmentors'] if 'cav_augmentors' in augmentors and augmentors['cav_augmentors'] else None
    proposal_augmentor = augmentors['proposal_augmentors'] if 'proposal_augmentors' in augmentors and augmentors['proposal_augmentors'] else None

    # lr scheduler setup
    epochs = hypes['train_params']['epochs']

    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_scheduler(hypes, optimizer, num_steps)

    print('Training start with num steps of %d' % num_steps)

    # Define train temporal model handler
    train_processor = TemporalCameraModelHandler(
        hypes=hypes,
        data_record_len=train_loader.dataset.len_record,
        max_frames=hypes['fusion']['args']['queue_length'] if 'queue_length' in hypes['fusion']['args'] else 1,
        bev_augmentor=bev_augmentor,
        cav_augmentor=cav_augmentors,
        proposal_augmentor=proposal_augmentor
    )
    train_processor.reset_history()

    # Define eval temporal model handler
    eval_max_frames = 5
    eval_processor = TemporalCameraModelHandler(
        hypes=hypes,
        data_record_len=val_loader.dataset.len_record,
        max_frames=eval_max_frames,
        bev_augmentor=None,
        cav_augmentor=None
    )
    eval_processor.reset_history()

    # Define visualizer
    # if you have more visualizers, use VisualizerCombiner
    eval_visualizer = BEVGTOverlayVisualizer(
        save_path=validation_image_save_path,
        max_frames=eval_max_frames
    )

    # Define metrics
    # if you have more metrics, use MetricCombiner
    train_metrics = CameraIoUMetric(
        dynamic_classes=hypes['model']['args']['dynamic_classes'],
        static_classes=hypes['model']['args']['static_classes']
    )
    eval_metrics = CameraIoUMetric(
        dynamic_classes=hypes['model']['args']['dynamic_classes'],
        static_classes=hypes['model']['args']['static_classes']
    )

    # Define wandb logger
    wandb_logger = WandBLogger(
        name=os.path.basename(saved_path),
        project=WANDB_PROJECT_NAME if not 'wandb_project_name' in kwargs else kwargs['wandb_project_name'],
        config=hypes,
        entity='urban-ai',
        mode='disabled' if debug else 'online'
    ) if not debug else NoLogger()

    val_log_every_x_visualizations = 100

    # Definer iterator for training
    train_temporal_model_iterator = TrainTemporalModelIterator(
        processor=train_processor,
        metrics=train_metrics,
        data_loader=train_loader
    )

    # Define iterator for evaluation
    eval_temporal_model_iterator = EvalTemporalModelIterator(
        processor=eval_processor,
        metrics=eval_metrics,
        data_loader=val_loader
    )

    for epoch in range(init_epoch, max(epochs, init_epoch)):
        # free history
        # train_processor.reset_history()
        eval_processor.reset_history()

        # free metric history
        train_metrics.reset()
        eval_metrics.reset()

        if epoch == unfreeze_after_epochs:
            print('Unfreezing pretrained layers')
            train_utils.un_freeze_model(model)

            for param_group in optimizer.param_groups:
                print('learning rate %.7f' % param_group["lr"])

        # run training
        wandb_logger.set_log_prefix('train')
        for iteration, data, output_processed in train_temporal_model_iterator.iterate(
            model=model, device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, logger=wandb_logger,
            **kwargs
        ):
            # nothing to do here
            # visualization = eval_visualizer.visualize(
            #     epoch=epoch,
            #     iteration=iteration,
            #     data=train_processor.previous_data_inputs,
            #     model_outputs=[
            #         train_processor.previous_model_outputs
            #     ]
            # )
            pass
            
        # save metrics to logger
        ious = train_metrics.get_mean()
        wandb_logger.log_metrics(ious, epoch, iteration, batch_len=len(train_loader))
        train_metrics.reset()
        
        # save model
        if epoch % hypes['train_params']['save_freq'] == 0:
            train_utils.save_model(model, saved_path, epoch)

        best_iou = -1
        # if epoch % hypes['train_params']['eval_freq'] == 0 and epoch > 0:
        if epoch >= 0:  #  universal - always true
            wandb_logger.set_log_prefix('eval')
            # run evaluation
            for iteration, data, output_processed in eval_temporal_model_iterator.iterate(
                model=model, device=device,
                criterion=criterion, logger=wandb_logger, epoch=epoch
            ):
                # save to disk visualizer
                eval_visualizer.update_path_suffix('%d' % epoch)

                if iteration % val_log_every_x_visualizations == 0:
                    visualization = eval_visualizer.visualize(
                        epoch=epoch,
                        iteration=iteration,
                        data=eval_processor.previous_data_inputs,
                        model_outputs=[
                            eval_processor.previous_model_outputs
                        ]
                    )

                    # save images (visualization to wandb)
                    wandb_logger.log_visualization(visualization, epoch, iteration, batch_len=len(val_loader))

            # save metrics to logger
            ious = eval_metrics.get_mean()
            # filter best epoch
            current_eval_iou = ious['dynamic']['vehicle']
            if current_eval_iou > best_iou:
                best_iou = current_eval_iou
                # save best model
                train_utils.save_model(model, saved_path, epoch, name='best_model.pth')

            wandb_logger.log_metrics(ious, epoch, iteration, batch_len=len(val_loader))
            eval_metrics.reset()

        # reinitialize train loader
        train_loader.dataset.reinitialize()
    
    wandb_logger.finalize()

    return saved_path, model, epochs


if __name__ == '__main__':
    run_training(
        parser=train_parser
    )
    print('done')
