import glob
import importlib
import sys
import yaml
import os
import re
import random
from datetime import datetime

import cv2
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.data_utils.datasets import build_dataset

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def un_freeze_model(model, freeze=True):
    for param in model.parameters():
        param.requires_grad = freeze


def save_model(model, saved_path, epoch, name=None):
    name = 'net_epoch%d.pth' % (epoch + 1) if name is None else name
    torch.save(model.state_dict(),
        os.path.join(saved_path, name)
    )


def set_seeds(seed):
    print('-----------------Seed Setting----------------------')
    seed = init_random_seed(None if seed == 0 else seed)
    print('Set seed to %d' % seed)
    set_random_seed(seed)

    return seed


def load_saved_model_temporal(saved_path, model, freeze_pretrained_layers=False):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    if os.path.isdir(saved_path):
        saved_path = os.path.join(saved_path, 'net_epoch%d.pth' % findLastCheckpoint(saved_path))

    checkpoint = torch.load(
        os.path.join(saved_path),
        map_location='cpu')
    
    # load all layers that can be loaded
    model_dict = model.state_dict()
    # iterate recursively over all keys in model_dict and check if they are in checkpoint
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}

    if not pretrained_dict:
        checkpoint_copy = checkpoint.copy()
        # search for modules that may fit
        for k, v in model._modules.items():
            if k in checkpoint:
                pretrained_dict.update({k: checkpoint[k]})
            elif hasattr(v, 'state_dict'):
                # concat the module name to the checkpoint key (k.checkpoint)
                for k2, v2 in checkpoint.items():
                    if k + '.' + k2 in model_dict:
                        pretrained_dict.update({k + '.' + k2: v2})
                        del checkpoint_copy[k2]
        
        # are there any keys left in the checkpoint?
        if checkpoint_copy:
            print(f'Error: The following keys from the checkpoint could not be loaded: {checkpoint_copy.keys()}')
            print(f'Error: Length of unmatched checkpoint: {len(checkpoint_copy)}')
            raise ValueError('Could not load all keys from checkpoint.')

    model_dict.update(pretrained_dict)

    if freeze_pretrained_layers:
        for param_name, param_tensor in model.named_parameters():
            if param_name in pretrained_dict:
                param_tensor.requires_grad = False

    model.load_state_dict(model_dict)

    del checkpoint

    return model


def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_


def load_best_model(saved_path, model, freeze: bool = False):
    """
    Load best model
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    checkpoint = torch.load(
        os.path.join(saved_path,
                     'best_model.pth'),
        map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)

    if freeze:
        for param_name, param_tensor in model.named_parameters():
            param_tensor.requires_grad = False

    del checkpoint

    return model


def load_saved_model(saved_path, model, use_epoch_x=-1):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    if use_epoch_x == 'best':
        return -1, load_best_model(saved_path, model)

    initial_epoch = findLastCheckpoint(saved_path) if use_epoch_x == -1 else max(1, use_epoch_x)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        checkpoint = torch.load(
            os.path.join(saved_path,
                         'net_epoch%d.pth' % initial_epoch),
            map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

        del checkpoint

    return initial_epoch, model


def save_hypes(hypes, saved_path, config_file_name = 'config.yaml'):
    """
    Save the hypes yaml file.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    saved_path : str
        The saved path.
    """
    if not os.path.exists(saved_path):
        if not os.path.exists(saved_path):
            try:
                os.makedirs(saved_path)
            except FileExistsError:
                pass

        save_name = os.path.join(saved_path, config_file_name)
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)


def setup_train(hypes, save_path_prefix: str = None, config_file_name: str = 'config.yaml', save_dir: str = '../../logs'):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    if save_dir is not None:
        current_path = save_dir
    else:
        current_path = os.path.dirname(__file__)
        current_path = os.path.join(current_path, '../logs')
    if save_path_prefix is not None:
        current_path = os.path.join(current_path, save_path_prefix)

    full_path = os.path.join(current_path, folder_name)

    # save hypes
    save_hypes(hypes, full_path, config_file_name)

    return full_path


def create_data_loader(hypes: dict, train: bool, visualize: bool, validate: bool = False, **kwargs):
    print(f'-----------------Dataset Building (train: {train})------------------')
    dataset = build_dataset(hypes, visualize=visualize, train=train, validate=validate)

    dl_kwargs = dict(
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=False,
        drop_last=True
    )
    dl_kwargs.update(**kwargs)

    return DataLoader(
        dataset,
        **dl_kwargs,
        collate_fn=dataset.collate_batch if hasattr(dataset, 'collate_batch') else None
    )


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')
    target_model_name = target_model_name.split('.')[-1]

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    print('optimizer method is: %s' % optimizer_method)

    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=method_dict['lr'])


def setup_lr_scheduler(hypes, optimizer, n_iter_per_epoch):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    n_iter_per_epoch : int
        Iterations per epoech.
    """
    lr_schedule_config = hypes['lr_scheduler']

    if lr_schedule_config['core_method'] == 'step':
        print('StepLR is chosen for lr scheduler')
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        print('MultiStepLR is chosen for lr scheduler')
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    elif lr_schedule_config['core_method'] == 'exponential':
        print('ExponentialLR is chosen for lr scheduler')
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    elif lr_schedule_config['core_method'] == 'cosineannealwarm':
        print('cosine annealing is chosen for lr scheduler')

        num_steps = lr_schedule_config['epochs'] * n_iter_per_epoch
        warmup_lr = lr_schedule_config['warmup_lr']
        warmup_steps = lr_schedule_config['warmup_epochs'] * n_iter_per_epoch
        lr_min = lr_schedule_config['lr_min']

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

    else:
        sys.exit('Unidentified scheduler')

    return scheduler


def setup_augmentation(hypes) -> dict:
    from opencood.tools.runner_temporal import AugmentorCombiner

    if not 'training_augmentation' in hypes['model']['args']:
        return dict(
            history_bev_augmentors=None,
            dataset_augmentors=None
        )
    
    augmentation_config = hypes['model']['args']['training_augmentation']

    bev_augmentation_functions = []
    if 'history_bev_augmentation' in augmentation_config:
        for bev_augmentation in augmentation_config['history_bev_augmentation']:
            core_method = bev_augmentation['core_method']
            args = bev_augmentation['args']
            bev_augmentors_module = 'opencood.tools.runner_temporal.bev_augmentors'
            lib = importlib.import_module(bev_augmentors_module)
            func = None

            for name, lfunc in lib.__dict__.items():
                if name.lower() == core_method.lower():
                    func = lfunc
            
            if func is None:
                print(f'BEV augmentation method {core_method} not found.')
                exit(0)

            bev_augmentation_functions.append(func(**args))
    
    bev_augmentors = AugmentorCombiner(bev_augmentation_functions) if bev_augmentation_functions else None
    
    cav_augmentation_functions = []
    if 'cav_augmentation' in augmentation_config:
        for dataset_augmentation in augmentation_config['cav_augmentation']:
            core_method = dataset_augmentation['core_method']
            args = dataset_augmentation['args']
            dataset_augmentors_module = 'opencood.tools.runner_temporal.cav_augmentors'
            lib = importlib.import_module(dataset_augmentors_module)
            func = None
            
            for name, lfunc in lib.__dict__.items():
                if name.lower() == core_method.lower():
                    func = lfunc
            
            if func is None:
                print(f'CAV augmentation method {core_method} not found.')
                exit(0)
            
            cav_augmentation_functions.append(func(**args))

    cav_augmentors = AugmentorCombiner(cav_augmentation_functions) if cav_augmentation_functions else None

    proposal_augmentation_functions = []
    if 'proposal_augmentation' in augmentation_config:
        for fusion_augmentation in augmentation_config['proposal_augmentation']:
            core_method = fusion_augmentation['core_method']
            args = fusion_augmentation['args']
            fusion_augmentors_module = 'opencood.tools.runner_temporal.proposal_augmentors'
            lib = importlib.import_module(fusion_augmentors_module)
            func = None
            
            for name, lfunc in lib.__dict__.items():
                if name.lower() == core_method.lower():
                    func = lfunc
            
            if func is None:
                print(f'Proposal augmentation method {core_method} not found.')
                exit(0)
            
            proposal_augmentation_functions.append(func(**args))
    
    proposal_augmentors = AugmentorCombiner(proposal_augmentation_functions) if proposal_augmentation_functions else None

    return dict(
        history_bev_augmentors=bev_augmentors,
        cav_augmentors=cav_augmentors,
        proposal_augmentors=proposal_augmentors
    )


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or inputs is None:
            return inputs
        elif isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).to(device)
        return inputs.to(device)


def save_bev_seg_binary(output_dict,
                        batch_dict,
                        output_dir,
                        batch_iter,
                        epoch,
                        test=False):
    """
    Save the bev segmetation results during training.

    Parameters
    ----------
    batch_dict: dict
        The data that contains the gt.

    output_dict : dict
        The output directory with predictions.

    output_dir : str
        The output directory.

    batch_iter : int
        The batch index.

    epoch : int
        The training epoch.

    test: bool
        Whether this is during test or train.
    """

    if test:
        output_folder = os.path.join(output_dir, 'test_vis')
    else:
        output_folder = os.path.join(output_dir, 'train_vis', str(epoch))

    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
        except FileExistsError:
            pass

    batch_size = batch_dict['ego']['gt_static'].shape[0]

    for i in range(batch_size):
        gt_static_origin = \
            batch_dict['ego']['gt_static'].detach().cpu().data.numpy()[i, 0]
        gt_static = np.zeros((gt_static_origin.shape[0],
                              gt_static_origin.shape[1],
                              3), dtype=np.uint8)
        gt_static[gt_static_origin == 1] = np.array([88, 128, 255])
        gt_static[gt_static_origin == 2] = np.array([244, 148, 0])

        gt_dynamic = \
            batch_dict['ego']['gt_dynamic'].detach().cpu().data.numpy()[i, 0]
        gt_dynamic = np.array(gt_dynamic * 255., dtype=np.uint8)

        pred_static_origin = \
            output_dict['static_map'].detach().cpu().data.numpy()[i]
        pred_static = np.zeros((pred_static_origin.shape[0],
                                pred_static_origin.shape[1],
                                3), dtype=np.uint8)
        pred_static[pred_static_origin == 1] = np.array([88, 128, 255])
        pred_static[pred_static_origin == 2] = np.array([244, 148, 0])

        pred_dynamic = \
            output_dict['dynamic_map'].detach().cpu().data.numpy()[i]
        pred_dynamic = np.array(pred_dynamic * 255., dtype=np.uint8)

        # try to find the right index for raw image visualization
        index = i
        if 'record_len' in batch_dict['ego']:
            cum_sum_len = \
                [0] + list(np.cumsum(
                    torch_tensor_to_numpy(batch_dict['ego']['record_len'])))
            index = cum_sum_len[i]

        # (M, H, W, 3)
        raw_images = \
            batch_dict['ego']['inputs'].detach().cpu().data.numpy()[index, 0]
        visualize_summary = np.zeros((raw_images[0].shape[0] * 2,
                                      raw_images[0].shape[1] * 4,
                                      3),
                                     dtype=np.uint8)
        for j in range(raw_images.shape[0]):
            raw_image = 255 * ((raw_images[j] * STD) + MEAN)
            raw_image = np.array(raw_image, dtype=np.uint8)
            # rgb = bgr
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            visualize_summary[:raw_image.shape[0],
            j * raw_image.shape[1]:
            (j + 1) * raw_image.shape[1],
            :] = raw_image
        # draw gts on the visualization summary
        gt_dynamic = cv2.resize(gt_dynamic, (raw_image.shape[0],
                                             raw_image.shape[1]))
        pred_dynamic = cv2.resize(pred_dynamic, (raw_image.shape[0],
                                                 raw_image.shape[1]))
        gt_static = cv2.resize(gt_static, (raw_image.shape[0],
                                           raw_image.shape[1]))
        pred_static = cv2.resize(pred_static, (raw_image.shape[0],
                                               raw_image.shape[1]))

        visualize_summary[raw_image.shape[0]:, :raw_image.shape[1], :] = \
            cv2.cvtColor(gt_dynamic, cv2.COLOR_GRAY2BGR)
        visualize_summary[raw_image.shape[0]:,
        raw_image.shape[1]:2 * raw_image.shape[1], :] = \
            cv2.cvtColor(pred_dynamic, cv2.COLOR_GRAY2BGR)
        visualize_summary[raw_image.shape[0]:,
        2 * raw_image.shape[1]:3 * raw_image.shape[1], :] = gt_static
        visualize_summary[raw_image.shape[0]:,
        3 * raw_image.shape[1]:4 * raw_image.shape[1], :] = pred_static

        cv2.imwrite(os.path.join(output_folder, '%d_%d_vis.png')
                    % (batch_iter, i), visualize_summary)


def create_temporal_bev_seg_binary(output_dict, batch_dict):
    batch_size = len(batch_dict['gt_static'])
    scenarios = len(batch_dict['gt_static'][0])

    batch_visualizations = []

    for i in range(batch_size):
        # for each temporal scenario
        gt_statics = []
        gt_dynamics = []

        for scenario_idx in range(scenarios):
            gt_static_origin = batch_dict['gt_static'][i][scenario_idx][0].detach().cpu().data.numpy()
            gt_static = np.zeros((gt_static_origin.shape[0],
                                gt_static_origin.shape[1],
                                3), dtype=np.uint8)
            gt_static[gt_static_origin == 1] = np.array([88, 128, 255])
            gt_static[gt_static_origin == 2] = np.array([244, 148, 0])

            gt_dynamic = \
                batch_dict['gt_dynamic'][i][scenario_idx][0].detach().cpu().data.numpy()
            gt_dynamic = np.array(gt_dynamic * 255., dtype=np.uint8)

            gt_statics.append(gt_static)
            gt_dynamics.append(gt_dynamic)

        pred_static_origin = \
            output_dict['static_map'][i].detach().cpu().data.numpy()
        pred_static = np.zeros((pred_static_origin.shape[0],
                                pred_static_origin.shape[1],
                                3), dtype=np.uint8)
        pred_static[pred_static_origin == 1] = np.array([88, 128, 255])
        pred_static[pred_static_origin == 2] = np.array([244, 148, 0])

        pred_dynamic = \
            output_dict['dynamic_map'].detach().cpu().data.numpy()[i]
        pred_dynamic = np.array(pred_dynamic * 255., dtype=np.uint8)
        
        grid_shape = (256, 256)

        # draw gts on the visualization summary
        for i in range(len(gt_dynamics)):
            gt_dynamics[i] = cv2.resize(gt_dynamics[i], grid_shape)
            gt_statics[i] = cv2.resize(gt_statics[i], grid_shape)

        pred_dynamic = cv2.resize(pred_dynamic, grid_shape)
        pred_static = cv2.resize(pred_static, grid_shape)


        # plot all gt dynammics + pred dynamic in one row
        visualize_summary = np.zeros((grid_shape[0] * 2, grid_shape[1] * (len(gt_dynamics) + 1), 3), dtype=np.uint8)
        for j in range(len(gt_dynamics)):
            visualize_summary[:grid_shape[0], j * grid_shape[1]:(j + 1) * grid_shape[1], :] = cv2.cvtColor(gt_dynamics[j], cv2.COLOR_GRAY2BGR)
            visualize_summary[grid_shape[0]:, j * grid_shape[1]:(j + 1) * grid_shape[1], :] = gt_statics[j]
        
        visualize_summary[:grid_shape[0], len(gt_dynamics) * grid_shape[1]:, :] = cv2.cvtColor(pred_dynamic, cv2.COLOR_GRAY2BGR)

        # plot all gt statics + pred static in one row
        for j in range(len(gt_statics)):
            visualize_summary[grid_shape[0]:, j * grid_shape[1]:(j + 1) * grid_shape[1], :] = gt_statics[j]
        
        visualize_summary[grid_shape[0]:, len(gt_statics) * grid_shape[1]:, :] = pred_static
    
        batch_visualizations.append(visualize_summary)
    
    return batch_visualizations


def save_temporal_bev_seg_binary(
        output_dict, batch_dict, output_dir,
        batch_iter, epoch, test=False):
    """
    Save the bev segmetation results during training.

    Parameters
    ----------
    batch_dict: dict
        The data that contains the gt.

    output_dict : dict
        The output directory with predictions.

    output_dir : str
        The output directory.

    batch_iter : int
        The batch index.

    epoch : int
        The training epoch.

    test: bool
        Whether this is during test or train.
    """

    if test:
        output_folder = os.path.join(output_dir, 'test_vis')
    else:
        output_folder = os.path.join(output_dir, 'train_vis', str(epoch))

    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
        except FileExistsError:
            pass

    batch_visualizations = create_temporal_bev_seg_binary(output_dict, batch_dict)

    for vis_summary in batch_visualizations:
        cv2.imwrite(os.path.join(output_folder, '%d_vis.png')
                    % (batch_iter), vis_summary)


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed
    

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_model(
        hypes, model_dir,
        device, model_weights = None,
        freeze_pretrained_layers: bool = False,
        save_dir: str = None,
        save_path_prefix: str = None,
        save_config_file_name: str = 'config.yaml'):
    print('---------------Creating Model------------------')
    model = create_model(hypes)

    # if we want to train from last checkpoint
    if model_dir:
        saved_path = model_dir
        init_epoch, model = load_saved_model(saved_path, model)
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = setup_train(hypes, save_path_prefix=save_path_prefix, config_file_name=save_config_file_name, save_dir=save_dir)

    if not model_dir and model_weights:
        init_epoch = 0
        model = load_saved_model_temporal(model_weights, model, freeze_pretrained_layers=freeze_pretrained_layers)

    # we assume gpu is necessary
    model.to(device)

    return model, saved_path, init_epoch
