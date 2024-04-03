import os
import copy
import torch
import tqdm

import pickle

import opencood.tools.evaluation_temporal.utils as eval_utils
from opencood.tools.training_temporal import train_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils


DATASET_CONFIG = dict(
    fusion=dict(
        core_method='BEVSegCamScenarioIntermediateFusionDataset',
        args=dict(
        )
    ),
    data_augment=dict(),
    train_params=dict(
        use_last_frame_full_view=True
    )
)


MODEL_CONFIG = lambda model_name: dict(
    core_method=f'camera_bev_seg.{model_name}',
    args=dict(
        target='dynamic',
        dynamic_classes=2,
        static_classes=3,
        max_cav=5,
    )
)


IDENTITY_TEMPORAL_MODULE_HYPES = dict(
    core_method='IdentityEncoder',
    fusion_level='fusion_based',
    args=dict()
)


def recursivly_extend_dict(dict1, dict2):
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict):
                recursivly_extend_dict(dict1[key], dict2[key])
            else:
                dict1[key] = dict2[key]
        else:
            dict1[key] = dict2[key]


def run_embedding_dataset_creation(
        dataset_path: str,
        save_path: str,
        hypes_path: str,
        saved_model_path: str,
        model_config: dict,
        dataset_config: dict,
    ):

    hypes = yaml_utils.load_yaml(hypes_path, None, test=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset_config['validate_dir'] = dataset_path
    dataset_config['preprocess'] = hypes['preprocess']
    dataset_config['postprocess'] = hypes['postprocess']

    # update temporal module to identity
    hypes['model']['args']['temporal_fusion'] = IDENTITY_TEMPORAL_MODULE_HYPES

    # extend model config
    recursivly_extend_dict(hypes['model'], model_config)

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(saved_model_path, model)

    model = model.to(device)
    model.eval()

    hypes.update(dataset_config)
    hypes['train_params']['max_cav'] = 5 # the setting the model was pretrained on

    hypes['model']['args']['max_cav'] = 5

    data_loader_all = eval_utils.create_data_loader(
        hypes=hypes,
        num_workers=8,
        batch_size=1,
        prefetch_factor=8
    )

    ego_hypes = copy.deepcopy(hypes)
    ego_hypes['train_params']['max_cav'] = 1
    ego_hypes['model']['args']['max_cav'] = 1
    data_loader_ego = eval_utils.create_data_loader(
        hypes=ego_hypes,
        num_workers=8,
        batch_size=1,
        prefetch_factor=8
    )

    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for iteration, data in enumerate(tqdm.tqdm(zip(data_loader_all, data_loader_ego), total=len(data_loader_all))):
            data_all = train_utils.to_device(data[0], device)
            data_ego = train_utils.to_device(data[1], device)
            # for each data take the first index (scenario length is always 1)
            # and use this data to run the model
            data_all = {k: v[0] for k, v in data_all.items()}
            data_ego = {k: v[0] for k, v in data_ego.items()}

            # run model until bev embedding
            model_out_all = model(
                data_all,
                history_cooperative_bev_embeddings=None,
                only_bev_embeddings=True
            )

            # always index 0 for batch size 1
            bev_embedding_all = model_out_all['bev_embedding'][0]

            # run model until bev embedding
            model_out_ego = model(
                data_ego,
                history_cooperative_bev_embeddings=None,
                only_bev_embeddings=True
            )

            bev_embedding_ego = model_out_ego['bev_embedding'][0]

            gt_dynamic_full_view = data_all['gt_dynamic_full_view'].squeeze()
            gt_dynamic = data_all['gt_dynamic'].squeeze()
            gt_static = data_all['gt_static'].squeeze()


            # put all data from torch cuda to numpy cpu
            vehicle_offsets = [{cav_id: v_offsets[cav_id].cpu().numpy() for cav_id in v_offsets} for v_offsets in data_all['vehicle_offsets']]
            vehicle_offsets = vehicle_offsets[0]

            bev_embedding_all = bev_embedding_all.cpu().numpy()
            bev_embedding_ego = bev_embedding_ego.cpu().numpy()

            gt_dynamic = gt_dynamic.cpu().numpy()
            gt_dynamic_full_view = gt_dynamic_full_view.cpu().numpy()
            gt_static = gt_static.cpu().numpy()

            # save everything to disk (pickle)
            save_dict = dict(
                bev_embedding=bev_embedding_all,
                bev_embedding_ego=bev_embedding_ego,
                gt_nofull=gt_dynamic,
                gt_dynamic=gt_dynamic_full_view,
                gt_static=gt_static,
                vehicle_offsets=vehicle_offsets
            )

            save_path_i = os.path.join(save_path, f'{iteration}.pkl')
            with open(save_path_i, 'wb') as f:
                pickle.dump(save_dict, f)


if __name__ == '__main__':
    model_name = 'fax'
    model_method_name = 'temporal_cobevt'

    # Dataset 1
    dataset_path = r'/data/OPV2V/original/train'
    save_path = f'/data/OPV2V/embeddings_temporal/{model_name}/train'

    saved_model_path = r'/data/OPV2V/original/pretrained_models/cobevt/dynamic'
    hypes_path = r'/data/OPV2V/original/pretrained_models/cobevt/dynamic/config.yaml'

    dataset_config = DATASET_CONFIG
    model_config = MODEL_CONFIG(model_method_name)

    print('Start creating dataset...')

    run_embedding_dataset_creation(
        dataset_path=dataset_path,
        save_path=save_path,
        hypes_path=hypes_path,
        saved_model_path=saved_model_path,
        dataset_config=dataset_config,
        model_config=model_config
    )

    print('Finished creating dataset...')


    # Dataset 2
    dataset_path = r'/data/OPV2V/original/test'
    save_path = f'/data/OPV2V/embeddings_temporal/{model_name}/test'

    print('Start creating dataset...')

    run_embedding_dataset_creation(
        dataset_path=dataset_path,
        save_path=save_path,
        hypes_path=hypes_path,
        saved_model_path=saved_model_path,
        dataset_config=dataset_config,
        model_config=model_config
    )

    print('Finished creating dataset...')


    # Dataset 3
    dataset_path = r'/data/OPV2V/original/validate'
    save_path = f'/data/OPV2V/embeddings_temporal/{model_name}/validate'

    print('Start creating dataset...')

    run_embedding_dataset_creation(
        dataset_path=dataset_path,
        save_path=save_path,
        hypes_path=hypes_path,
        saved_model_path=saved_model_path,
        dataset_config=dataset_config,
        model_config=model_config
    )

    print('Finished creating dataset...')
