import os
import pickle
from typing import Any
import torch
from torch.utils.data import DataLoader
from opencood.data_utils.datasets import build_dataset
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def recursive_hypes_update(target, update):
    for key, value in update.items():
        if key in target:
            if isinstance(value, dict) and isinstance(target[key], dict):
                # If both the current value and the value in target are dictionaries,
                # recursively update the nested dictionary
                target[key] = recursive_hypes_update(target[key], value)
            else:
                # If not a dictionary, simply update the value
                target[key] = value
        else:
            # If the key does not exist in the target, insert it
            target[key] = value
    return target


def create_data_loader(hypes: dict, **kwargs):
    print('-----------------Dataset Building------------------')

    # Make sure that you use the test dataset for evaluation
    # validate_dir must be set to the path of the test dataset
    opencood_dataset = build_dataset(hypes, train=False, validate=True)

    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 1
    num_workers = kwargs['num_workers'] if 'num_workers' in kwargs else 8
    if 'collate_fn' in kwargs:
        collate_fn = kwargs['collate_fn']
    elif 'collate_batch_test'in kwargs and kwargs['collate_batch_test']:
        collate_fn = opencood_dataset.collate_batch_test
    else:
        collate_fn = opencood_dataset.collate_batch
    shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else False
    pin_memory = kwargs['pin_memory'] if 'pin_memory' in kwargs else False
    drop_last = kwargs['drop_last'] if 'drop_last' in kwargs else True

    test_loader = DataLoader(
        opencood_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return test_loader


def drop_cavs_static(batch, drop_count=1, drop_type='random', keep_at_least=1):
    # Drop from batch before feeding to the model
    for b in range(len(batch['inputs'])):  # batch
        for s in range(len(batch['inputs'][b])):  # sequence
            data = dict(
                inputs = batch['inputs'][b][s],
                extrinsic = batch['extrinsic'][b][s],
                intrinsic = batch['intrinsic'][b][s],
                # vehicle_offsets = batch['vehicle_offsets'][b][s],
                gt_static = batch['gt_static'][b][s],
                gt_dynamic = batch['gt_dynamic'][b][s],
                gt_dynamic_non_corp = batch['gt_dynamic_non_corp'][b][s],
                # transformation_matrix = batch['transformation_matrix'][b][s],
                # pairwise_t_matrix = batch['pairwise_t_matrix'][b][s],
                # record_len = batch['record_len'][b][s]
                cav_ids = batch['cav_ids'][b][s],
            )
    
            # we always keep the index 0 (ego vehicle)
            max_drop = len(batch['inputs'][b][s]) - keep_at_least
            if drop_count > max_drop:
                drop_count = max_drop
            
            # todo random drop type

            # for now we only drop the last cavs
            if drop_count > 0:
                new_entries = {
                    key: batch[key][b][s][:-drop_count] if len(batch[key][b][s]) > 1 else batch[key][b][s][:1]
                    for key in data.keys()
                }

                for key in new_entries.keys():
                    batch[key][b][s] = new_entries[key]

                # record length
                batch['record_len'][b][s] = batch['record_len'][b][s] - drop_count

    
    return batch


def drop_cavs_random(batch, drop_prob=0.5, drop_type='random', keep_at_least=1):
    # Drop from batch before feeding to the model
    pass


def plot_embeddings(embeddings: list, save_path: str, mean_separate=False):
    # list of embeddings of shape [C, H, W]
    if mean_separate:
        new_embeddings = []
        for embedding in embeddings:
            # scale them all between [0, 1]
            mean_embedding = (embedding - torch.min(embedding)) / (torch.max(embedding) - torch.min(embedding))
            mean_embedding = mean_embedding.mean(dim=0)
            new_embeddings.append(mean_embedding)
        
        mean_embedding = torch.stack(new_embeddings).tolist()
    else:
        # mean of all embeddings across channel dimension
        mean_embedding = torch.mean(torch.stack(embeddings), dim=1)
        # scale them all between [0, 1]
        mean_embedding = (mean_embedding - torch.min(mean_embedding)) / (torch.max(mean_embedding) - torch.min(mean_embedding))

        # to list again shape: [H, W]
        mean_embedding = mean_embedding.tolist()

    # plot all next to each other (cmap gray)
    fig, axs = plt.subplots(1, len(mean_embedding))
    for i in range(len(mean_embedding)):
        if len(mean_embedding) == 1:
            axs.imshow(mean_embedding[i], cmap='gray')
            axs.axis('off')
        else:
            axs[i].imshow(mean_embedding[i], cmap='gray')
            axs[i].axis('off')
    
    # save to disk
    plt.savefig(save_path)
    plt.close()


def plot_channels(embedding: torch.Tensor, save_dir: str):
    # shape [C, H, W]
    # plot each channel separately
    os.makedirs(save_dir, exist_ok=True)
    for i in range(embedding.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(embedding[i].cpu(), cmap='gray')
        ax.axis('off')
        plt.savefig(os.path.join(save_dir, f'channel_{i}.png'))
        plt.close()


def visualize_vehicle_specific_bev_data(data, save_path: str):
    # Image 1 (For all sequences separately)
        # 1. Row -> Vehicle 1 cameras + gt bev (only vehicle gt view)
        # 2. Row -> Vehicle 2 cameras + gt bev (only vehicle gt view)
        # 3. Row -> Vehicle n cameras + gt bev (only vehicle gt view)
    # shape [BS, Sequence length, CAVs, 1, images, H, W, 3]
    images = data['inputs'][0][0].squeeze(1).cpu()

    # shape [BS, Sequence length, CAVS, H, W]
    bev_gts = data['gt_dynamic_non_corp'][0][0].cpu()

    rows = images.shape[0]  # cav count
    cols = 5  # 1 for each camera + 1 for bev gt
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5*rows))  # Create a row of subplots        
    # Iterate over each camera image
    for cav_idx in range(rows):
        for c, image in enumerate(images[cav_idx]):
            raw_image = 255 * ((image * STD) + MEAN)
            raw_image = np.array(raw_image, dtype=np.uint8)
            # rgb = bgr
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            ax = axes[cav_idx, c] if rows > 1 else axes[c]
            ax.imshow(raw_image)
            ax.set_axis_off()
            ax.set_title(f'Camera {c+1}')
        
        # Add the BEV ground truth image
        ax = axes[cav_idx, cols-1] if rows > 1 else axes[cols-1]
        np_bev_gt = bev_gts[cav_idx].detach().cpu().data.numpy()
        np_bev_gt = np.array(np_bev_gt * 255., dtype=np.uint8)
        ax.imshow(np_bev_gt, cmap='gray')
        ax.set_axis_off()
        ax.set_title('BEV GT')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to release memory


def visualize_temporal_bev_data(data, save_path: str):
    pass


def visualize_cobevt_comparison_bev_seg_binary(temporal_predictions: list, cobevt_predictions: list, ground_truths: list, max_frames: int, save_path: str,
                                               row_text: list = None):
    assert len(temporal_predictions) == len(ground_truths) == len(cobevt_predictions), 'Predictions and ground truths must have the same length.'

    assert row_text is None or len(row_text) == 3, 'Row text must have the same length as the predictions.'
    if row_text is None:
        row_text = ['CoBEVT', 'Temporal CoBEVT', 'BEV GT']
    
    rows = 3
    cols = len(temporal_predictions)

    if cols > max_frames:
        temporal_predictions = temporal_predictions[-max_frames:]
        ground_truths = ground_truths[-max_frames:]
    elif cols < max_frames:
        # fill predictions and ground truths with zeros
        for _ in range(max_frames - cols):
            temporal_predictions.insert(0, torch.zeros_like(temporal_predictions[0]))
            ground_truths.insert(0, torch.zeros_like(ground_truths[0]))
            cobevt_predictions.insert(0, torch.zeros_like(cobevt_predictions[0]))
    
    cols = max_frames

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5*rows))  # Create a row of subplots

    for i, (temporal_pred, cobevt_pred, gt) in enumerate(zip(temporal_predictions, cobevt_predictions, ground_truths)):
        # cobevt
        cobevt_pred = cobevt_pred[0]
        ax = axes[0, i] if rows > 1 and cols > 1 else axes[i]
        np_bev_pred = cobevt_pred.detach().cpu().data.numpy()
        np_bev_pred = np.array(np_bev_pred * 255., dtype=np.uint8)
        ax.imshow(np_bev_pred, cmap='gray')
        ax.set_axis_off()
        # if last column, add row title
        if i == cols - 1:
            ax.set_title(f'{row_text[0]}')
        else:
            ax.set_title(f'CoBEVT Prediction {i+1}')

        # temporal cobevt
        temporal_pred = temporal_pred[0]
        ax = axes[1, i] if rows > 1 and cols > 1 else axes[i]
        np_bev_pred = temporal_pred.detach().cpu().data.numpy()
        np_bev_pred = np.array(np_bev_pred * 255., dtype=np.uint8)
        ax.imshow(np_bev_pred, cmap='gray')
        ax.set_axis_off()
        # if last column, add row title
        if i == cols - 1:
            ax.set_title(f'{row_text[1]}')
        else:
            ax.set_title(f'Temporal CoBEVT Prediction {i+1}')

        # Ground truth
        gt = gt[0]
        ax = axes[2, i] if rows > 1 and cols > 1 else axes[i]
        np_bev_gt = gt.detach().cpu().data.numpy()
        np_bev_gt = np.array(np_bev_gt * 255., dtype=np.uint8)
        ax.imshow(np_bev_gt, cmap='gray')
        ax.set_axis_off()
        if i == cols - 1:
            ax.set_title(f'{row_text[2]}')
        else:
            ax.set_title(f'BEV GT {i+1}')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to release memory


def save_metrics_to_file(save_path: str, metrics: Any, use_pickle: bool = False):
    if use_pickle:
        with open(save_path, 'wb') as f:  # Open in binary write mode ('wb') for pickle
            pickle.dump(metrics, f)
    else:
        with open(save_path, 'w') as f:
            f.write(str(metrics))
            f.close()
    
