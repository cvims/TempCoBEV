import argparse
import os
from opencood.tools.evaluation_temporal.eval_temporal import run_temporal_model
import opencood.hypes_yaml.yaml_utils as yaml_utils


def test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temporal_model_path', type=str, required=True)
    parser.add_argument('--use_ego_embeddings_only', action='store_true')
    parser.add_argument('--use_full_bev_view', action='store_true')
    parser.add_argument('--vis_ious', action='store_true')
    parser.add_argument('--vis_gt_overlay', action='store_true')

    return parser.parse_args()



if __name__ == '__main__':
    opt = test_parser()
    use_ego_embeddings_only = opt.use_ego_embeddings_only
    use_full_bev_view = opt.use_full_bev_view
    vis_ious = opt.vis_ious
    vis_gt_overlay = opt.vis_gt_overlay
    temporal_model_path = opt.temporal_model_path

    dl_kwargs = dict(
        num_workers=8,
        shuffle=False,
    )

    hypes = yaml_utils.load_yaml(os.path.join(opt.temporal_model_path, 'config_eval.yaml'), None)
    validate_dir_base = os.path.dirname(hypes['validate_dir'])

    hypes_update = None
    #for dataset_type in ['train', 'validate', 'test']:
    for dataset_type in ['test']:
        hypes_update = dict(
            train_params=dict(
                use_last_frame_full_view=use_full_bev_view
            ),
            validate_dir=f'{validate_dir_base}/{dataset_type}',
            fusion=dict(
                args=dict(
                    use_ego_only=use_ego_embeddings_only
                )
            )
        )

        for iteration, data, output_processed, kwargs in run_temporal_model(
            eval_name=f'temporal_epoch_best_epoch',
            temporal_model_dir=temporal_model_path,
            parser=None,
            temporal_net_epoch='best',
            hypes_update=hypes_update,
            dl_kwargs=dl_kwargs,
            metrics_file_name='temporal_metrics',
        ):
            iou_visualizer = kwargs['iou_visualizer']
            gt_overlay_visualizer = kwargs['gt_overlay_visualizer']
            temporal_processor = kwargs['temporal_processor']
            temporal_iou_calculator = kwargs['temporal_iou_calculator']
            visualization_frames = kwargs['visualization_frames']

            if vis_ious:
                iou_visualizer.visualize(
                    epoch=None,
                    iteration=iteration,
                    save_to_disk=True,
                    data=temporal_processor.previous_data_inputs,
                    model_outputs=[temporal_processor.previous_model_outputs],
                    ious=[temporal_iou_calculator.get_latest(visualization_frames)]
                )

            if vis_gt_overlay:
                gt_overlay_visualizer.visualize(
                    epoch=None,
                    iteration=iteration,
                    save_to_disk=True,
                    data=temporal_processor.previous_data_inputs,
                    model_outputs=[temporal_processor.previous_model_outputs]                
                )
