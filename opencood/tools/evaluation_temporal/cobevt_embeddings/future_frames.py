import argparse
import os
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools.evaluation_temporal._temporal_evaluation import run_temporal_evaluation_future_frames


def test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temporal_model_path', type=str, required=True)
    parser.add_argument('--future_predictions', type=int, default=4)
    parser.add_argument('--use_ego_embeddings_only', action='store_true')
    parser.add_argument('--use_full_bev_view', action='store_true')
    parser.add_argument('--vis_ious', action='store_true')
    parser.add_argument('--vis_gt_overlay', action='store_true')

    return parser.parse_args()


def run_model(model_dir, root_dir, validate_dir, epoch=-1):
    hypes_update = dict(
        train_params=dict(
            use_last_frame_full_view=use_full_bev_view
        ),
        root_dir=root_dir,
        validate_dir=validate_dir,
        fusion=dict(
            args=dict(
                use_ego_only=use_ego_embeddings_only
            )
        )
    )

    eval_postfix_name = f'best_epoch'
    if use_full_bev_view:
        eval_postfix_name += '_full_bev'
    if use_ego_embeddings_only:
        eval_postfix_name += '_ego_only'

    if use_full_bev_view:
        eval_postfix_name += '_full_bev'
    if use_ego_embeddings_only:
        eval_postfix_name += '_ego_only'

    _ = run_temporal_evaluation_future_frames(
        eval_postfix_name=eval_postfix_name,
        temporal_model_path=model_dir,
        parser=None,
        temporal_net_epoch=epoch,
        hypes_update=hypes_update,
        save_results=True,
        vis_ious=vis_ious,
        vis_gt_overlay=vis_gt_overlay,
        save_hypes=False,
        future_predictions=future_predictions,
        use_last_future_pred_only=True,
        check_mean=False,
        dl_kwargs=dl_kwargs
    )



if __name__ == '__main__':
    opt = test_parser()
    use_ego_embeddings_only = opt.use_ego_embeddings_only
    use_full_bev_view = opt.use_full_bev_view
    vis_ious = opt.vis_ious
    vis_gt_overlay = opt.vis_gt_overlay
    future_predictions = opt.future_predictions
    temporal_model_path = opt.temporal_model_path

    dl_kwargs = dict(
        num_workers=8,
        shuffle=False,
    )

    hypes = yaml_utils.load_yaml(os.path.join(opt.temporal_model_path, 'config_eval.yaml'), None)
    validate_dir_base = os.path.dirname(hypes['validate_dir'])

    root_dir = os.path.join(validate_dir_base, 'test')
    validate_dir = os.path.join(validate_dir_base, 'test')

    run_model(temporal_model_path, root_dir, validate_dir)
