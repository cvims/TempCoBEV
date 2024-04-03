import os
import numpy as np

from opencood.tools.evaluation_temporal.eval_temporal import run_temporal_model
from opencood.tools.runner_temporal.metrics import CameraIoUMetric
import opencood.tools.evaluation_temporal.utils as eval_utils

from opencood.tools.training_temporal.train_utils import to_device


def run_temporal_evaluation_future_frames(
        temporal_model_path: str,
        temporal_net_epoch: int = -1,
        eval_postfix_name: str = 'future_frames',
        future_predictions: int = 0,  # 0 = current; 1 = t + 1; 2 = t +2; ...
        vis_gt_overlay: bool = False,
        vis_ious: bool = False,
        save_results: bool = True,
        check_mean: bool = False,
        hypes_update: dict = None,
        save_hypes: bool = True,
        save_suffix: str = None,
        dl_kwargs: dict = None,
        **kwargs
    ):

    temporal_iou_calculators = [
        CameraIoUMetric(
            dynamic_classes=2,
            static_classes=3
        ) for _ in range(future_predictions)
    ]

    mean_tmp_iou_calculators = [
        CameraIoUMetric(
            dynamic_classes=2,
            static_classes=3
        ) for _ in range(future_predictions)
    ]

    temporal_iou_calculator = None
    save_path = os.path.join(temporal_model_path, 'eval', 'future_frames')
    eval_name = f'temporal_epoch_{temporal_net_epoch}_{eval_postfix_name}'

    for iteration, data, output_processed, _kwargs in run_temporal_model(
        eval_name=eval_name,
        temporal_model_dir=temporal_model_path,
        temporal_net_epoch=temporal_net_epoch,
        hypes_update=hypes_update,
        save_hypes=save_hypes,
        save_metrics=False,
        save_suffix=save_suffix,
        dl_kwargs=dl_kwargs,
    ):
        iou_visualizer = _kwargs['iou_visualizer']
        gt_overlay_visualizer = _kwargs['gt_overlay_visualizer']
        visualization_frames = _kwargs['visualization_frames']

        temporal_processor = _kwargs['temporal_processor']

        temporal_iou_calculator = _kwargs['temporal_iou_calculator']

        if vis_ious:
            iou_visualizer.update_path_suffix('t_0')
            iou_visualizer.visualize(
                epoch=None,
                iteration=iteration,
                save_to_disk=True,
                data=temporal_processor.previous_data_inputs,
                model_outputs=[temporal_processor.previous_model_outputs],
                ious=[temporal_iou_calculator.get_latest(visualization_frames)]
            )
            iou_visualizer.update_path_suffix('')

        if vis_gt_overlay:
            gt_overlay_visualizer.update_path_suffix('t_0')
            gt_overlay_visualizer.visualize(
                epoch=None,
                iteration=iteration,
                save_to_disk=True,
                data=temporal_processor.previous_data_inputs,
                model_outputs=[temporal_processor.previous_model_outputs]
            )
            gt_overlay_visualizer.update_path_suffix('')

        device = _kwargs['device']
        # if queue length is not fully used, skip future predictions

        previous_cooperative_bev_embeddings = temporal_processor.previous_cooperative_bev_embeddings

        if len(previous_cooperative_bev_embeddings) < _kwargs['queue_length']:
            continue

        previous_vehicle_offsets = temporal_processor.previous_vehicle_offsets
        previous_cooperative_bev_embeddings = to_device(previous_cooperative_bev_embeddings, device)
        previous_vehicle_offsets = to_device(previous_vehicle_offsets, device)

        # calculator for current prediction
        postprocessor = _kwargs['dataset'].post_processor

        dataset = _kwargs['dataset']

        if not dataset.has_future_frame(iteration):
            # reset proposal augmentor (only possible with embeddings dataset)
            if dataset.proposal_augmentor is not None:
                dataset.proposal_augmentor._reset()
            continue

        # get model from kwargs
        temporal_model = _kwargs['temporal_model']
        temporal_model.eval()

        future_temporal_outputs = []

        future_frames = []
        future_scenarios = []

        for t in range(future_predictions):
            # get_item with iteration + t + 1
            if not dataset.has_future_frame(iteration + t):
                break

            # get future frame
            future_frame = dataset.__getitem__(iteration + t + 1)
            future_frame = dataset.collate_batch([future_frame])
            # put ego input to input
            future_frame['inputs'] = future_frame['inputs_ego']
            future_frame = to_device(future_frame, device)

            # add future frame to list
            future_frames.append(future_frame)

            # latest scenario
            scenario = {key: future_frame[key][-1] for key in future_frame}
            future_scenarios.append(scenario)
            
            if 'scenario' in _kwargs:
                del _kwargs['scenario']

            # forward pass
            f_temporal_output, __kwargs = temporal_model.until_fusion(
                scenario,
                **_kwargs
            )

            _kwargs.update(__kwargs)

            f_temporal_output = temporal_model.temporal_fusion(
                f_temporal_output,
                previous_cooperative_bev_embeddings,
                **_kwargs
            )
            f_temporal_output = temporal_model.from_fusion(
                f_temporal_output,
                **_kwargs
            )

            # process output
            f_temporal_output = postprocessor.post_process(scenario, f_temporal_output)

            # save output to list
            future_temporal_outputs.append(f_temporal_output)

            # add new output to previous_cooperative_bev_embeddings
            previous_cooperative_bev_embeddings.append(f_temporal_output['bev_embedding'])

            # calculate IoU
            temporal_iou_calculators[t].calculate(
                future_frame, f_temporal_output
            )

            # simple_mean
            if check_mean:
                # use latest model output
                _latest_tmp_mdl_output = temporal_processor.previous_model_outputs[-1]
                _future_mean_tmp = temporal_processor.previous_model_outputs + [_latest_tmp_mdl_output for _ in range(t+1)]

                mean_tmp_iou_calculators[t].calculate(
                    future_frame, postprocessor.post_process(scenario, _latest_tmp_mdl_output)
                )


            if vis_ious:
                # data
                _data = temporal_processor.previous_data_inputs + future_scenarios
                _model_outputs = [
                    temporal_processor.previous_model_outputs + future_temporal_outputs
                ]

                if check_mean:
                    _model_outputs.append(_future_mean_tmp)
            
                _ious = [
                    merge_ious(temporal_iou_calculator.get_latest(visualization_frames), [temporal_iou_calculators[t].get_latest(1) for t in range(t+1)])
                ]
                if check_mean:
                    _ious.append(merge_ious(temporal_iou_calculator.get_latest(visualization_frames), [mean_tmp_iou_calculators[t].get_latest(1) for t in range(t+1)]))

                iou_visualizer.update_path_suffix(f't_{t+1}')
                iou_visualizer.visualize(
                    epoch=None,
                    iteration=iteration,
                    save_to_disk=True,
                    data=_data,
                    model_outputs=_model_outputs,
                    ious=_ious
                )
                iou_visualizer.update_path_suffix('')


    if save_results:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save metrics
        eval_utils.save_metrics_to_file(
            save_path=os.path.join(save_path, 'temporal_metrics_t_0.txt'),
            metrics=temporal_iou_calculator.get_mean(),
        )

        for t in range(future_predictions):
            # Save metrics to txt
            eval_utils.save_metrics_to_file(
                save_path=os.path.join(save_path, f'temporal_metrics_t_{t+1}.txt'),
                metrics=temporal_iou_calculators[t].get_mean(),
            )

            eval_utils.save_metrics_to_file(
                save_path=os.path.join(save_path, f'temporal_all_metrics_t_{t+1}.pkl'),
                metrics=temporal_iou_calculators[t]._subclasses_iou(
                    dynamic_ious=temporal_iou_calculators[t].dynamic_ious,
                    static_ious=temporal_iou_calculators[t].static_ious,
                ),
                use_pickle=True
            )

            if check_mean:
                eval_utils.save_metrics_to_file(
                    save_path=os.path.join(save_path, f'temporal_mean_metrics_t_{t+1}.txt'),
                    metrics=mean_tmp_iou_calculators[t].get_mean(),
                )

                eval_utils.save_metrics_to_file(
                    save_path=os.path.join(save_path, f'temporal_mean_all_metrics_t_{t+1}.pkl'),
                    metrics=mean_tmp_iou_calculators[t]._subclasses_iou(
                        dynamic_ious=mean_tmp_iou_calculators[t].dynamic_ious,
                        static_ious=mean_tmp_iou_calculators[t].static_ious,
                    ),
                    use_pickle=True
                )
    
    return temporal_model, temporal_iou_calculator.get_mean()


def merge_ious(original: dict, new: list):
    # original is key: dict
    # new is list of key: dict
    # merge new into original
    _org = original.copy()
    for key in original:
        for subkey in original[key]:
            # stack new and original
            _org[key][subkey] = np.concatenate([original[key][subkey]] + [new[i][key][subkey] for i in range(len(new))])
    
    return _org
