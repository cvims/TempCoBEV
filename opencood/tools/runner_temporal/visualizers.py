from opencood.tools.runner_temporal import Visualizer
import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np
import open3d as o3d
from opencood.visualization.vis_utils import color_encoding, bbx2oabb, linset_assign_list, visualize_inference_sample_dataloader
from opencood.utils import common_utils
import time


class TemporalBEVFrameVisualizer(Visualizer):
    def __init__(self, save_path: str, max_frames: int = 5) -> None:
        super().__init__(save_path)
        self.max_frames = max_frames

    def _visualize(
            self, data: List[Dict], model_outputs: List[Dict], **kwargs
    ) -> plt.Figure:
        
        # preprocess data
        data = self._preprocess_data(data)
        model_outputs = self._preprocess_model_output(model_outputs)

        # get gt and pred
        gt_dynamic = data['gt_dynamic']
        pred_dynamic = model_outputs['pred_dynamic']

        sequences = len(gt_dynamic)

        if sequences > self.max_frames:
            gt_dynamic = gt_dynamic[-self.max_frames:]
            pred_dynamic = pred_dynamic[-self.max_frames:]

        # for each batch a separate image
        if sequences == 0:
            return
        
        # # save gt and pred as npy file (only the last frame)
        # gt_path = full_image_save_path.replace('.png', '_gt.npy')
        # np.save(gt_path, gt_dynamic[-1])

        # pred_path = full_image_save_path.replace('.png', '_pred.npy')
        # np.save(pred_path, pred_dynamic[-1])

        
        # get dimensions
        # shape [history, bs, scenario, h, w]
        h, w = gt_dynamic[0].shape
        
        rows = 2
        cols = min(self.max_frames, sequences)

        if sequences > self.max_frames:
            # cut gt and pred
            gt_dynamic = gt_dynamic[-self.max_frames:]
            pred_dynamic = pred_dynamic[-self.max_frames:]

        # create figure
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))  # Create a row of subplots

        # ground truth in first row
        # predictions in second row
        for i, (gt, pred) in enumerate(zip(gt_dynamic, pred_dynamic)):
            # ground truth
            ax = axes[0, i] if rows > 1 and cols > 1 else axes[0]
            ax.imshow(gt, cmap='gray')
            ax.set_axis_off()
            # if last column, add row title
            if i == cols - 1:
                ax.set_title(f'Ground Truth')
            else:
                ax.set_title(f'Ground Truth {i+1}')

            # prediction
            ax = axes[1, i] if rows > 1 and cols > 1 else axes[1]
            ax.imshow(pred, cmap='gray')
            ax.set_axis_off()
            # if last column, add row title
            if i == cols - 1:
                ax.set_title(f'Prediction')
            else:
                ax.set_title(f'Prediction {i+1}')
        
        # self.matplot_to_numpy(fig=fig)
        return fig


class IoUVisualizer(Visualizer):
    def __init__(self, save_path: str, max_frames: int = 5) -> None:
        super().__init__(save_path)
        self.max_frames = max_frames

    def _visualize(
            self, data: List[Dict], model_outputs: List[List[Dict]], **kwargs
    ) -> np.ndarray:

        # preprocess data
        data = self._preprocess_data(data)

        model_outputs = [self._preprocess_model_output(out) for out in model_outputs]

        # get gt and pred
        gt_dynamic = data['gt_dynamic']
        preds_dynamics = [out['pred_dynamic'] for out in model_outputs]

        sequences = len(gt_dynamic)

        if sequences > self.max_frames:
            gt_dynamic = gt_dynamic[-self.max_frames:]
            preds_dynamics = [pred_dynamic[-self.max_frames:] for pred_dynamic in preds_dynamics]
            sequences = self.max_frames

        # for each batch a separate image
        if sequences == 0:
            return
        
        # # save gt and pred as npy file (only the last frame)
        # for i, pred in enumerate(preds_dynamics):
        #     pred_path = full_image_save_path.replace('.png', f'_pred_{i}.npy')
        #     np.save(pred_path, pred[-1])
        
        rows = len(preds_dynamics) + 1
        cols = sequences

        # create figure
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        # ground truth in first row
        for i, gt in enumerate(gt_dynamic):
            gt = gt[0]
            # ground truth
            ax = axes[0, i] if rows > 1 and cols > 1 else axes[0]
            ax.imshow(gt * 255.0, cmap='gray')
            ax.set_axis_off()
            # if last column, add row title
            if i == cols - 1:
                ax.set_title(f'Ground Truth')
            else:
                ax.set_title(f'Ground Truth {i+1}')

        ious = kwargs['ious'] # shape [[[iou_dynamic, iou_static], [iou_dynamic, iou_static]], ...]

        for i, pred_dynamics in enumerate(preds_dynamics):
            for j, pred_dynamic in enumerate(pred_dynamics):
                # prediction
                ax = axes[i+1, j] if rows > 1 and cols > 1 else axes[i+1]
                ax.imshow(pred_dynamic * 255.0, cmap='gray')
                ax.set_axis_off()

                # dynamic
                # text iou dynamic
                iou_dynamic_text = 'Dyn: '
                for key, k_iou in ious[i]['dynamic'].items():
                    iou = k_iou[-sequences:][j]
                    iou_dynamic_text += f'{key[0:3]}: {iou:.2f}, '
                # remove last comma
                iou_dynamic_text = iou_dynamic_text[:-2]

                # static
                # text iou static
                # iou_static_text = 'Stat: '
                # for key, k_iou in ious[i]['static'].items():
                #     iou = k_iou[-sequences:][j]
                #     iou_static_text += f'{key[0:3]}: {iou:.2f}, '
                # # remove last comma
                # iou_static_text = iou_static_text[:-2]

                # add text to plot
                # visualize text above image and add space between images
                ax.text(0.5, 1.1,
                        iou_dynamic_text + '\n', # + iou_static_text,
                        horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, fontsize=10)
        
        # return self.matplot_to_numpy(fig=fig)
        return fig


class BEVGTOverlayVisualizer(Visualizer):
    def __init__(self, save_path: str, max_frames: int = 5) -> None:
        super().__init__(save_path)
        self.max_frames = max_frames

    def _visualize(
            self, data: List[Dict], model_outputs: List[List[Dict]],
            **kwargs
    ) -> plt.Figure:
        # preprocess data
        data = self._preprocess_data(data)

        model_outputs = [self._preprocess_model_output(out) for out in model_outputs]

        # get gt and pred
        gt_dynamic = data['gt_dynamic']
        preds_dynamics = [out['pred_dynamic'] for out in model_outputs]

        sequences = len(gt_dynamic)

        if sequences > self.max_frames:
            gt_dynamic = gt_dynamic[-self.max_frames:]
            preds_dynamics = [pred_dynamic[-self.max_frames:] for pred_dynamic in preds_dynamics]

        # for each batch a separate image
        if sequences == 0:
            return
        
        rows = len(preds_dynamics) + 1
        cols = min(self.max_frames, sequences)

        # create figure
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))

        # ground truth in first row. ground truth values contain only 0 and 1
        # visualize the gt values green
        rgb_gts = []
        for i, gt in enumerate(gt_dynamic):
            gt = gt[0]
            # ground truth
            ax = axes[0, i] if rows > 1 and cols > 1 else axes[0]
            # gt from shape [255, 255] to [255, 255, 3]
            gt = np.expand_dims(gt, -1)
            gt = np.repeat(gt, 3, axis=-1)
            # set red and blue to 0
            gt[:, :, 0] = 0
            gt[:, :, 1] = gt[:, :, 1]
            gt[:, :, 2] = 0
            # to float type
            gt = gt.astype(np.float32)

            rgb_gts.append(gt)

            ax.imshow(gt)
            ax.set_axis_off()
            # if last column, add row title
            if i == cols - 1:
                ax.set_title(f'Ground Truth')
            else:
                ax.set_title(f'Ground Truth {i+1}')

        for i, pred_dynamics in enumerate(preds_dynamics):
            # combine gt and pred
            for j, pred_dynamic in enumerate(pred_dynamics):
                # prediction
                ax = axes[i+1, j] if rows > 1 and cols > 1 else axes[i+1]
                # pred from shape [255, 255] to [255, 255, 3]
                pred_dynamic = np.expand_dims(pred_dynamic, -1)
                pred_dynamic = np.repeat(pred_dynamic, 3, axis=-1)
                # set green and blue to zero
                pred_dynamic[:, :, 0] = pred_dynamic[:, :, 0]
                pred_dynamic[:, :, 1] = 0
                pred_dynamic[:, :, 2] = 0

                # add gt to pred
                pred_dynamic = pred_dynamic + rgb_gts[j]

                ax.imshow(pred_dynamic)
                ax.set_axis_off()
                # if last column, add row title
                if i == cols - 1:
                    ax.set_title(f'Prediction')
                else:
                    ax.set_title(f'Prediction {i+1}')

        # return self.matplot_to_numpy(fig=fig)
        return fig


class ObjectDetection3DVisualizer(Visualizer):
    def __init__(self, save_path: str, post_processor) -> None:
        super().__init__(save_path)
        self.post_processor = post_processor

    def _visualize(
            self, data: List[Dict], model_outputs: List[List[Dict]],
            **kwargs
    ) -> dict:
        def get_pred_corners_and_colors(inp, color):
            corners = [
                np.asarray(o3d.geometry.LineSet.create_from_oriented_bounding_box(a).points)
                for a in inp
            ]
            colors = [
                np.asarray(color)
                for _ in inp
            ]

            # to format dict {'boxes': np.array([ { "corners": [...] }, ...])}
            corners = np.array([{"corners": corner.tolist()} for corner in corners])
            colors = np.array([{"color": color.tolist()} for color in colors])

            return corners, colors
        
        # latest of data
        data = data[-1]
        # latest of model outputs (first model, latest output)
        model_output = model_outputs[0][-1]

        # TODO
        # calculate pred_box_tensor, pred_score, gt_box_tensor (see inference_late_fusion or LossMetric)
        _data = dict(
            ego=data
        )
        _model_output = dict(
            ego=model_output
        )
        pred_box_tensor, pred_score, gt_box_tensor = \
            self.post_processor(_data, _model_output)
        
        origin_lidar = data['origin_lidar']
        
        mode = 'intensity'

        if not isinstance(origin_lidar, np.ndarray):
            origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)

        origin_lidar_intcolor = \
            color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                        else origin_lidar[:, 2], mode=mode)
        # left -> right hand
        origin_lidar[:, :1] = -origin_lidar[:, :1]

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
        o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

        oabbs_pred = bbx2oabb(pred_box_tensor, color=(1, 0, 0))
        oabbs_gt = bbx2oabb(gt_box_tensor, color=(0, 1, 0))

        # oriented bounding box to corners (N x 8 x 3 numpy array)
        oabbs_pred_corners, oabbs_pred_colors = get_pred_corners_and_colors(oabbs_pred, (255, 0, 0))
        oabbs_gt_corners, oabbs_gt_colors = get_pred_corners_and_colors(oabbs_gt, (0, 255, 0))

        # merge corners and colors
        pcd_full = {
            "points": [],
            "boxes": [],
        }

        # add pcd points to dict
        pcd_full['points'] = np.asarray(o3d_pcd.points)

        # add pred boxes to dict
        i=0
        for i, (corners, colors) in enumerate(zip(oabbs_pred_corners, oabbs_pred_colors)):
            pcd_full['boxes'].append({**corners, **colors, "label": f"Box-{i}"})
        
        # add gt boxes to dict
        for j, (corners, colors) in enumerate(zip(oabbs_gt_corners, oabbs_gt_colors)):
            pcd_full['boxes'].append({**corners, **colors, "label": f"Box-{i+j}"})
        
        pcd_full['boxes'] = np.array(pcd_full['boxes'])

        # return the o3d geometry point cloud
        return pcd_full


class ObjectDetection3DRenderer(Visualizer):
    def __init__(self, save_path: str, post_processor) -> None:
        # max frames is not important here
        super().__init__(save_path)
        self.post_processor = post_processor

        self.vis = None
        self.vis_pcd = o3d.geometry.PointCloud()
        self._latest_pcd = None
        self._latest_pred_o3d_box = None
        self._latest_gt_o3d_box = None
        # used to visualize object bounding box, maximum 50
        self._vis_aabbs_gt = []
        self._vis_aabbs_pred = []
        for _ in range(500):
            self._vis_aabbs_gt.append(o3d.geometry.TriangleMesh())
            self._vis_aabbs_pred.append(o3d.geometry.TriangleMesh())

    def _visualize(
            self, data: List[Dict], model_outputs: List[List[Dict]],
            **kwargs
    ) -> dict:
        first_step = True
        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()

            self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            self.vis.get_render_option().point_size = 1.0
            self.vis.get_render_option().line_width = 10
            self.vis.get_render_option().show_coordinate_frame = True

            print('Open3D Renderer openend. Do not forget to close it programmatically with method "destroy".')
        else:
            first_step = False

        # latest of data
        data = data[-1]
        # latest of model outputs (first model, latest output)
        model_output = model_outputs[0][-1]

        # TODO
        # calculate pred_box_tensor, pred_score, gt_box_tensor (see inference_late_fusion or LossMetric)
        _data = dict(
            ego=data
        )
        _model_output = dict(
            ego=model_output
        )
        pred_box_tensor, pred_score, gt_box_tensor = \
            self.post_processor(_data, _model_output)
        
        self._latest_pcd, self._latest_pred_o3d_box, self._latest_gt_o3d_box = \
            visualize_inference_sample_dataloader(
                pred_box_tensor,
                gt_box_tensor,
                data['origin_lidar'],
                self.vis_pcd,
                mode='constant'
            )
        
        if first_step:
            self.vis.add_geometry(self._latest_pcd)
            linset_assign_list(
                self.vis,
                self._vis_aabbs_pred,
                self._latest_pred_o3d_box,
                update_mode='add')

            linset_assign_list(
                self.vis,
                self._vis_aabbs_gt,
                self._latest_gt_o3d_box,
                update_mode='add')
        
        linset_assign_list(
            self.vis,
            self._vis_aabbs_pred,
            self._latest_pred_o3d_box)
        
        linset_assign_list(
            self.vis,
            self._vis_aabbs_gt,
            self._latest_gt_o3d_box)

        self.vis.update_geometry(self._latest_pcd)
        self.vis.poll_events()
        self.vis.update_renderer() 

        time.sleep(0.001)

        return None

    def destroy(self):
        self.vis.destroy_window()
        self.vis = None
        self.vis_pcd = o3d.geometry.PointCloud()
        self._latest_pcd = None
        self._latest_pred_o3d_box = None
        self._latest_gt_o3d_box = None
        # used to visualize object bounding box, maximum 50
        self._vis_aabbs_gt = []
        self._vis_aabbs_pred = []
        for _ in range(500):
            self._vis_aabbs_gt.append(o3d.geometry.TriangleMesh())
            self._vis_aabbs_pred.append(o3d.geometry.TriangleMesh())

