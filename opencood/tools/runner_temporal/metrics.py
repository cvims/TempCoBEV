import copy
import torch
from opencood.utils.seg_utils import mean_IU
from opencood.loss import BaseLoss
from opencood.tools.runner_temporal import BaseMetric
import numpy as np
from opencood.utils import eval_utils


class NoMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
    
    def reset(self) -> None:
        pass

    def calculate(self, data: dict, model_output: dict) -> dict:
        return dict()

    def get_mean(self) -> dict:
        return dict()
    
    def get_latest(self, last_x: int = 1) -> dict:
        return dict()


class CameraIoUMetric(BaseMetric):
    def __init__(self, dynamic_classes: int, static_classes: int) -> None:
        super().__init__()
        self.dynamic_classes = dynamic_classes
        self.static_classes = static_classes
        self.dynamic_ious = []
        self.static_ious = []

    def reset(self):
        self.dynamic_ious = []
        self.static_ious = []

    def calculate(self, data: dict, model_output: dict) -> dict:
        # check if data is available
        # if 'static_map' not in model_output.keys() or 'dynamic_map' not in model_output.keys():
        if 'dynamic_map' not in model_output.keys():
            return None

        batch_size = data['gt_dynamic'][0].shape[0]

        ious_dynamic = []
        # ious_static = []

        for b in range(batch_size):
            # get last scneario ground-truth
            # data['gt_static'] is a list of [batch size, CAVs, H, W]
            # CAVs 0 is the ego vehicle; depending on the config, the other indices are other CAVs or no other information is available

            # gt_static = data['gt_static'][-1][b][0].detach().cpu().data.numpy()
            # gt_static = np.array(gt_static, dtype=np.int32)

            gt_dynamic = data['gt_dynamic'][-1][b][0].detach().cpu().data.numpy()
            gt_dynamic = np.array(gt_dynamic, dtype=np.int32)

            # pred_static = model_output['static_map'][b].detach().cpu().data.numpy()
            # pred_static = np.array(pred_static, dtype=np.int32)

            pred_dynamic = model_output['dynamic_map'][b].detach().cpu().data.numpy()
            pred_dynamic = np.array(pred_dynamic, dtype=np.int32)

            iou_dynamic = mean_IU(pred_dynamic, gt_dynamic, n_classes=self.dynamic_classes)
            # iou_static = mean_IU(pred_static, gt_static, n_classes=self.static_classes)

            ious_dynamic.append(iou_dynamic)
            # ious_static.append(iou_static)

            self.dynamic_ious.append(iou_dynamic)
            # self.static_ious.append(iou_static)

        return dict(
            dynamic=ious_dynamic,
            # static=ious_static
        )
    
    def get_latest(self, last_x: int = 1):
        return self._subclasses_iou(self.dynamic_ious[-last_x:], self.static_ious[-last_x:])

    def _subclasses_iou(self, dynamic_ious, static_ious):
        """
        Returns the mean IoU for dynamic and static sub classes.
        """
        # dynamic
        # 0: background
        # 1: vehicle

        # static
        # 0: background
        # 1: driving area
        # 2: lane

        dynamic_ious = np.stack(dynamic_ious)
        # static_ious = np.stack(static_ious)
        dynamic_ious = dict(
            background=dynamic_ious[:, 0],
            vehicle=dynamic_ious[:, 1]
        )
        # static_ious = dict(
        #     background=static_ious[:, 0],
        #     driving_area=static_ious[:, 1],
        #     lane=static_ious[:, 2]
        # )

        return dict(
            dynamic=dynamic_ious,
            # static=static_ious
        )
    
    def get_mean(self):
        return self.get_mean_ext(self.dynamic_ious, self.static_ious)

    def get_mean_ext(self, dynamic_ious, static_ious):
        """
        Returns the mean IoU for dynamic and static sub classes.
        """
        # dynamic
        # 0: background
        # 1: vehicle

        # static
        # 0: background
        # 1: driving area
        # 2: lane

        dynamic_ious = np.stack(dynamic_ious)
        # static_ious = np.stack(static_ious)
        dynamic_ious = dict(
            background=np.mean(dynamic_ious[:, 0]),
            vehicle=np.mean(dynamic_ious[:, 1])
        )
        # static_ious = dict(
        #     background=np.mean(static_ious[:, 0]),
        #     driving_area=np.mean(static_ious[:, 1]),
        #     lane=np.mean(static_ious[:, 2])
        # )

        return dict(
            dynamic=dynamic_ious,
            # static=static_ious
        )


class LidarAveragePrecision(BaseMetric):
    def __init__(self, dataset_post_processor) -> None:
        super().__init__()
        self.dataset_post_processor = dataset_post_processor

        # Create the dictionary for evaluation
        stats = self._create_stats()
        self.result_stat = stats[0]
        self.result_stat_short = stats[1]
        self.result_stat_middle = stats[2]
        self.result_stat_long = stats[3]
    
    def reset(self):
        # Create the dictionary for evaluation
        stats = self._create_stats()
        self.result_stat = stats[0]
        self.result_stat_short = stats[1]
        self.result_stat_middle = stats[2]
        self.result_stat_long = stats[3]

    def _create_stats(self):
        result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                            0.7: {'tp': [], 'fp': [], 'gt': 0}}
        result_stat_short = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                                    0.7: {'tp': [], 'fp': [], 'gt': 0}}
        result_stat_middle = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                                    0.7: {'tp': [], 'fp': [], 'gt': 0}}
        result_stat_long = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                                0.7: {'tp': [], 'fp': [], 'gt': 0}}
        
        return result_stat, result_stat_short, result_stat_middle, result_stat_long


    def calculate(self, data: dict, model_output: dict) -> dict:
        # put ego key to data
        # we only process the last scenario for calculation
        # because this is the one used for prediction and evaluation
        _data = dict(
            ego={key: data[key][0][-1] for key in data if key != 'label_dict'}
        )
        _model_output = dict(
            ego=model_output
        )
        pred_box_tensor, pred_score, gt_box_tensor = \
            self.dataset_post_processor(_data, _model_output)

        # overall calculating
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    self.result_stat,
                                    0.5)
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    self.result_stat,
                                    0.7)
        # short range
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    self.result_stat_short,
                                    0.5,
                                    left_range=0,
                                    right_range=30)
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    self.result_stat_short,
                                    0.7,
                                    left_range=0,
                                    right_range=30)

        # middle range
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    self.result_stat_middle,
                                    0.5,
                                    left_range=30,
                                    right_range=50)
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    self.result_stat_middle,
                                    0.7,
                                    left_range=30,
                                    right_range=50)

        # right range
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    self.result_stat_long,
                                    0.5,
                                    left_range=50,
                                    right_range=100)
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    self.result_stat_long,
                                    0.7,
                                    left_range=50,
                                    right_range=100)


    def _final_results(self, result_stat):
        ap_50, mrec_50, mpre_50 = eval_utils.calculate_ap(copy.deepcopy(result_stat), 0.50)
        ap_70, mrec_70, mpre_70 = eval_utils.calculate_ap(copy.deepcopy(result_stat), 0.70)

        return {
            'ap_50': ap_50,
            'ap_70': ap_70,
            #'mpre_50': mpre_50,
            #'mrec_50': mrec_50,
            #'mpre_70': mpre_70,
            #'mrec_70': mrec_70,
        }


    def get_mean(self) -> dict:
        return dict(
            result_stat=self._final_results(self.result_stat),
            result_stat_short=self._final_results(self.result_stat_short),
            result_stat_middle=self._final_results(self.result_stat_middle),
            result_stat_long=self._final_results(self.result_stat_long)
        )
    
    def get_latest(self, last_x: int = 1) -> dict:
        # not implemented yet
        raise NotImplementedError


class CriterionMetric(BaseMetric):
    """
    Takes the criterion as input and uses it as a metric.
    """
    def __init__(self, criterion: BaseLoss) -> None:
        super().__init__()
        self.criterion = criterion
        self.crit_log = dict()

    def reset(self) -> None:
        self.crit_log = dict()
    
    def calculate(self, data: dict, model_output: dict) -> dict:
        self.criterion(model_output, data)
        # use loss dict
        for key in self.criterion.loss_dict.keys():
            if key not in self.crit_log.keys():
                self.crit_log[key] = []
            self.crit_log[key].append(self.criterion.loss_dict[key].detach().cpu().data.numpy())

        return self.criterion.loss_dict

    def get_latest(self, last_x: int = 1) -> dict:
        # for each key in crit log take the mean
        return {key: np.mean(self.crit_log[key][-last_x:]) for key in self.crit_log.keys()}
    
    def get_mean(self) -> dict:
        # for each key in crit log take the mean
        return {key: np.mean(self.crit_log[key]) for key in self.crit_log.keys()}


class V2V4RealLidarMetric(CriterionMetric):
    def __init__(self, criterion: BaseLoss) -> None:
        super().__init__(criterion)
    
    def calculate(self, data: dict, model_output: dict) -> dict:
        # calculate loss (add batch dimension which must be of shape 1)
        data = {key: data[key][0][-1] for key in data}

        _data = {
            key: data[key][None] if isinstance(data[key], torch.Tensor) else data[key]
            for key in data
        }

        return super().calculate(_data, model_output)
