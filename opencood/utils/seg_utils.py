import math

import numpy as np


def mean_precision(eval_segm, gt_segm):
    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    mAP = [0] * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        n_ij = np.sum(curr_eval_mask)
        val = n_ii / float(n_ij)
        if math.isnan(val):
            mAP[i] = 0.
        else:
            mAP[i] = val
    # print(mAP)
    return mAP


def mean_IU(eval_segm, gt_segm, n_classes=None):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    Make sure that the classes are ordered and without gaps, e.g. 0,1,2,3,...)
    '''

    check_size(eval_segm, gt_segm)

    if n_classes is None:
        cl, n_cl = union_classes(eval_segm, gt_segm)
    else:
        cl = range(n_classes)
        n_cl = n_classes

    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    return IU


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


def calc_iou_temporal_training(batch_dict, output_dict):
    """
    Calculate IoU during training.

    Parameters
    ----------
    batch_dict: dict
        The data that contains the gt.

    output_dict : dict
        The output directory with predictions.

    Returns
    -------
    The iou for static and dynamic bev map.
    """

    # check if data is available
    if 'static_map' not in output_dict.keys() or 'dynamic_map' not in output_dict.keys():
        return None, None


    batch_size = len(batch_dict['gt_static'])

    for i in range(batch_size):
        # get last scneario ground-truth
        gt_static = batch_dict['gt_static'][i][-1][0].detach().cpu().data.numpy()
        gt_static = np.array(gt_static, dtype=np.int32)

        gt_dynamic = batch_dict['gt_dynamic'][i][-1][0].detach().cpu().data.numpy()
        gt_dynamic = np.array(gt_dynamic, dtype=np.int32)

        pred_static = output_dict['static_map'][i].detach().cpu().data.numpy()
        pred_static = np.array(pred_static, dtype=np.int32)

        pred_dynamic = output_dict['dynamic_map'][i].detach().cpu().data.numpy()
        pred_dynamic = np.array(pred_dynamic, dtype=np.int32)

        iou_dynamic = mean_IU(pred_dynamic, gt_dynamic)
        iou_static = mean_IU(pred_static, gt_static)

        return iou_dynamic, iou_static


def cal_iou_training(batch_dict, output_dict):
    """
    Calculate IoU during training.

    Parameters
    ----------
    batch_dict: dict
        The data that contains the gt.

    output_dict : dict
        The output directory with predictions.

    Returns
    -------
    The iou for static and dynamic bev map.
    """

    batch_size = batch_dict['ego']['gt_static'].shape[0]

    for i in range(batch_size):

        gt_static = \
            batch_dict['ego']['gt_static'].detach().cpu().data.numpy()[i, 0]
        gt_static = np.array(gt_static, dtype=np.int32)

        gt_dynamic = \
            batch_dict['ego']['gt_dynamic'].detach().cpu().data.numpy()[i, 0]
        gt_dynamic = np.array(gt_dynamic, dtype=np.int32)

        pred_static = \
            output_dict['static_map'].detach().cpu().data.numpy()[i]
        pred_static = np.array(pred_static, dtype=np.int32)

        pred_dynamic = \
            output_dict['dynamic_map'].detach().cpu().data.numpy()[i]
        pred_dynamic = np.array(pred_dynamic, dtype=np.int32)

        iou_dynamic = mean_IU(pred_dynamic, gt_dynamic)
        iou_static = mean_IU(pred_static, gt_static)

        return iou_dynamic, iou_static


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)