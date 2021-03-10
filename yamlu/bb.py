"""Bounding Box methods"""
import functools

import numpy as np

# https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
import torch


def iou_matrix(bb_tlbr1: np.ndarray, bb_tlbr2: np.ndarray):
    """Pairwise vectorized iou matrix calculation for two arrays of bounding boxes with t,l,b,r convention
    :param bb_tlbr1 shape (m,4)
    :param bb_tlbr2 shape (n,4)
    :return iou matrix with shape (m,n)
    """
    assert bb_tlbr1.shape[1] == 4
    assert bb_tlbr2.shape[1] == 4
    t1, l1, b1, r1 = np.split(bb_tlbr1, 4, axis=1)
    t2, l2, b2, r2 = np.split(bb_tlbr2, 4, axis=1)

    t_max = np.maximum(t1, np.transpose(t2))
    l_max = np.maximum(l1, np.transpose(l2))
    b_min = np.minimum(b1, np.transpose(b2))
    r_min = np.minimum(r1, np.transpose(r2))

    interArea = np.maximum((b_min - t_max), 0) * np.maximum((r_min - l_max), 0)
    boxAArea = (b1 - t1) * (r1 - l1)
    boxBArea = (b2 - t2) * (r2 - l2)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def iou_vector(bb_tlbr1: np.ndarray, bb_tlbr2: np.ndarray):
    """computes iou between the bounding boxes at the same index in bboxes1 and bboxes2
    :param bboxes1 shape (n,4)
    :param bboxes2 shape (n,4)
    :return iou vector with shape (n,)
    """
    assert bb_tlbr1.shape[0] == bb_tlbr2.shape[0]
    ious = [iou_matrix(b1[np.newaxis, :], b2[np.newaxis, :]).squeeze() for b1, b2 in zip(bb_tlbr1, bb_tlbr2)]
    return np.array(ious)


# inspired by:
# https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares
def bbs_distances(bboxes1: np.ndarray, bboxes2: np.ndarray):
    """Vectorized axis-aligned bounding box distance calculation between two arrays of t,l,b,r bounding boxes"""
    t1, l1, b1, r1 = np.split(bboxes1, 4, axis=1)
    t2, l2, b2, r2 = np.split(bboxes2, 4, axis=1)

    t_min = np.minimum(t1, t2.T)
    l_min = np.minimum(l1, l2.T)
    b_max = np.maximum(b1, b2.T)
    r_max = np.maximum(r1, r2.T)

    bb_outer_widths = r_max - l_min
    bb_outer_heights = b_max - t_min

    bbs1_widths = r1 - l1
    bbs1_heights = b1 - t1

    bbs2_widths = r2 - l2
    bbs2_heights = b2 - t2

    inner_width = np.maximum(0, bb_outer_widths - bbs1_widths - bbs2_widths.T)
    inner_height = np.maximum(0, bb_outer_heights - bbs1_heights - bbs2_heights.T)
    min_distance = np.sqrt(inner_width ** 2 + inner_height ** 2)
    return min_distance


def pts_boxes_distance(pts: torch.Tensor, boxes_ltrb: torch.Tensor, zero_dist_pt_within_box: bool = True):
    """Distance of a point to a bounding box rectangle
    Based on Stackoverflow: https://stackoverflow.com/a/18157551/1501100
    NOTE: This is an exact solution that leverages the fact that the bounding boxes are axis-aligned
    :param zero_dist_pt_within_box: return zero distance for point in a box (True) or min. distance to all sides (False)
    """
    assert pts.dim() == 2
    assert pts.shape[1] == 2

    # this is pairwise, i.e. measure distance of all points to all boxes
    xs, ys = [p.view(-1, 1) for p in pts.t()]
    x_min, y_min, x_max, y_max = [p.view(1, -1) for p in boxes_ltrb.t()]

    xmin_d = x_min - xs
    xmax_d = xs - x_max
    ymin_d = y_min - ys
    ymax_d = ys - y_max

    dx = torch.clamp(torch.max(xmin_d, xmax_d), 0)
    dy = torch.clamp(torch.max(ymin_d, ymax_d), 0)
    ds = torch.sqrt(dx ** 2 + dy ** 2)

    pts_in_box_mask = (ds == 0)
    if not zero_dist_pt_within_box and pts_in_box_mask.any():
        d_min = functools.reduce(torch.min, [xmin_d.abs(), xmax_d.abs(), ymin_d.abs(), ymax_d.abs()])
        ds[pts_in_box_mask] = d_min[pts_in_box_mask]

    return ds
