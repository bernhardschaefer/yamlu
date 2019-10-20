"""Bounding Box methods"""

import numpy as np


# https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
def bbs_ious(bboxes1: np.ndarray, bboxes2: np.ndarray):
    """Vectorized iou matrix calculation for two arrays of bounding boxes with t,l,b,r convention"""
    assert bboxes1.shape[1] == 4
    assert bboxes2.shape[1] == 4
    t1, l1, b1, r1 = np.split(bboxes1, 4, axis=1)
    t2, l2, b2, r2 = np.split(bboxes2, 4, axis=1)

    t_max = np.maximum(t1, np.transpose(t2))
    l_max = np.maximum(l1, np.transpose(l2))
    b_min = np.minimum(b1, np.transpose(b2))
    r_min = np.minimum(r1, np.transpose(r2))

    interArea = np.maximum((b_min - t_max + 1), 0) * np.maximum((r_min - l_max + 1), 0)
    boxAArea = (b1 - t1 + 1) * (r1 - l1 + 1)
    boxBArea = (b2 - t2 + 1) * (r2 - l2 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


# inspired by:
# https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares
def bbs_distances(bboxes1: np.ndarray, bboxes2: np.ndarray):
    """Vectorized axis-aligned bounding box distance calculation between two arrays of t,l,b,r bounding boxes.
    Assumes inclusive bb coordinates (this is why e.g. width is calc. as r-l+1).
    """
    t1, l1, b1, r1 = np.split(bboxes1, 4, axis=1)
    t2, l2, b2, r2 = np.split(bboxes2, 4, axis=1)

    t_min = np.minimum(t1, t2.T)
    l_min = np.minimum(l1, l2.T)
    b_max = np.maximum(b1, b2.T)
    r_max = np.maximum(r1, r2.T)

    bb_outer_widths = r_max - l_min + 1
    bb_outer_heights = b_max - t_min + 1

    bbs1_widths = r1 - l1 + 1
    bbs1_heights = b1 - t1 + 1

    bbs2_widths = r2 - l2 + 1
    bbs2_heights = b2 - t2 + 1

    inner_width = np.maximum(0, bb_outer_widths - bbs1_widths - bbs2_widths.T)
    inner_height = np.maximum(0, bb_outer_heights - bbs1_heights - bbs2_heights.T)
    min_distance = np.sqrt(inner_width ** 2 + inner_height ** 2)
    return min_distance
