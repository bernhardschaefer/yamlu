"""Bounding Box methods"""
import functools

import numpy as np


def _check_bb_shape(bb_tlbr: np.ndarray):
    if len(bb_tlbr) == 0:
        bb_tlbr = np.reshape(bb_tlbr, (0, 4))
    assert bb_tlbr.shape[1] == 4, bb_tlbr.shape
    return bb_tlbr


def _get_pairwise_inter_area(bb_tlbr1: np.ndarray, bb_tlbr2: np.ndarray):
    """Pairwise vectorized bounding box intersection area computation for two arrays with t,l,b,r convention
    Args:
        bb_tlbr1: shape (m,4)
        bb_tlbr2: shape (n,4)
    Returns:
        intersection areas with shape (m,n), bb1 areas (m,), bb2 areas (n,)
    """
    bb_tlbr1 = _check_bb_shape(bb_tlbr1)
    bb_tlbr2 = _check_bb_shape(bb_tlbr2)
    t1, l1, b1, r1 = np.split(bb_tlbr1, 4, axis=1)
    t2, l2, b2, r2 = np.split(bb_tlbr2, 4, axis=1)

    t_max = np.maximum(t1, np.transpose(t2))
    l_max = np.maximum(l1, np.transpose(l2))
    b_min = np.minimum(b1, np.transpose(b2))
    r_min = np.minimum(r1, np.transpose(r2))

    inter_area = np.maximum((b_min - t_max), 0) * np.maximum((r_min - l_max), 0)
    bb1_area = (b1 - t1) * (r1 - l1)
    bb2_area = (b2 - t2) * (r2 - l2)
    return inter_area, bb1_area, bb2_area


# https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
def iou_matrix(bb_tlbr1: np.ndarray, bb_tlbr2: np.ndarray):
    """Pairwise vectorized iou matrix calculation for two arrays of bounding boxes with t,l,b,r convention
    Args:
        bb_tlbr1: shape (m,4)
        bb_tlbr2: shape (n,4)
    Returns:
        iou matrix with shape (m,n)
    """
    inter_area, bb1_area, bb2_area = _get_pairwise_inter_area(bb_tlbr1, bb_tlbr2)
    iou = inter_area / (bb1_area + np.transpose(bb2_area) - inter_area)
    return iou


def pairwise_intersection_over_area(bb_tlbr1: np.ndarray, bb_tlbr2: np.ndarray):
    """Pairwise intersection over bb1 area calculation for two arrays of bounding boxes with t,l,b,r convention
    Args:
        bb_tlbr1: shape (m,4)
        bb_tlbr2: shape (n,4)
    Returns:
        intersection over area of bb1 with shape (m,n)
    """
    inter_area, bb1_area, bb2_area = _get_pairwise_inter_area(bb_tlbr1, bb_tlbr2)
    return inter_area / bb1_area


def do_matching(match_matrix: np.ndarray, thresh: float):
    match_max = match_matrix.max(axis=1)
    bb2_idxs = match_matrix.argmax(axis=1)
    # set matched idx to -1 if below thresh
    bb2_idxs[match_max < thresh] = -1

    bb1_idxs = np.arange(match_matrix.shape[0])
    return bb1_idxs, bb2_idxs, match_max


def ioa_matching(bb_tlbr1: np.ndarray, bb_tlbr2: np.ndarray, ioa_thresh: float):
    ioa_matrix = pairwise_intersection_over_area(bb_tlbr1, bb_tlbr2)
    return do_matching(ioa_matrix, ioa_thresh)


def iou_matching(bb_tlbr1: np.ndarray, bb_tlbr2: np.ndarray, iou_thresh: float):
    return do_matching(iou_matrix(bb_tlbr1, bb_tlbr2), iou_thresh)


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
    min_distance = np.hypot(inner_width, inner_height)
    return min_distance


def pts_boxes_distance(pts, boxes_ltrb, zero_dist_pt_within_box: bool = True):
    """Distance of a point to a bounding box rectangle
    Based on Stackoverflow: https://stackoverflow.com/a/18157551/1501100
    NOTE: This is an exact solution that leverages the fact that the bounding boxes are axis-aligned
    :param zero_dist_pt_within_box: return zero distance for point in a box (True) or min. distance to all sides (False)
    """
    assert pts.ndim == 2
    assert pts.shape[1] == 2

    # this is pairwise, i.e. measure distance of all points to all boxes
    xs, ys = [p.view(-1, 1) for p in pts.T]
    x_min, y_min, x_max, y_max = [p.view(1, -1) for p in boxes_ltrb.T]

    xmin_d = x_min - xs
    xmax_d = xs - x_max
    ymin_d = y_min - ys
    ymax_d = ys - y_max

    import torch
    dx = torch.clamp(torch.max(xmin_d, xmax_d), 0)
    dy = torch.clamp(torch.max(ymin_d, ymax_d), 0)
    # torch.hypot only implemented in torch >= 1.7
    ds = torch.hypot(dx, dy) if hasattr(torch, "hypot") else torch.sqrt(dx ** 2 + dy ** 2)

    pts_in_box_mask = (ds == 0)
    if not zero_dist_pt_within_box and pts_in_box_mask.any():
        d_min, _ = torch.stack([xmin_d, xmax_d, ymin_d, ymax_d]).abs().min(dim=0)
        ds[pts_in_box_mask] = d_min[pts_in_box_mask]

    return ds


def pairwise_box_in_box(bbs_tlbr1: np.ndarray, bbs_tlbr2: np.ndarray, pad: float = 0):
    """
    Given two lists of boxes B1 and B2 with N and M boxes respectively,
    computes if box B1_i is contained in padded box B2_j between __all__ N x M pairs of boxes.
    NOTE: works for both tlbr and ltrb bounding box convention and is also PyTorch compatible
    """
    assert bbs_tlbr1.ndim == 2 and bbs_tlbr1.shape[1] == 4, bbs_tlbr1.shape
    assert bbs_tlbr2.ndim == 2 and bbs_tlbr2.shape[1] == 4, bbs_tlbr2.shape
    # torch compatibility
    bbs1 = bbs_tlbr1.unsqueeze(1) if hasattr(bbs_tlbr1, "unsqueeze") else bbs_tlbr1[:, np.newaxis, ...]
    bbs2 = bbs_tlbr2.unsqueeze(0) if hasattr(bbs_tlbr2, "unsqueeze") else bbs_tlbr2[np.newaxis, ...]
    isin_min = (bbs1[..., [0, 1]] >= (bbs2[..., [0, 1]] - pad)).all(-1)  # torch/numpy <-> dim/axis
    isin_max = (bbs1[..., [2, 3]] <= (bbs2[..., [2, 3]] + pad)).all(-1)
    b1_in_b2 = isin_min & isin_max
    assert b1_in_b2.shape == (len(bbs_tlbr1), len(bbs_tlbr2))
    return b1_in_b2


def union_boxes(boxes_ltrb):
    import torch
    lt = torch.min(boxes_ltrb[:, [0, 1]], dim=0)[0]
    rb = torch.max(boxes_ltrb[:, [2, 3]], dim=0)[0]
    return torch.cat([lt, rb])


def polygon_box_degree(box: np.ndarray):
    """
    Computes the degree of a bounding box described as a polygon with 4 points.
    Args:
        box: bounding box described through 4 corner points starting from left-top and going clockwise
    Returns: the degree of the rotated bounding box
    """
    assert box.shape == (4, 2), box.shape

    # for horizontal boxes mid1->mid2 is the line from left-mid to right-mid
    mid1 = (box[0] + box[3]) / 2.
    mid2 = (box[1] + box[2]) / 2.

    # see https://stackoverflow.com/questions/41855261/calculate-the-angle-between-a-line-and-x-axis
    # compute angle of mid1->mid2 to x-axis
    xd, yd = mid2 - mid1
    angle = np.arctan2(yd, xd)
    return -np.rad2deg(angle)


def direction_from_degree(degree):
    if -45 < degree < 45:
        return "lr"
    elif 45 <= degree < 135:
        return "bt"
    elif degree <= -135 or degree >= 135:
        return "rl"
    elif -135 < degree <= -45:
        return "tb"
    else:
        raise ValueError(f"Unexpected degree: {degree:.2f}")
