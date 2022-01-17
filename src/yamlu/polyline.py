import numpy as np
import torch


def line_segment_lengths(points: np.ndarray) -> np.ndarray:
    """
    Computes the segment lenghts of a polyline represented by a sequence of points
    Args:
        points: point sequence of shape (N,2)

    Returns: segment lengths of shape (N-1,)
    """
    assert points.ndim == 2 and points.shape[1] == 2, f"{points.shape}"
    # the trick here is np.diff, which computes difference between points i and i-1
    return np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))


def pairwise_segment_intersect(xyxy1: torch.Tensor, xyxy2: torch.Tensor) -> torch.Tensor:
    """
    Computes a bool matrix where A(i,j) tracks if line segments xyxy1[i] and xyxy2[j] intersect.
    Returns False for parallel line segments, even if they are on the same line.
    Args:
        xyxy1: line segments x11, y11, x12, y12, i.e. x11y11 -> x12y12
        xyxy2: line segments x21, y21, x22, y22, i.e. x21y21 -> x22y22
    """
    # pairwise vectorized version of https://stackoverflow.com/a/2824596
    assert xyxy1.ndim == xyxy2.ndim == 2
    assert xyxy1.shape[1] == xyxy2.shape[1] == 4
    x11, y11, x12, y12 = [c.unsqueeze(1) for c in xyxy1.unbind(1)]
    x21, y21, x22, y22 = [c.unsqueeze(0) for c in xyxy2.unbind(1)]

    dx1 = x12 - x11
    dy1 = y12 - y11
    dx2 = x22 - x21
    dy2 = y22 - y21
    delta = (dy1 * dx2 - dx1 * dy2).to(torch.float)

    # Note: for parallel segments delta == 0, which results in NaN's for s and t
    # these are evaluated to False in the conditions below
    s = (dx1 * (y21 - y11) + dy1 * (x11 - x21)) / delta
    t = (dx2 * (y11 - y21) + dy2 * (x21 - x11)) / -delta
    intersect_matrix = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)

    return intersect_matrix


def pairwise_segment_distance(xyxy1: torch.Tensor, xyxy2: torch.Tensor):
    """
    Computes a bool matrix where A(i,j) is the distance between the segments xyxy1[i] and xyxy2[j].
    Args:
        xyxy1: line segments x11, y11, x12, y12, i.e. x11y11 -> x12y12
        xyxy2: line segments x21, y21, x22, y22, i.e. x21y21 -> x22y22
    """
    # pairwise vectorized version of https://stackoverflow.com/a/2824596
    # that also handles distances between segments that are on the same line
    assert xyxy1.ndim == xyxy2.ndim == 2
    assert xyxy1.shape[1] == xyxy2.shape[1] == 4
    M, N = len(xyxy1), len(xyxy2)
    if M == 0 or N == 0:
        return xyxy1.new_zeros((M, N), dtype=torch.float32)

    # try each of the 4 vertices w/the other segment
    dist_matrices = torch.stack([
        pairwise_point_segment_distance(xyxy1[:, [0, 1]], xyxy2),
        pairwise_point_segment_distance(xyxy1[:, [2, 3]], xyxy2),
        pairwise_point_segment_distance(xyxy2[:, [0, 1]], xyxy1).T,
        pairwise_point_segment_distance(xyxy2[:, [2, 3]], xyxy1).T,
    ])
    dist_matrix, _ = torch.min(dist_matrices, dim=0)

    intersect_matrix = pairwise_segment_intersect(xyxy1, xyxy2)
    dist_matrix[intersect_matrix] = 0.0
    assert dist_matrix.shape == (M, N)
    return dist_matrix


def pairwise_point_segment_distance(pts: torch.Tensor, xyxy: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pts: x,y points of shape Mx2
        xyxy: line segments x11, y11, x12, y12, i.e. x11y11 -> x12y12 in shape Nx4

    Returns: segment distances MxN
    """
    px, py = [c.unsqueeze(1) for c in pts.unbind(1)]
    x1, y1, x2, y2 = [c.unsqueeze(0) for c in xyxy.unbind(1)]

    dx = x2 - x1
    dy = y2 - y1

    # Calculate the t that minimizes the distance.
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    # See if this represents one of the segment's
    # end points or a point in the middle.
    dx_res = xyxy.new_zeros((pts.shape[0], xyxy.shape[0]), dtype=torch.float32)
    dy_res = dx_res.clone()

    # mask: t < 0 or segment is a point (i.e. t == NaN)
    segment_is_point_mask = (dx == 0) & (dy == 0)
    mask = (t < 0) | segment_is_point_mask
    dx_res[mask] = (px - x1)[mask]
    dy_res[mask] = (py - y1)[mask]

    mask = t > 1
    dx_res[mask] = (px - x2)[mask]
    dy_res[mask] = (py - y2)[mask]

    mask = (t >= 0) & (t <= 1)
    near_x = x1 + t * dx
    near_y = y1 + t * dy
    dx_res[mask] = (px - near_x)[mask]
    dy_res[mask] = (py - near_y)[mask]

    # torch.hypot only implemented in torch >= 1.7
    return torch.hypot(dx_res, dy_res) if hasattr(torch, "hypot") else torch.sqrt(dx_res ** 2 + dy_res ** 2)


def sample_equidistant_points(points: np.ndarray, k: int) -> np.ndarray:
    """
    Resamples a line defined through a sequence of points by generating k equidistant points.
    Args:
        points: polyline of shape (N,2)
        k: the number of equidistant points to sample

    Returns: the sequence of resampled points with shape (k,2)

    """
    segment_lengths = line_segment_lengths(points)
    assert any([l > 0 for l in segment_lengths]), f"Polyline with zero-length: {points}"

    sampled_line_distances = np.linspace(0, np.sum(segment_lengths), k)

    segment_idx = 0
    resampled_pts = [points[0]]

    prev_pt_line_dist = 0
    next_pt_line_dist = segment_lengths[segment_idx]

    # TODO this would be more readable with bisect + list comprehension
    for i in range(1, k - 1):
        line_dist = sampled_line_distances[i]
        # print(f"i={i}, line_dist={line_dist}, segment_idx={segment_idx}, next_pt_line_dist={next_pt_line_dist}")

        # find segment that contains pt specified through line distance
        while line_dist > next_pt_line_dist:
            # print("segment_idx++")
            segment_idx += 1
            prev_pt_line_dist = next_pt_line_dist
            next_pt_line_dist += segment_lengths[segment_idx]

        # calculate factor for segment vector, i.e. where in segment is point located
        ratio = (line_dist - prev_pt_line_dist) / (next_pt_line_dist - prev_pt_line_dist)
        assert 0 < ratio <= 1, f"{ratio}"

        p_i = points[segment_idx]
        p_j = points[segment_idx + 1]
        segment_vect = p_j - p_i

        pt = p_i + ratio * segment_vect
        resampled_pts.append(pt)
        # print(f"p_i={p_i} + ratio={ratio} * (p_j={p_j} - p_i={p_i}) => {pt}")

    resampled_pts.append(points[-1])

    resampled_pts = np.array(resampled_pts)
    return resampled_pts
