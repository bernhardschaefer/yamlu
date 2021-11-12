import numpy as np


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
