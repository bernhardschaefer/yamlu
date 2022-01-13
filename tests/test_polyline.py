import math

import numpy as np
import torch

from yamlu.polyline import pairwise_segment_intersect, pairwise_segment_distance, pairwise_point_segment_distance


def test_pairwise_segment_intersect():
    line_segments = torch.tensor([
        [10, 5, 10, 15],
        [10, 20, 10, 30],
        [30, 20, 30, 30],
        [5, 5, 15, 15],
        [0, 0, 20, 0],
    ], dtype=torch.float32)
    # (0,1,2) are parallel
    # (0,1) even have the same x-coordinate, but do not intersect
    # (0,3) intersect
    intersect_matrix = pairwise_segment_intersect(line_segments, line_segments[:-1])
    assert intersect_matrix.nonzero().tolist() == [[0, 3], [3, 0]]


def test_pairwise_segment_distance():
    line_segments = torch.tensor([
        [10, 5, 10, 15],
        [10, 20, 10, 30],
        [30, 20, 30, 30],
        [5, 5, 15, 15],
        [0, 0, 20, 0],
    ], dtype=torch.float32)

    dist_matrix = pairwise_segment_distance(line_segments, line_segments[:-1])

    assert torch.diagonal(dist_matrix).eq(0).all()
    assert dist_matrix[0, 1] == 5  # segments on the same x-axis line, so distance should be 20-15 == 5
    assert np.isclose(dist_matrix[0, 2], math.hypot(10 - 30, 15 - 20))  # closest two points
    assert dist_matrix[0, 3] == 0  # intersect
    assert dist_matrix[1, 2] == 20  # x-axis parallel segments

    print(dist_matrix)


def test_pairwise_point_segment_distance():
    pts = torch.tensor([[20, 50]], dtype=torch.float32)
    xyxy = torch.tensor([[10, 0, 10, 100]], dtype=torch.float32)

    dist_matrix = pairwise_point_segment_distance(pts, xyxy)
    assert dist_matrix.squeeze() == 10.
