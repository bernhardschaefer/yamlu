import torch

from yamlu.polyline import pairwise_segments_intersect


def test_pairwise_segments_intersect():
    line_segments = torch.tensor([
        [10, 5, 10, 15],
        [10, 20, 10, 30],
        [30, 20, 30, 30],
        [5, 5, 15, 15],
        [0, 0, 20, 0],
    ])
    # (0,1,2) are parallel
    # (0,1) even have the same x-coordinate, but do not intersect
    # (0,3) intersect
    intersect_matrix = pairwise_segments_intersect(line_segments, line_segments)
    intersect_idxs = torch.triu(intersect_matrix).nonzero()
    assert intersect_idxs.squeeze().tolist() == [0, 3]
