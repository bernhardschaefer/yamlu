import math
import unittest

import numpy as np

from yamlu.bb import iou_matrix, bbs_distances, iou_vector


class TestBoundingBoxes(unittest.TestCase):
    def test_iou_matrix_no_overlap(self):
        bbs1 = np.array([
            [10, 10, 20, 20],
            [10, 10, 20, 20],
        ])
        bbs2 = np.array([
            [50, 50, 100, 100],
            [50, 50, 100, 100]
        ])
        ious = iou_matrix(bbs1, bbs2)
        assert ious.shape == (2, 2)
        assert np.all(ious == 0.)

    def test_iou_matrix(self):
        bbs1 = np.array([
            [10, 10, 19, 19],
            [10, 10, 29, 29]
        ])
        bbs2 = np.array([
            [10, 10, 19, 19],
            [0, 0, 19, 19],
            [50, 50, 100, 100]
        ])
        ious = iou_matrix(bbs1, bbs2)
        assert ious.shape == (2, 3)
        assert ious[0, 0] == 1.
        assert ious[0, 1] == .25
        assert ious[0, 2] == 0.
        assert ious[1, 0] == .25
        assert ious[1, 1] == 1 / 7
        assert ious[1, 2] == 0

    def test_iou_vector(self):
        bbs1 = np.array([
            [10, 10, 19, 19],
            [0, 0, 39, 39]
        ])
        bbs2 = np.array([
            [10, 10, 19, 19],
            [0, 0, 19, 19],
        ])
        ious = iou_vector(bbs1, bbs2)
        assert ious.shape == (2,)
        assert ious[0] == 1.
        assert ious[1] == .25

    def test_bbs_distances(self):
        bbs1 = np.array([
            [10, 10, 19, 19],
            [10, 10, 29, 29]
        ])
        bbs2 = np.array([
            [10, 10, 19, 19],
            [0, 0, 19, 19],
            [50, 50, 100, 100]
        ])
        distances = bbs_distances(bbs1, bbs2)
        assert distances.shape == (2, 3)
        assert distances[0, 0] == 0
        assert distances[0, 1] == 0
        assert distances[0, 2] == math.sqrt(30 ** 2 + 30 ** 2)
        assert distances[1, 0] == 0
        assert distances[1, 1] == 0
        assert distances[1, 2] == math.sqrt(20 ** 2 + 20 ** 2)

    def test_bbs_distances_intersect(self):
        bbs1 = np.array([
            [10, 0, 19, 49]
        ])
        bbs2 = np.array([
            [0, 10, 14, 29]
        ])
        distances = bbs_distances(bbs1, bbs2)
        assert distances.shape == (1, 1)
        assert distances[0, 0] == 0


if __name__ == "__main__":
    unittest.main()
