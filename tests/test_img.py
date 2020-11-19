import unittest

import numpy as np

from yamlu.img import Annotation, BoundingBox


class TestAnnotation(unittest.TestCase):
    def test_annotation_repr(self):
        a = Annotation("test", BoundingBox(0, 5, 10, 10), text="foo")
        print(a)
        assert str(a) == "Annotation(category='test', bb=BoundingBox(t=0.00,l=5.00,b=10.00,r=10.00), text=foo)"


class TestBoundingBox(unittest.TestCase):
    def test_intersection(self):
        b1 = BoundingBox(0, 0, 10, 10)
        b2 = BoundingBox(0, 0, 5, 5)
        assert b1.intersection(b2) == b2
        assert b1.intersection(b1) == b1

        b3 = BoundingBox(10, 10, 15, 15)
        assert b1.intersection(b3).area == 0.

    def test_from_points(self):
        pts = np.array([
            [0, 2],
            [1, 0]
        ])
        assert BoundingBox.from_points(pts) == BoundingBox(0, 0, 3, 2)

    def test_iou(self):
        b1 = BoundingBox(0, 0, 10, 10)
        b2 = BoundingBox(0, 0, 5, 5)
        assert b1.iou(b2) == 0.25
        assert b1.iou(b1) == 1.

        b3 = BoundingBox(10, 10, 15, 15)
        assert b1.iou(b3) == 0.

        b4 = BoundingBox(5, 5, 15, 15)
        assert b1.iou(b4) == 1 / 7
