import unittest

import numpy as np
import torch

from yamlu.regression import regression_metrics


class TestRegressionMetrics(unittest.TestCase):

    def test_torch_perfect_predictions(self):
        y_true = torch.tensor([[0, 5], [1, 2], [1, 4]])
        y_pred = torch.tensor([[0, 5], [1, 2], [1, 4]])
        target_names = ["height", "width"]
        metrics = regression_metrics(y_true, y_pred, target_names=target_names)
        for m in ["mse", "mae"]:
            assert np.all(metrics[m] == 0)
