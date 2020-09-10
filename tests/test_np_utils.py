import numpy as np
import torch

from yamlu import np_utils


def test_bin_stats_np():
    arr = np.array([True, False, True, False, False])
    stat = np_utils.bin_stats(arr)
    assert stat == "2/5 (40.00%)"


def test_bin_stats_np_int():
    arr = np.array([1, 0, 1, 0, 0])
    stat = np_utils.bin_stats(arr)
    assert stat == "2/5 (40.00%)"


def test_bin_stats_torch():
    arr = torch.tensor([True, False, True, False, False])
    stat = np_utils.bin_stats(arr)
    assert stat == "2/5 (40.00%)"
