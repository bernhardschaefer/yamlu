import numpy as np
import torch

from yamlu import np_utils


def test_bin_stats_np():
    arr = np.array([True, False, True, False, False])
    stat = np_utils.bin_stats(arr)
    assert stat == "2/5 (40.00%)"


def test_bin_stats_np_int():
    arr = np.array([0, 0, 1])
    stat = np_utils.bin_stats(arr)
    assert stat == "1/3 (33.33%)"


def test_bin_stats_torch():
    arr = torch.tensor([False, False, True])
    stat = np_utils.bin_stats(arr)
    assert stat == "1/3 (33.33%)"
