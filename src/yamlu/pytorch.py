import logging
import os
import random
import warnings
from datetime import datetime
from typing import Union, Collection

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


# based on detectron2.modeling.sampling.subsample_labels
def subsample_by_pos_fraction(pos_mask: torch.Tensor, num_samples: int, positive_fraction: float):
    """
        Return `num_samples` (or fewer, if not enough found)
        random samples from `labels` which is a mixture of positives & negatives.
        It will try to return as many positives as possible without
        exceeding `positive_fraction * num_samples`, and then try to
        fill the remaining slots with negatives.

        Args:
            pos_mask (Tensor): (N, ) vector where True indicates positive and false negative
            num_samples (int): The total number of labels with value >= 0 to return.
                Values that are not sampled will be filled with -1 (ignore).
            positive_fraction (float): The number of subsampled labels with values > 0
                is `min(num_positives, int(positive_fraction * num_samples))`. The number
                of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
                In order words, if there are not enough positives, the sample is filled with
                negatives. If there are also not enough negatives, then as many elements are
                sampled as is possible.

        Returns:
            pos_idx, neg_idx (Tensor):
                1D vector of indices. The total length of both is `num_samples` or fewer.
        """

    positive, = torch.nonzero(pos_mask, as_tuple=True)
    negative, = torch.nonzero(~pos_mask, as_tuple=True)

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    idxs = torch.cat([pos_idx, neg_idx])
    return indices_to_mask(idxs, len(pos_mask))


def indices_to_mask(indices: torch.Tensor, mask_length: int) -> torch.BoolTensor:
    assert len(indices) == 0 or mask_length > indices.max(), f"mask_length={mask_length} < max(indices)={indices.max()}"
    mask = torch.full((mask_length,), False, dtype=torch.bool, device=indices.device)
    mask[indices] = True
    # noinspection PyTypeChecker
    return mask


# https://github.com/pytorch/pytorch/issues/3025#issuecomment-392601780
# https://github.com/pytorch/pytorch/pull/26144
def isin(element: torch.Tensor, test_elements: Union[Collection, np.ndarray, torch.Tensor]):
    """see numpy isin: https://docs.scipy.org/doc/numpy/reference/generated/numpy.isin.html#numpy.isin
    """
    if not isinstance(test_elements, torch.Tensor):
        if not isinstance(test_elements, list):
            test_elements = list(test_elements)
        test_elements = torch.tensor(test_elements, dtype=element.dtype, device=element.device)
    return (element[..., None] == test_elements).any(-1)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def score(model: nn.Module, dl: DataLoader, with_loss=True):
    model.eval()

    with torch.no_grad():
        y_scores, y_true = list(zip(*((model(xb), yb) for xb, yb in dl)))
        y_scores, y_true = torch.cat(y_scores), torch.cat(y_true)
        y_pred = torch.argmax(y_scores, dim=1, keepdim=False)

        if with_loss:
            losses = F.cross_entropy(y_scores, y_true, reduction='none')
            return y_true, y_pred, losses
        else:
            return y_true, y_pred


def evaluate(model: nn.Module, valid_dl: DataLoader, split: str):
    model.eval()

    with torch.no_grad():
        y_scores, y_true = [], []
        for xb, yb in valid_dl:
            y_true.append(yb)
            y_scores.append(model(xb))
        return calc_metrics(y_true, y_scores, split)


def calc_metrics(y_true, y_scores, split: str):
    with torch.no_grad():
        # concatenate tensors in case argument is a list of batches
        if isinstance(y_true, list):
            y_true = torch.cat(y_true)
        if isinstance(y_scores, list):
            y_scores = torch.cat(y_scores)

        n_classes = 1 if y_scores.dim == 1 else y_scores.shape[1]

        loss = F.cross_entropy(y_scores, y_true).item()

        y_true = y_true.numpy()
        if n_classes > 1:
            y_pred = torch.argmax(y_scores, dim=1).numpy()
        else:
            y_pred = (y_scores > .5).numpy()

    import sklearn.metrics
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)

    average = "binary" if n_classes == 1 else "macro"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = sklearn.metrics.precision_score(y_true, y_pred, average=average)
        r = sklearn.metrics.recall_score(y_true, y_pred, average=average)

    m = {f"{split}_loss": loss,
         f"{split}_acc": acc,
         f"{split}_prec": p,
         f"{split}_rec": r}
    return m


# copied from detectron2
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/env.py
def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big")
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
