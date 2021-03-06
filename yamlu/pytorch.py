import warnings
from typing import Union, Collection

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


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
