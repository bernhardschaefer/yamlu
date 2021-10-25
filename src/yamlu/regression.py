import logging
from typing import Union, List

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def regression_metrics(y_true: Union[np.ndarray, torch.Tensor],
                       y_pred: Union[np.ndarray, torch.Tensor],
                       target_names: List[str]):
    """
    Regression metric calculation that supports both multiple targets and single targets scenario.
    Result can be visualized e.g. using `metrics[["mse", "mae"]].plot.bar(figsize=(8,4));`
    :param y_true: true
    :param y_pred: predictions
    :param target_names a list of target names
    :return: a pandas DataFrame with metrics as columns and target names as rows
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    y_true_df = pd.DataFrame(y_true, columns=target_names)
    y_pred_df = pd.DataFrame(y_pred, columns=target_names)
    return _regression_metrics(y_true_df, y_pred_df)


def regression_metrics_pandas(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Regression metric calculation that supports both multiple targets and single targets scenario.
    Result can be visualized e.g. using `metrics[["mse", "mae"]].plot.bar(figsize=(8,4));`
    :param y_true: true
    :param y_pred: predictions
    :return: a pandas DataFrame with metrics as columns and target names as rows
    """
    return _regression_metrics(y_true, y_pred)


def _regression_metrics(y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
                        y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
                        name: str = "") -> pd.DataFrame:
    if y_true.ndim > 1 and y_true.shape[1] > 1:  # multiple output scenario
        # recursively call function for each column and combine results
        return pd.concat(_regression_metrics(y_true[col], y_pred[col], name=col) for col in y_true.columns)

    # extract values if necessary
    if isinstance(y_true, pd.DataFrame) and y_true.shape[1] == 1:
        y_true = y_true.iloc[:, 0]
    if isinstance(y_pred, pd.DataFrame) and y_pred.shape[1] == 1:
        y_pred = y_pred.iloc[:, 0]
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    res_df = pd.DataFrame({
        'target': name,
        'y_true': y_true,
        'y_pred': y_pred,
        'ae': np.abs(y_true - y_pred),
        'se': (y_true - y_pred) ** 2
    })
    res_df['ape'] = (res_df['ae'] / res_df.y_true) * 100.
    mse = np.mean(res_df.se)
    mae = np.mean(res_df.ae)
    mape = np.mean(res_df.ape)
    median_ae = np.median(res_df.se)
    return pd.Series(dict(mse=mse, mae=mae, mape=mape, median_ae=median_ae), name=name).to_frame().T
