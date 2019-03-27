import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def regression_metrics_npy(y_test, y_pred, col_names):
    y_test_df = pd.DataFrame(y_test, columns=col_names)
    y_pred_df = pd.DataFrame(y_pred, columns=col_names)
    return regression_metrics(y_test_df, y_pred_df)


def regression_metrics(y_test, y_pred, name=""):
    if y_test.ndim > 1 and y_test.shape[1] > 1:  # multiple output scenario
        # recursively call function for each column and combine results
        return pd.concat(regression_metrics(y_test[col], y_pred[col], name=col) for col in y_test.columns)

    # extract values if necessary
    if type(y_test) == pd.DataFrame and y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]
    if type(y_pred) == pd.DataFrame and y_pred.shape[1] == 1:
        y_pred = y_pred.iloc[:, 0]
    if type(y_test) == pd.Series:
        y_test = y_test.values
    if type(y_pred) == pd.Series:
        y_pred = y_pred.values

    res_df = pd.DataFrame({
        'target': name,
        'y_test': y_test,
        'y_pred': y_pred,
        'ae': np.abs(y_test - y_pred),
        'se': (y_test - y_pred) ** 2
    })
    res_df['ape'] = (res_df['ae'] / res_df.y_test) * 100.
    mse = np.mean(res_df.se)
    mae = np.mean(res_df.ae)
    mape = np.mean(res_df.ape)
    median_ae = np.median(res_df.se)
    return pd.Series(dict(mse=mse, mae=mae, mape=mape, median_ae=median_ae), name=name).to_frame().T
