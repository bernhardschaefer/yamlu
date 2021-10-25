import itertools
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


def cl_report(y_true, y_pred, classes: List[str], digits=3, only_avg=False) -> pd.DataFrame:
    # micro AP == micro AR == accuracy in a multi-class setting (https://datascience.stackexchange.com/a/29054)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        class_rep = metrics.classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                                  target_names=classes, digits=digits, output_dict=True)
    cr_df = pd.DataFrame(class_rep).T
    for col in ['f1-score', 'precision', 'recall']:
        cr_df[col] = cr_df[col].round(decimals=digits)
    if only_avg:
        cr_df = cr_df[cr_df.index.str.contains("avg")]
    return cr_df


def plot_roc_auc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
                                             drop_intermediate=True)
    roc_auc = metrics.roc_auc_score(y_true, y_score)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_cm(y_true, y_pred, classes, figsize=(6, 6)):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=classes, normalize=None)
    plot_confusion_matrix(cm, classes, figsize=figsize)


def plot_confusion_matrix(cm, display_labels, normalize=False, cmap="Blues", figsize=(8, 8),
                          xticks_rotation="vertical", colorbar=True, plot_zero=True, xy_label=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Difference from sklearn.metrics.ConfusionMatrixDisplay:
    - zeros: uses alpha and allows not plotting zeros
    - colorbar: allows disabling colorbar
    """
    # with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if colorbar:
        ax.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cnt = cm[i, j]
        if cnt == 0 and not plot_zero:
            continue
        ax.text(j, i, format(cnt, fmt), alpha=1 if cnt > 0 else .3, ha="center", va="center",
                color="white" if cnt > thresh else "black")

    n_classes = cm.shape[0]
    ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=display_labels, yticklabels=display_labels)
    if xy_label:
        ax.set(ylabel="True label", xlabel="Predicted label")

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    if xticks_rotation == 45:
        plt.setp(ax.get_xticklabels(), ha="right")

    # otherwise x/y labels are sometimes cut off when saving figure
    plt.tight_layout()


def confusion_classes(cm, le):
    n = cm.shape[0]
    cm_nondiag = np.where(np.eye(n), -1, cm)  # hack: set diagonal entries to -1
    true_le, pred_le = np.where(cm_nondiag > 0)

    return pd.DataFrame({
        'support': [cm[i, j] for i, j in zip(true_le, pred_le)],
        "pred_le": pred_le,
        "true_le": true_le,
        "pred_label": le.inverse_transform(pred_le),
        "true_label": le.inverse_transform(true_le)
    }).sort_values("support", ascending=False)


def test_cm():
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    cm_test = metrics.confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm_test, ["a", "b", "c"], figsize=(3, 3))
    le_test = LabelEncoder()
    le_test.fit_transform(["a", "b", "c"])
    confusion_classes(cm_test, le_test)
