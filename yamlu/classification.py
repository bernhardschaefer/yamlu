import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder


def cl_report(y_true, y_pred, classes: List[str], digits=3, only_avg=False) -> pd.DataFrame:
    # micro AP == micro AR == accuracy in a multi-class setting (https://datascience.stackexchange.com/a/29054)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        class_rep = sklearn.metrics.classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                                          target_names=classes, digits=digits, output_dict=True)
    cr_df = pd.DataFrame(class_rep).T
    for col in ['f1-score', 'precision', 'recall']:
        cr_df[col] = cr_df[col].round(decimals=digits)
    if only_avg:
        cr_df = cr_df[cr_df.index.str.contains("avg")]
    return cr_df


def plot_roc_auc(y_true, y_score):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
                                                     drop_intermediate=True)
    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score)

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
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes, figsize=figsize)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=(6, 6),
                          colorbar=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    with sns.axes_style("white"):
        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        if colorbar:
            plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, ha="center")
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        import itertools
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            cnt = cm[i, j]
            plt.text(j, i, format(cnt, fmt),
                     alpha=1 if cnt > 0 else .3,
                     horizontalalignment="center",
                     color="white" if cnt > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
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
    cm_test = sklearn.metrics.confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm_test, ["a", "b", "c"], figsize=(3, 3))
    le_test = LabelEncoder()
    le_test.fit_transform(["a", "b", "c"])
    confusion_classes(cm_test, le_test)
