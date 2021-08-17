import numpy as np
from scipy.sparse.csgraph import connected_components

from yamlu.img import AnnotatedImage


def transcribe_words(word_ai: AnnotatedImage, word_idxs: np.array, line_thresh: float, line_delim="\n",
                     direction: str = "lr") -> str:
    """
    Transcribe a text block given words and their bounding boxes

    Args:
        word_ai: annotated image with word annotations that have a text attribute
        word_idxs: the annotation indices to transcribe
        line_thresh: max y_mid dist in px such that two word bounding boxes are still considered to be on the same line
        line_delim: delimiter character for multiple lines
        direction: reading direction, one of lr, rl, tb, bt

    Returns:
        the transcribed text block as string

    """
    assert direction in {"lr", "rl", "tb", "bt"}, f"Invalid direction: {direction}"

    xc, yc = np.array([word_ai[i].bb.center for i in word_idxs]).T

    line_split_axis = yc if direction in ["lr", "rl"] else xc
    word_split_axis = xc if direction in ["lr", "rl"] else yc

    adj_matrix = np.abs(line_split_axis[np.newaxis, :] - line_split_axis[:, np.newaxis]) <= line_thresh
    n_components, labels = connected_components(adj_matrix, directed=False)

    line_means = np.array([line_split_axis[labels == i].mean() for i in range(n_components)])
    labels_ordered = line_means.argsort() if direction in ["lr", "bt"] else line_means.argsort()[::-1]

    lines = []
    for i in labels_ordered:
        word_mask_line = labels == i
        word_idxs_line = word_idxs[word_mask_line]
        word_order = word_split_axis[word_mask_line].argsort()
        word_order = word_order if direction in ["lr", "tb"] else np.flip(word_order, [0])
        sorted_word_idxs = word_idxs_line[word_order]
        if not len(sorted_word_idxs.shape):
            sorted_word_idxs = sorted_word_idxs[..., np.newaxis]
        line = " ".join([word_ai[wi].category for wi in sorted_word_idxs])
        lines.append(line)

    return line_delim.join(lines)
