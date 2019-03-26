import math
import matplotlib.pyplot as plt
import numpy as np


def plot_imgs(imgs: np.array, ncols=5, figsize=(20, 8), cmap="gray", axis_off=True):
    """
    :param imgs: batch of imgs with shape (batch_size, h, w) or (batch_size, h*w)
    """
    n_imgs = len(imgs)
    assert n_imgs < 100

    if imgs.ndim == 2:
        s = int(math.sqrt(imgs.shape[-1]))
        assert s ** 2 == imgs.shape[-1], "Second dimension does not have equal width & height"
        imgs = imgs.reshape(-1, s, s)

    nrows = math.ceil(n_imgs / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for i, img in enumerate(imgs):
        if nrows == 1:
            ax = axs[i]
        else:
            ax = axs[i // ncols, i % ncols]
        if axis_off:
            ax.axis('off')
        ax.imshow(img, cmap=cmap)

    fig.tight_layout()
