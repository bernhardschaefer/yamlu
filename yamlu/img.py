import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_img(img, vmin=0, vmax=255, cmap="gray", figsize=None, save_path=None):
    height, width = img.shape

    if not figsize:
        dpi = mpl.rcParams['figure.dpi']
        figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

    if save_path:
        plt.savefig(save_path, cmap=cmap)


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
