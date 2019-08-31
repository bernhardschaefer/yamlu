import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import patheffects


@dataclass
class BoundingBox:
    """
    This Bounding Box implementation assumes the coordinates to be inclusive.
    This means that e.g. the width of the bbox is right-left+1.
    """

    # top, left, bottom, right
    t: float
    l: float
    b: float
    r: float

    @classmethod
    def from_xyxy(cls, xmin, ymin, xmax, ymax):
        """ xyxy convention is commonly x<->width and y<->height"""
        return cls(ymin, xmin, ymax, xmax)

    @property
    def tlbr(self):
        return self.t, self.l, self.b, self.r

    @property
    def w(self):
        return self.r - self.l + 1

    @property
    def h(self):
        return self.b - self.t + 1

    @property
    def size(self):
        return self.w * self.h

    @property
    def tb_mid(self):
        return self.t + self.h / 2

    @property
    def lr_mid(self):
        return self.l + self.w / 2

    @property
    def xy_w_h(self):
        return (self.l, self.t), self.w, self.h

    @property
    def bb_coco(self):
        return self.l, self.t, self.w, self.h


@dataclass
class Annotation:
    category: str
    bb: BoundingBox
    text: str = field(default=None)  # only for text category


@dataclass
class AnnotatedImage:
    filename: str
    width: int
    height: int
    annotations: List[Annotation]
    img: Optional[np.ndarray] = field(repr=False)

    def plot(self, figsize, with_bb=True):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.img)

        if with_bb:
            plot_bb(ax, self.annotations)
        return fig, ax

    def save(self, imgs_path: Path):
        img = Image.fromarray(self.img)
        img.save(imgs_path / self.filename)

    def del_image_data(self):
        """Delete img and annotation img data and only keep metadata to reduce memory usage"""
        self.img = None
        for ann in self.annotations:
            ann.img = None


def compute_colors_for_annotations(annotations, cmap='jet'):
    categories = set(a.category for a in annotations)
    cat_to_id = dict((c, i) for i, c in enumerate(categories))
    cat_ids = np.array([cat_to_id[ann.category] for ann in annotations])

    values = range(len(categories))

    cm = plt.get_cmap(cmap)
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    scalarMap.get_clim()

    return [scalarMap.to_rgba(c) for c in cat_ids]


def plot_bb(ax, annotations: List[Annotation]):
    ann_colors = compute_colors_for_annotations(annotations)

    for ann, color in zip(annotations, ann_colors):
        patch = mpatches.Rectangle(*ann.bb.xy_w_h, fill=True, facecolor=color, edgecolor=color, lw=0, alpha=.05)
        ax.add_patch(patch)
        patch = mpatches.Rectangle(*ann.bb.xy_w_h, fill=False, facecolor="none", edgecolor=color, lw=1, alpha=.8)
        ax.add_patch(patch)

        txt = ax.text(ann.bb.l, ann.bb.t, ann.category, verticalalignment='bottom', color=color, fontsize=10,
                      alpha=.5)
        txt.set_path_effects([patheffects.Stroke(linewidth=1, foreground='BLACK'), patheffects.Normal()])


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
    :param ncols: number of columns
    :param figsize: matplotlib figsize
    :param cmap: matplotlib colormap
    :param axis_off: plot axis or not
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
