import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import patheffects


@dataclass(eq=True, frozen=True)
class BoundingBox:
    """
    This Bounding Box implementation assumes the coordinates to be inclusive.
    This means that e.g. the width of the bbox is right-left+1.
    For some background see: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L23
    """

    # top, left, bottom, right
    t: float
    l: float
    b: float
    r: float

    def __post_init__(self):
        assert self.t >= 0, f"Invalid bouding box coordinates: {self}"
        assert self.l >= 0, f"Invalid bouding box coordinates: {self}"
        assert self.b > 0, f"Invalid bouding box coordinates: {self}"
        assert self.r > 0, f"Invalid bouding box coordinates: {self}"
        assert self.t < self.b, f"Invalid bouding box coordinates: {self}"
        assert self.l < self.r, f"Invalid bouding box coordinates: {self}"

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

    def is_within_bb(self, bb):
        t, l, b, r = self.tlbr
        return t >= bb.t and l >= bb.l and b <= bb.b and r <= bb.r

    def union(self, bb):
        return BoundingBox(t=min(self.t, bb.t), l=min(self.l, bb.l), b=max(self.b, bb.b), r=max(self.r, bb.r))

    def shrink(self, pad) -> "BoundingBox":
        return BoundingBox(self.t + pad, self.l + pad, self.b - pad, self.r - pad)


@dataclass(eq=True, frozen=True)
class Annotation:
    category: str  # class label
    bb: BoundingBox
    img: np.ndarray = field(default=None, repr=False, compare=False)  # optional img for just this annotation
    # TODO use extra_fields
    text: str = field(default=None, compare=False)  # optional: only for text category
    head: Tuple[int, int] = field(default=None, compare=False)  # optional: only for arrow category
    tail: Tuple[int, int] = field(default=None, compare=False)  # optional: only for arrow category
    xml_id: str = field(default=None)


@dataclass
class AnnotatedImage:
    filename: str
    width: int
    height: int
    annotations: List[Annotation]
    img: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        if not hasattr(self, "img") or self.img is None:
            self.reload_img()

    @property
    def arrows(self):
        return [a for a in self.annotations if a.category == 'arrow']

    @property
    def nodes(self):
        return [a for a in self.annotations if a.category not in ['text', 'arrow']]

    def reload_img(self):
        # 255 <-> BLACK
        img = np.full((self.height, self.width, 3), fill_value=255, dtype=np.uint8)
        for a in self.annotations:
            img = np.minimum(a.img, img)
        self.img = img

    def plot(self, figsize=None, with_bb=True, with_head_tail=True, with_index=False, axis_off=True):
        plot_ann_img(self, figsize=figsize, with_bb=with_bb, with_head_tail=with_head_tail,
                     with_index=with_index, axis_off=axis_off)

    def save(self, imgs_path: Path):
        img = Image.fromarray(self.img)
        img.save(imgs_path / self.filename)


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


def plot_ann_img(ann_img: AnnotatedImage, figsize, with_bb=True, with_head_tail=True, with_index=True, axis_off=True):
    fig = plt.figure(figsize=figsize)
    ax: plt.Axes = fig.add_axes([0, 0, 1, 1])
    if axis_off:
        ax.axis('off')

    ax.imshow(ann_img.img)

    if with_bb:
        plot_anns(ax, ann_img.annotations, with_index=with_index)

    if with_head_tail:
        # get heads and arrows as np arrays with shape (N,2), where N is number of arrows
        heads_tails = list(
            (a.head, a.tail) for a in ann_img.annotations if a.category == 'arrow' and a.head is not None)
        if len(heads_tails) > 0:
            heads, tails = list(map(np.array, zip(*heads_tails)))
            ax.scatter(*heads.T, s=50, zorder=10, alpha=.8, color="green")
            ax.scatter(*tails.T, s=50, zorder=10, alpha=.8, color="SkyBlue")


def plot_anns(ax, annotations: List[Annotation], with_index=False):
    ann_colors = compute_colors_for_annotations(annotations)

    # very rough estimate
    fontsize = ax.figure.get_size_inches()[0]
    lw = fontsize / 10

    for i, ann, color in zip(range(len(annotations)), annotations, ann_colors):
        patch = mpatches.Rectangle(*ann.bb.xy_w_h, fill=True, facecolor=color, edgecolor=color, lw=0, alpha=.05)
        ax.add_patch(patch)
        patch = mpatches.Rectangle(*ann.bb.xy_w_h, fill=False, facecolor="none", edgecolor=color, lw=lw, alpha=.8)
        ax.add_patch(patch)

        text = ann.category
        if with_index:
            text += f" {i}"
        txt = ax.text(ann.bb.l, ann.bb.t, text, verticalalignment='bottom', color=color, fontsize=fontsize,
                      alpha=.5)
        txt.set_path_effects([patheffects.Stroke(linewidth=1, foreground='BLACK'), patheffects.Normal()])


def plot_img(img, vmin=0, vmax=255, cmap="gray", figsize=None, save_path=None):
    height, width = img.shape[:2]

    if not figsize:
        dpi = mpl.rcParams['figure.dpi']
        figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

    if save_path:
        plt.savefig(save_path, cmap=cmap)

    return fig, ax


def plot_imgs(imgs: np.array, ncols=5, figsize=(20, 8), cmap="gray", axis_off=True, vmin=None, vmax=None):
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
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

    fig.tight_layout()
