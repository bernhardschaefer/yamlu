import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import patheffects

from yamlu.bb import bbs_ious


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
        assert self.t >= 0, f"Invalid bounding box coordinates: {self}"
        assert self.l >= 0, f"Invalid bounding box coordinates: {self}"
        assert self.b >= 0, f"Invalid bounding box coordinates: {self}"
        assert self.r >= 0, f"Invalid bounding box coordinates: {self}"
        assert self.t <= self.b, f"Invalid bounding box coordinates: {self}"
        assert self.l <= self.r, f"Invalid bounding box coordinates: {self}"

    @property
    def tlbr(self):
        return self.t, self.l, self.b, self.r

    @property
    def w(self):
        # bounding boxes are inclusive, which e.g. means l=10 and r=11 is 2px width
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

    def iou(self, bb):
        bbs1 = np.array(self.tlbr).reshape(1, -1)
        bbs2 = np.array(bb.tlbr).reshape(1, -1)
        return bbs_ious(bbs1, bbs2).item()

    def union(self, bb):
        return BoundingBox(t=min(self.t, bb.t), l=min(self.l, bb.l), b=max(self.b, bb.b), r=max(self.r, bb.r))

    def shrink(self, pad) -> "BoundingBox":
        return BoundingBox(self.t + pad, self.l + pad, self.b - pad, self.r - pad)

    def rotate(self, angle, img_size):
        assert angle % 90 == 0 and angle >= 0, f"Invalid angle: {angle}"
        img_w, img_h = img_size
        if angle == 0:
            return self
        else:
            bb = self.rot90(img_h)
            return bb.rotate(angle - 90, (img_h, img_w))

    def rot90(self, img_h) -> "BoundingBox":
        return BoundingBox(
            t=self.l,
            l=img_h - self.b - 1,
            b=self.r,
            r=img_h - self.t - 1
        )


@dataclass(eq=True, frozen=True)
class Annotation:
    category: str  # class label
    bb: BoundingBox
    img: np.ndarray = field(default=None, repr=False, compare=False)  # optional img for just this annotation
    # TODO use extra_fields
    text: str = field(default=None, compare=False)  # optional: only for text category
    head: Tuple[float, float] = field(default=None, compare=False)  # optional: only for arrow category
    tail: Tuple[float, float] = field(default=None, compare=False)  # optional: only for arrow category
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

    def plot(self, figsize=None, with_bb=True, with_head_tail=True, with_index=False, axis_opt="off"):
        plot_ann_img(self, figsize=figsize, with_bb=with_bb, with_head_tail=with_head_tail,
                     with_index=with_index, axis_opt=axis_opt)

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


def plot_ann_img(ann_img: AnnotatedImage, figsize, with_bb=True, with_head_tail=True, with_index=True, axis_opt="off"):
    fig, ax = plot_img(ann_img.img, figsize=figsize, axis_opt=axis_opt)

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


def plot_img(img: Union[np.ndarray, Image.Image], cmap="gray", interpolation="bilinear", alpha=None, vmin=0, vmax=255,
             axis_opt="off", extent=None, figsize=None, save_path=None) -> Tuple[plt.Figure, plt.Axes]:
    if isinstance(img, Image.Image):
        # noinspection PyTypeChecker
        img = np.asarray(img)

    if not figsize:
        figsize = figsize_from_img(img)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis(axis_opt)
    ax.imshow(img, cmap=cmap, interpolation=interpolation, alpha=alpha, vmin=vmin, vmax=vmax, extent=extent)

    if save_path:
        plt.savefig(save_path, cmap=cmap)

    return fig, ax


def plot_imgs(imgs: Union[np.ndarray, List[np.ndarray]], ncols=4, img_size=(5, 5), cmap="gray", axis_opt="off",
              vmin=None, vmax=None, titles: List[str] = None):
    """
    :param imgs: batch of imgs with shape (batch_size, h, w) or (batch_size, h, w, 3)
    :param ncols: number of columns
    :param img_size: matplotlib size to use for each image
    :param cmap: matplotlib colormap
    :param axis_opt: plot axis or not ("off"/"on")
    """
    n_imgs = len(imgs)
    assert n_imgs < 100

    if titles is not None:
        assert len(imgs) == len(titles), f"{len(imgs)} != {len(titles)}"

    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()

    nrows = math.ceil(n_imgs / ncols)
    img_w, img_h = img_size
    figsize = img_w * ncols, img_h * nrows

    nrows = math.ceil(n_imgs / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for i, img in enumerate(imgs):
        if nrows == 1 or ncols == 1:
            ax: plt.Axes = axs[i]
        else:
            ax: plt.Axes = axs[i // ncols, i % ncols]
        ax.axis(axis_opt)
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        if titles is not None:
            ax.set_title(titles[i])

    # this is quite slow:
    # fig.tight_layout()


def plot_img_paths(img_paths: Union[List[Path], List[str]], ncols=4, img_size=(5, 5)):
    imgs = [np.asarray(Image.open(p)) for p in img_paths]
    return plot_imgs(imgs, ncols=ncols, img_size=img_size, titles=[Path(p).name for p in img_paths])


def figsize_from_img(img: Union[Image.Image, np.ndarray]):
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
        return figsize_from_wh(w, h)
    elif isinstance(img, Image.Image):
        return figsize_from_wh(*img.size)
    else:
        raise ValueError(f"Unknown image type: {type(img)}")


def figsize_from_wh(img_w, img_h):
    # What size does the figure need to be in inches to fit the image?
    dpi = mpl.rcParams['figure.dpi']
    return img_w / float(dpi), img_h / float(dpi)
