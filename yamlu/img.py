import logging
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

_logger = logging.getLogger(__name__)


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

    @classmethod
    def from_center_wh(cls, center, w, h):
        x, y = center
        return cls(t=y - h / 2, l=x - w / 2, b=y + h / 2, r=x + w / 2)

    @classmethod
    def from_pascal_voc(cls, l, t, r, b):
        return cls(t, l, b, r)

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

    @property
    def bb_pascal_voc(self):
        """
        :return: bounding box in pascal voc format ltrb, which corresponds to standard bottom-left origin
        """
        return self.l, self.t, self.r, self.b

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


def plot_ann_img(ann_img: AnnotatedImage, figsize, with_bb=True, with_head_tail=True, with_index=True, axis_opt="off",
                 **imshow_kwargs):
    plot_img(ann_img.img, figsize=figsize, axis_opt=axis_opt, **imshow_kwargs)
    ax = plt.gca()

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
    if len(annotations) == 0:
        _logger.warning("plot_anns: passed empty annotations list")
        return

    ann_colors = compute_colors_for_annotations(annotations)

    # very rough estimate
    fontsize = max(ax.figure.get_size_inches()[0], 10)
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


def plot_img(img: Union[np.ndarray, Image.Image, torch.Tensor], cmap="gray", axis_opt="off", figsize=None,
             save_path=None, **imshow_kwargs):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    if not figsize:
        figsize = figsize_from_img(img)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis(axis_opt)
    ax.imshow(np.asarray(img), cmap=cmap, **imshow_kwargs)

    if save_path:
        plt.savefig(save_path, cmap=cmap)


def plot_imgs(imgs: Union[np.ndarray, List[np.ndarray], List[Image.Image], torch.Tensor], ncols: int = None,
              img_size=(5, 5), cmap="gray", axis_opt="off", titles: List[str] = None, **imshow_kwargs):
    """
    :param imgs: batch of imgs with shape (batch_size, h, w) or (batch_size, h, w, 3)
    :param ncols: number of columns
    :param img_size: matplotlib size to use for each image
    :param cmap: matplotlib colormap
    :param axis_opt: plot axis or not ("off"/"on")
    :param titles: axis titles
    """
    n_imgs = len(imgs)
    assert 0 < n_imgs < 100

    if ncols is None:
        ncols = min(n_imgs, 5)

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
        ax.imshow(np.asarray(img), cmap=cmap, **imshow_kwargs)
        if titles is not None:
            ax.set_title(titles[i])

    # this is quite slow:
    # fig.tight_layout()
    return axs


def plot_img_paths(img_paths: Union[List[Path], List[str]], ncols=4, img_size=(5, 5)):
    imgs = [read_img(p) for p in img_paths]
    plot_imgs(imgs, ncols=ncols, img_size=img_size, titles=[Path(p).name for p in img_paths])


def read_img(img_path: Union[Path, str]):
    img = Image.open(img_path)
    return exif_transpose(img)


# copied from detectron2
def exif_transpose(image):
    """
    If an image has an EXIF Orientation tag, return a new image that is
    transposed accordingly. Otherwise, return a copy of the image.

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112)
    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)
    if method is not None:
        transposed_image = image.transpose(method)
        del exif[0x0112]
        transposed_image.info["exif"] = exif.tobytes()
        return transposed_image
    return image.copy()


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
