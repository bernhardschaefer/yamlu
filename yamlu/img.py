import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import patheffects

from yamlu.bb import iou_vector

_logger = logging.getLogger(__name__)


class BoundingBox:
    """
    This Bounding Box implementation assumes the coordinates to be inclusive.
    This means that e.g. the width of the bbox is right-left+1.
    For some background see: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L23
    """

    def __init__(self, t: float, l: float, b: float, r: float):
        self.t, self.l, self.b, self.r = t, l, b, r

        assert t >= 0, f"Invalid bounding box coordinates: {self}"
        assert l >= 0, f"Invalid bounding box coordinates: {self}"
        assert b >= 0, f"Invalid bounding box coordinates: {self}"
        assert r >= 0, f"Invalid bounding box coordinates: {self}"
        assert t <= b, f"Invalid bounding box coordinates: {self}"
        assert l <= r, f"Invalid bounding box coordinates: {self}"

    def __repr__(self):
        return f"BoundingBox(t={self.t:.2f},l={self.l:.2f},b={self.b:.2f},r={self.r:.2f})"

    @classmethod
    def clipped_to_image(cls, t: float, l: float, b: float, r: float, img_w: int, img_h: int):
        t = max(t, 0)
        l = max(l, 0)
        b = min(b, img_h)
        r = min(r, img_w)
        return BoundingBox(t, l, b, r)

    @classmethod
    def from_center_wh(cls, center, w, h, clip_tl=False):
        x, y = center
        t = y - h / 2
        l = x - w / 2
        if clip_tl:
            t = max(t, 0)
            l = max(l, 0)
        return cls(t=t, l=l, b=y + h / 2, r=x + w / 2)

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
        return iou_vector(bbs1, bbs2).item()

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


class Annotation:
    def __init__(self, category: str, bb: BoundingBox, **kwargs: Any):
        self._fields: Dict[str, Any] = {
            "category": category,
            "bb": bb,
            **kwargs
        }

    @property
    def category(self):
        return self._fields["category"]

    @property
    def bb(self):
        return self._fields["bb"]

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self._fields[name] = val

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        return self._fields[name]

    def __contains__(self, key):
        return key in self._fields

    def __repr__(self):
        fields = self._fields
        fields_str = ", ".join(f"{k}={v}" for k, v in fields.items() if k not in ["category", "bb", "next", "prev"])
        return f"{self.__class__.__name__}(category='{self.category}', bb={self.bb}, {fields_str})"


@dataclass
class AnnotatedImage:
    filename: str
    width: int
    height: int
    annotations: List[Annotation]
    img: Optional[Image.Image] = field(default=None, repr=False)

    @classmethod
    def from_img_path(cls, img_path: Path, annotations: List[Annotation]):
        img = Image.open(img_path)
        return cls(img_path.name, width=img.width, height=img.height, annotations=annotations, img=img)

    def plot(self, figsize=None, with_bb=True, with_index=True, axis_opt="off", **imshow_kwargs):
        assert self.img is not None, f"{self}: missing img attribute!"
        plot_img(self.img, figsize=figsize, axis_opt=axis_opt, **imshow_kwargs)
        ax = plt.gca()

        if with_bb:
            plot_anns(ax, self.annotations, with_index=with_index)

    def save(self, imgs_path: Path):
        self.img.save(imgs_path / self.filename)

    def save_with_anns(self, directory: Path):
        self.plot()
        img_name = Path(self.filename).stem + "_annotated.png"
        img_path = directory / img_name
        plt.savefig(str(img_path))
        plt.close()
        return img_path

    @property
    def boxes_tlbr(self):
        boxes = np.array([a.bb.tlbr for a in self.annotations])
        # reshape so that shape is (0,4) for images with no annotations
        return boxes.reshape(len(self.annotations), 4)

    @property
    def size(self):
        return self.width, self.height

    def __repr__(self):
        n_anns = len(self.annotations)
        cat_cnt = Counter(a.category for a in self.annotations)
        s = f"AnnotatedImage(filename='{self.filename}', size={self.size}, {n_anns} annotations ({cat_cnt}))"
        return s


def compute_colors_for_annotations(annotations: List[Annotation], cmap='jet'):
    categories = set(a.category for a in annotations)
    cat_to_id = dict((c, i) for i, c in enumerate(categories))
    return compute_colors(annotations, cat_to_id, cmap)


def compute_colors(annotations: List[Annotation], cat_to_id: Dict[str, int], cmap):
    cat_ids = np.array([cat_to_id[ann.category] for ann in annotations])

    cm = plt.get_cmap(cmap)
    cNorm = colors.Normalize(vmin=0, vmax=len(cat_to_id) - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    scalarMap.get_clim()

    return [scalarMap.to_rgba(c) for c in cat_ids]


def plot_anns(ax, annotations: List[Annotation], ann_colors=None, with_index=False, digits=2, min_score: float = 0.0):
    if len(annotations) == 0:
        _logger.warning("plot_anns: passed empty annotations list")
        return

    if ann_colors is None:
        ann_colors = compute_colors_for_annotations(annotations)
    else:
        assert len(annotations) == len(ann_colors), f"{len(annotations)} != {len(ann_colors)}"

    annotations = [a for a in annotations if "score" not in a or a.score >= min_score]

    # very rough estimate
    fontsize = max(ax.figure.get_size_inches()[0], 10)
    lw = fontsize / 10

    for i, ann, color in zip(range(len(annotations)), annotations, ann_colors):
        patch = mpatches.Rectangle(*ann.bb.xy_w_h, fill=True, facecolor=color, edgecolor=color, lw=0, alpha=.05)
        ax.add_patch(patch)
        patch = mpatches.Rectangle(*ann.bb.xy_w_h, fill=False, facecolor="none", edgecolor=color, lw=lw, alpha=.8)
        ax.add_patch(patch)

        text = ann.category
        if "score" in ann:
            text += f" {round(ann.score * 100, digits)}%"
        if with_index:
            text += f" {i}"
        txt = ax.text(ann.bb.l, ann.bb.t, text, verticalalignment='bottom', color=color, fontsize=fontsize,
                      alpha=.5)
        txt.set_path_effects([patheffects.Stroke(linewidth=1, foreground='BLACK'), patheffects.Normal()])

        if "head" in ann:
            ax.scatter(*ann.head, marker=">", s=50, alpha=.5, color="green", edgecolor="black", linewidth=1)
        if "tail" in ann:
            ax.scatter(*ann.tail, marker="o", s=50, alpha=.5, color="green", edgecolor="black", linewidth=1)


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
