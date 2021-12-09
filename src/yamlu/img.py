import copy
import functools
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
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import patheffects

from yamlu.img_ops import crop_img

_logger = logging.getLogger(__name__)

DEFAULT_FIGSIZE = (12, 12)


class BoundingBox:
    """
    This Bounding Box implementation assumes the coordinates in between pixels.
    This means that e.g. the width of the bbox is right-left.
    So a bounding box that contains the top-left pixel would be BoundingBox(0,0,1,1).
    This is in contrast to the convention where BoundingBox(0,0,0,0) corresponds to first pixel, and w = r-l+1
    For some background see: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L23
    """

    def __init__(self, t: float, l: float, b: float, r: float, allow_neg_coord=False):
        self.t, self.l, self.b, self.r = t, l, b, r
        self.allow_neg_coord = allow_neg_coord

        assert t <= b, f"Invalid bounding box coordinates: {self}"
        assert l <= r, f"Invalid bounding box coordinates: {self}"

        if any(x < 0 for x in [t, l, b, r]):
            msg = f"Bounding box has coordinates <= 0: {self}"
            if allow_neg_coord:
                _logger.debug(msg)
            else:
                raise ValueError(msg)

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            return False
        return self.tlbr == other.tlbr

    def __hash__(self):
        return hash(self.tlbr)

    def __repr__(self):
        return f"BoundingBox(t={self.t:.2f},l={self.l:.2f},b={self.b:.2f},r={self.r:.2f})"

    @classmethod
    def clipped_to_image(cls, t: float, l: float, b: float, r: float, img_w: int, img_h: int):
        return BoundingBox(t=max(t, 0), l=max(l, 0), b=min(b, img_h), r=min(r, img_w))

    @classmethod
    def from_center_wh(cls, center, width, height, allow_neg_coord=False):
        x, y = center
        return cls(
            t=y - height / 2,
            l=x - width / 2,
            b=y + height / 2,
            r=x + width / 2,
            allow_neg_coord=allow_neg_coord
        )

    @classmethod
    def from_xywh(cls, x, y, width, height, allow_neg_coord=False):
        return cls(t=y, l=x, b=y + height, r=x + width, allow_neg_coord=allow_neg_coord)

    @classmethod
    def from_ltrb(cls, ltrb, allow_neg_coord=False):
        l, t, r, b = ltrb
        return cls(t, l, b, r, allow_neg_coord)

    @classmethod
    def from_pascal_voc(cls, xmin, ymin, xmax, ymax, allow_neg_coord=False):
        # pascal has pixel-based boxes in range [1, W/H]
        # convert to continuous coordinate boxes in range [0,W/H]
        return cls(t=ymin - 1, l=xmin - 1, b=ymax, r=xmax, allow_neg_coord=allow_neg_coord)

    @classmethod
    def from_points(cls, pts: np.ndarray, allow_neg_coord=False):
        """
        :param pts: xy pts array assuming top-left origin
        :param allow_neg_coord: allow negative coordinates outside of image
        """
        assert pts.ndim == 2
        assert pts.shape[1] == 2
        l, t = pts.min(axis=0)
        r, b = pts.max(axis=0) + 1  # +1 to convert from pixel to coordinate-based representation
        return BoundingBox(t, l, b, r, allow_neg_coord)

    @classmethod
    def from_mask(cls, mask: np.ndarray, assert_any=False):
        """
        :param mask: 2-d binary mask
        :param assert_any: assert that at least one mask entry is True
        """
        assert mask.ndim == 2
        ys, xs = np.where(mask)
        if assert_any:
            assert len(ys) > 0, "mask has no positive entries"
        pts = np.stack([xs, ys], axis=1)
        return BoundingBox.from_points(pts), (ys, xs)

    @property
    def tlbr(self):
        return self.t, self.l, self.b, self.r

    @property
    def w(self):
        return self.r - self.l

    @property
    def h(self):
        return self.b - self.t

    @property
    def area(self):
        return self.w * self.h

    @property
    def diameter(self):
        return math.sqrt(self.w ** 2 + self.h ** 2)

    @property
    def aspect_ratio(self):
        return self.w / self.h

    @property
    def tb_mid(self):
        return self.t + self.h / 2

    @property
    def lr_mid(self):
        return self.l + self.w / 2

    @property
    def center(self):
        return self.lr_mid, self.tb_mid

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

    @property
    def ltrb(self):
        return self.bb_pascal_voc

    def is_within_img(self, img_w, img_h):
        return self.is_within_bb(BoundingBox(0, 0, b=img_h, r=img_w))

    def is_within_bb(self, bb):
        t, l, b, r = self.tlbr
        return t >= bb.t and l >= bb.l and b <= bb.b and r <= bb.r

    def iou(self, bb) -> float:
        intersection_area = self.intersection(bb).area
        union_area = self.area + bb.area - intersection_area
        return intersection_area / union_area

    def distance(self, bb) -> float:
        bbs1 = np.array(self.tlbr).reshape(1, -1)
        bbs2 = np.array(bb.tlbr).reshape(1, -1)
        from yamlu.bb import bbs_distances
        return bbs_distances(bbs1, bbs2).item()

    def intersection(self, bb):
        t = max(self.t, bb.t)
        l = max(self.l, bb.l)
        b = min(self.b, bb.b)
        r = min(self.r, bb.r)
        if b < t:
            b = t
        if r < l:
            r = l
        return BoundingBox(t=t, l=l, b=b, r=r, allow_neg_coord=self.allow_neg_coord)

    def union(self, bb):
        return BoundingBox(t=min(self.t, bb.t), l=min(self.l, bb.l), b=max(self.b, bb.b), r=max(self.r, bb.r),
                           allow_neg_coord=self.allow_neg_coord)

    def pad(self, pad, allow_neg_coord=False) -> "BoundingBox":
        return BoundingBox(self.t - pad, self.l - pad, self.b + pad, self.r + pad, allow_neg_coord=allow_neg_coord)

    def pad_min_size(self, w_min, h_min):
        w_new = max(self.w, w_min)
        h_new = max(self.h, h_min)
        return BoundingBox.from_center_wh(self.center, w_new, h_new)

    def shrink(self, px) -> "BoundingBox":
        return self.pad(-px)

    def shift(self, t_delta=0, l_delta=0) -> "BoundingBox":
        t, l, b, r = self.tlbr
        return BoundingBox(t=t + t_delta, l=l + l_delta, b=b + t_delta, r=r + l_delta,
                           allow_neg_coord=self.allow_neg_coord)

    def clip_to_image(self, img_w: int, img_h: int) -> "BoundingBox":
        t, l, b, r = self.tlbr
        return BoundingBox(t=max(t, 0), l=max(l, 0), b=min(b, img_h), r=min(r, img_w))

    def scale(self, factor) -> "BoundingBox":
        tlbr = np.array(self.tlbr) * factor
        return BoundingBox(*tlbr, allow_neg_coord=self.allow_neg_coord)

    def rotate(self, angle, img_size) -> "BoundingBox":
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
            l=img_h - self.b,
            b=self.r,
            r=img_h - self.t
        )


class Annotation:
    def __init__(self, category: str, bb: BoundingBox, **kwargs: Any):
        self._fields: Dict[str, Any] = {
            "category": category,
            "bb": bb,
            **kwargs
        }

    @property
    def category(self) -> str:
        return self._fields["category"]

    @property
    def bb(self) -> BoundingBox:
        return self._fields["bb"]

    @property
    def extra_fields(self):
        return {k: v for k, v in self._fields.items() if k not in ["category", "bb"]}

    def plot(self, img: Image.Image, figsize=None, pad_perc=10, **plot_ann_kwargs):
        plot_img(img, figsize=figsize)
        if "alpha_bb" not in plot_ann_kwargs:
            plot_ann_kwargs["alpha_bb"] = 0.0
        plot_anns([self], **plot_ann_kwargs)

        bb = self.bb
        pad_x, pad_y = round(bb.w * pad_perc / 100.), round(bb.h * pad_perc / 100.)
        plt.xlim(bb.l - pad_x, bb.r + pad_x)
        # NOTE: ylim requires (b,t) instead of (t,b), otherwise it flips the image
        plt.ylim(bb.b + pad_y, bb.t - pad_y)

    def img_cropped(self, img: Union[Image.Image, np.ndarray], pad=0):
        return crop_img(img, self.bb.ltrb, pad)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self._fields[name] = val

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        return self._fields[name]

    def __delattr__(self, name: str):
        if name.startswith("_"):
            return super().__delattr__(name)
        del self._fields[name]

    def set(self, name: str, value: Any):
        setattr(self, name, value)

    def get(self, name: str):
        return self._fields[name]

    def __contains__(self, key):
        return key in self._fields

    def __repr__(self):
        # only get fields that are not of type Annotation to prevent potentially infinite recursion
        fields_str = ", ".join(f"{k}={v}" for k, v in self.extra_fields.items() if not isinstance(v, Annotation))
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
        img = read_img(img_path)
        return cls(img_path.name, width=img.width, height=img.height, annotations=annotations, img=img)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return self.annotations[idx]

    def copy(self) -> "AnnotatedImage":
        """
        Creates a copy of the AnnotatedImage, but without copying the image itself
        """
        return AnnotatedImage(
            filename=self.filename,
            width=self.width,
            height=self.height,
            annotations=copy.deepcopy(self.annotations),
            img=self.img
        )

    def plot(self, figsize=DEFAULT_FIGSIZE, **plot_anns_kwargs):
        assert self.img is not None, f"{self}: missing img attribute!"
        plot_img(self.img, figsize=figsize)
        plot_anns(self.annotations, **plot_anns_kwargs)

    def save(self, directory: Path, **params):
        directory.mkdir(exist_ok=True, parents=True)
        self.img.save(directory / self.filename, **params)

    def save_with_anns(self, directory: Path, figsize=None, suffix="_bb", jpg_quality=75, **plot_anns_kwargs):
        self.plot(figsize=figsize, **plot_anns_kwargs)
        directory.mkdir(exist_ok=True, parents=True)
        img_path = directory / f"{self.img_id}{suffix}.jpg"
        plt.savefig(str(img_path), pil_kwargs={"quality": jpg_quality})
        plt.close()
        return img_path

    def filter(self, category: str, *other_categories) -> List[Annotation]:
        assert isinstance(category, str), f"Wrong type for category {category}: {type(category)}"
        return [a for a in self.annotations if a.category in {category, *other_categories}]

    def filter_substr(self, substr: str):
        return [a for a in self.annotations if substr in a.category]

    def union_bb(self):
        bbs = [a.bb for a in self.annotations]
        return functools.reduce(lambda bb1, bb2: bb1.union(bb2), bbs)

    @property
    def boxes_tlbr(self):
        boxes = np.array([a.bb.tlbr for a in self.annotations])
        # reshape so that shape is (0,4) for images with no annotations
        return boxes.reshape(len(self.annotations), 4)

    @property
    def boxes_ltrb(self):
        return self.boxes_tlbr[:, [1, 0, 3, 2]]

    @property
    def size(self):
        return self.width, self.height

    @property
    def img_id(self) -> str:
        return Path(self.filename).stem

    @property
    def categories(self):
        return [a.category for a in self.annotations]

    def __repr__(self):
        n_anns = len(self.annotations)
        cat_cnt = Counter(a.category for a in self.annotations)
        s = f"AnnotatedImage(filename='{self.filename}', size={self.size}, {n_anns} annotations ({cat_cnt}))"
        return s


def compute_colors(annotations: List[Annotation], categories: List[str], cmap='jet') -> List[np.ndarray]:
    """
    :param annotations: list of annotations
    :param categories: dataset categories
    :param cmap: matplotlib cmap (other options include Dark2, Accent)
    :return: a list with an rgba array for each annotation
    """
    dataset_cat_to_id = {c: i for i, c in enumerate(categories)}

    cat_ids = np.array([dataset_cat_to_id[ann.category] for ann in annotations])

    cm = plt.get_cmap(cmap)
    cNorm = colors.Normalize(vmin=0, vmax=len(categories) - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    scalarMap.get_clim()

    return [scalarMap.to_rgba(c) for c in cat_ids]


def plot_anns(annotations: List[Annotation], categories: List[str] = None, ann_colors=None, ax=None, with_index=False,
              with_score=True, digits: int = 1, min_score: float = 0.0, draw_connections=True, draw_keypoints=True,
              alpha_bb=.15, alpha_txt=.5, alpha_kp=.3,
              font_size_scale=.7, lw_scale=.1, black_lw_font_scale=0.05, text_horizontal_pos="left",
              text_vertical_pos="top", text_field="category"):
    """
    :param annotations: annotations to plot
    :param categories: dataset categories for computing deterministic annotation colors
    :param ann_colors: color for each annotation to use (categories is ignored if ann_colors is set) or single color
    :param ax: matplotlib axes
    :param with_index: include the annotation index into the bounding box label
    :param with_score: include the score into the bounding box label
    :param digits: score digits
    :param min_score: score threshold for plotting the annotation
    :param draw_connections: draw arrow connections
    :param alpha_bb: alpha for the bounding box fill (not the border)
    :param alpha_txt: alpha for the text
    :param alpha_kp: alpha for scattered keypoints
    :param font_size_scale: font size scale relative to larger size of the figure
    :param lw_scale: bounding box linewidth relative to larger size of the figure
    :param black_lw_font_scale: scale of the black stroke surrounding the font relative to font size
    :param text_horizontal_pos: horizontal position of bounding box text (one of 'left', 'center', 'right')
    :param text_vertical_pos: vetical position of bounding box text (one of 'top', 'bottom')
    :param text_field: field to use for the bounding box title
    """
    if len(annotations) == 0:
        _logger.warning("plot_anns: passed empty annotations list")
        return

    if ax is None:
        ax = plt.gca()
    if categories is None:
        categories = list(set(a.category for a in annotations))
    if ann_colors is None:
        ann_colors = compute_colors(annotations, categories)
    elif not isinstance(ann_colors, list):
        # assume only a single color has been passed
        ann_colors = [ann_colors] * len(annotations)

    annotations = [a for a in annotations if "score" not in a or a.score >= min_score]

    # rough estimates so that text + kp sizes scale with figure size
    figsize = ax.figure.get_size_inches()
    larger_size = max(figsize)
    fontsize = max(larger_size * font_size_scale, 8)
    conn_size = larger_size * .3
    lw = larger_size * lw_scale
    kp_size = (fontsize * 0.66) ** 2  # in plt.scatter s is area (w*h)
    black_lw_font = fontsize * black_lw_font_scale

    for i, ann, color in zip(range(len(annotations)), annotations, ann_colors):
        patch = mpatches.Rectangle(*ann.bb.xy_w_h, fill=True, facecolor=color, edgecolor=color, lw=0, alpha=alpha_bb)
        ax.add_patch(patch)
        patch = mpatches.Rectangle(*ann.bb.xy_w_h, fill=False, facecolor="none", edgecolor=color, lw=lw, alpha=.8)
        ax.add_patch(patch)

        text = "" if text_field is None or text_field not in ann else getattr(ann, text_field)
        if with_score and "score" in ann:
            text += f" {round(ann.score * 100, digits)}%".replace(".0", "")
        if with_index:
            text += f" {i}"

        txt_kwargs = _txt_coord_alignment(ann.bb, text_horizontal_pos, text_vertical_pos)
        txt = ax.text(**txt_kwargs, s=text, color=color, fontsize=fontsize, alpha=alpha_txt)
        if black_lw_font > 0:
            txt.set_path_effects(
                [patheffects.Stroke(linewidth=black_lw_font, foreground='BLACK'), patheffects.Normal()])

        if draw_keypoints:
            if "keypoints" in ann or "waypoints" in ann:
                kps = ann.keypoints if "keypoints" in ann else ann.waypoints
                xs, ys = kps.T[:2]
                # ax.plot(xs, ys, ".-", markersize=fontsize * 0.66, alpha=alpha_kp, linewidth=conn_size, color="red")
                ax.scatter(xs, ys, s=kp_size, alpha=alpha_kp, color="red")
            if "head" in ann:
                ax.scatter(*ann.head, marker=">", s=kp_size, alpha=alpha_kp, color="blue", edgecolor="black", lw=1)
            if "tail" in ann:
                ax.scatter(*ann.tail, marker="o", s=kp_size, alpha=alpha_kp, color="blue", edgecolor="black", lw=1)
        if draw_connections:
            draw_arrow_connections(ann, ax, lw_conn=conn_size, head_length=conn_size * 2, head_width=conn_size)


def _txt_coord_alignment(bb: BoundingBox, text_horizontal_pos: str, text_vertical_pos: str):
    assert text_horizontal_pos in ["left", "center", "right"]
    assert text_vertical_pos in ["top", "center", "bottom"]
    if text_horizontal_pos == "left":
        horiz_coord = bb.l
    elif text_horizontal_pos == "center":
        horiz_coord = bb.lr_mid
    else:
        horiz_coord = bb.r

    # text vertical alignment is opposite of positioning, i.e. text is aligned to bottom when it's placed on top of bb
    if text_vertical_pos == "top":
        va = "bottom"
        vert_coord = bb.t
    elif text_vertical_pos == "bottom":
        va = "top"
        vert_coord = bb.b
    elif text_vertical_pos == "center":
        va = "center"
        vert_coord = bb.tb_mid

    # text horizontal alignment equals the position
    ha = text_horizontal_pos

    return {"x": horiz_coord, "y": vert_coord, "va": va, "ha": ha}


def draw_arrow_connections(ann: Annotation, ax, lw_conn=4, color=(220 / 255., 20 / 255., 60 / 255.),
                           ls="-", head_length=10, head_width=4, alpha=.3):
    vertices = []

    if "arrow_prev" in ann:
        vertices.append(ann.arrow_prev.bb.center)

    if "waypoints" in ann:
        vertices += ann.waypoints.tolist()
    elif "keypoints" in ann:
        vertices += ann.keypoints[:, :2].tolist()
    else:
        kps = [kp for kp in ["tail", "head"] if kp in ann]
        for kp in kps:
            vertices.append(ann.get(kp))

    if "arrow_next" in ann:
        vertices.append(ann.arrow_next.bb.center)

    if len(vertices) <= 1:
        return

    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(vertices) - 1)

    path = mpath.Path(vertices, codes)

    if ann.extra_fields.get("directed", True):
        arw_style = mpatches.ArrowStyle.CurveFilledB(head_width=head_width, head_length=head_length)
    else:
        arw_style = mpatches.ArrowStyle.Curve()

    # noinspection PyTypeChecker
    p = mpatches.FancyArrowPatch(
        path=path,
        arrowstyle=arw_style,
        linewidth=lw_conn,
        linestyle=ls,
        color=color,
        alpha=alpha,
        # path_effects=[patheffects.Stroke(linewidth=2, foreground='BLACK', alpha=alpha), patheffects.Normal()]
    )
    # TODO capstyle and joinstyle https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/joinstyle.html
    ax.add_patch(p)


def plot_img(img: Union[np.ndarray, Image.Image], figsize=DEFAULT_FIGSIZE, save_path=None, **imshow_kwargs):
    if hasattr(img, "cpu"):  # check if pytorch tensor without importing torch
        img = img.cpu().numpy()

    if not figsize:
        figsize = figsize_from_img(img)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    # hard-coded cmap gray: seems to have no effect for RGB images but is required for grayscale images
    ax.imshow(np.asarray(img), cmap="gray", **imshow_kwargs)

    if save_path:
        plt.savefig(save_path)


def plot_imgs(imgs: Union[np.ndarray, List[np.ndarray], List[Image.Image]], ncols: int = None,
              img_size=(5, 5), axis_opt="off", titles: List[str] = None, max_allowed_imgs=100,
              **imshow_kwargs):
    """
    :param imgs: batch of imgs with shape (batch_size, h, w) or (batch_size, h, w, 3)
    :param ncols: number of columns
    :param img_size: matplotlib size to use for each image
    :param axis_opt: plot axis or not ("off"/"on")
    :param titles: axis titles
    :param max_allowed_imgs: throws an error if more than max_allowed_imgs are passed
    """
    n_imgs = len(imgs)
    assert 0 < n_imgs < max_allowed_imgs, f"n_imgs: {n_imgs}"

    if ncols is None:
        ncols = min(n_imgs, 5)

    if titles is not None:
        assert len(imgs) == len(titles), f"{len(imgs)} != {len(titles)}"

    if hasattr(imgs, "detach"):  # check if pytorch tensor without importing torch
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
        ax.imshow(np.asarray(img), **imshow_kwargs)
        if titles is not None:
            ax.set_title(titles[i])

    # this is quite slow:
    # fig.tight_layout()
    return axs


def plot_img_paths(img_paths: Union[List[Path], List[str]], ncols=4, img_size=(5, 5)):
    imgs = [read_img(p) for p in img_paths]
    plot_imgs(imgs, ncols=ncols, img_size=img_size, titles=[Path(p).name for p in img_paths])


def read_img(img_path: Union[Path, str]) -> Image.Image:
    img = Image.open(img_path)
    return exif_transpose(img)


# same as ImageOps.exif_transpose, but does not copy the image, which avoids Image.load()
def exif_transpose(image) -> Image.Image:
    """
    If an image has an EXIF Orientation tag, return a new image that is
    transposed accordingly. Otherwise, return the image.

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
    return image


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
