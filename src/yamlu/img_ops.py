import math
from typing import Union, Tuple

import numpy as np
from PIL import Image
from matplotlib import colors


def crop_img(img: Image.Image, ltrb, pad: int = 0):
    # noinspection PyTypeChecker
    img_cropped = crop_img_np(np.asarray(img), ltrb, pad)
    # TODO use Image.crop() to prevent converting two times
    return Image.fromarray(img_cropped)


def crop_img_np(img: np.ndarray, ltrb, pad: int = 0) -> np.ndarray:
    """
    Crops the image using the specified ltrb float bounding box while ensuring the bounding box is clipped to the image
    Args:
        img: image
        ltrb: bounding box in ltrb format
        pad: pixels to pad the bounding box

    Returns:

    """
    l, t, r, b = ltrb

    # floor to int and clip to image
    t, l = [max(math.floor(x - pad), 0) for x in [t, l]]

    # ceil to int and clip to image
    img_h, img_w = img.shape[:2]
    b, r = [min(math.ceil(x + pad), m - 1) for x, m in zip([b, r], [img_h, img_w])]

    return img[t:b, l:r, ...]


def grayscale_transparency(img: Image.Image) -> Image.Image:
    """
    Sets the alpha transparency of an image according to its grayscale value,
    i.e. white is fully transparent whereas black is not transparent.
    """
    # noinspection PyTypeChecker
    img_np = np.asarray(img).copy()
    # noinspection PyTypeChecker
    grayscale = np.asarray(img.convert("L"))
    img_np[:, :, -1] = 255 - grayscale
    return Image.fromarray(img_np)


def white_to_transparency(img: Image.Image, thresh=255) -> Image.Image:
    """
    Makes the white pixels in an image transparent
    :param img: source image
    :param thresh: pixels where all RGB values are higher or equal than this threshold are considered white
    """
    # FIXME change implementation so that it selects white pixels and only modifies their alpha
    # noinspection PyTypeChecker
    x = np.asarray(img.convert('RGBA')).copy()
    # inspired by https://stackoverflow.com/a/54148416
    non_white_mask = (x[:, :, :3] < thresh).any(axis=2)
    x[:, :, 3] = (255 * non_white_mask).astype(np.uint8)
    return Image.fromarray(x)


def black_to_color(img: Image.Image, color: Union[str, Tuple[int, int, int]], thresh=128) -> Image.Image:
    """
    Converts black pixels in an image to another color
    :param img: source image
    :param color: target color to apply to black pixels
    :param thresh: a pixel value threshold, each pixel with intensity less than thresh will be converted
    """
    rgb = (np.array(colors.to_rgb(color)) * 255.).astype(np.uint8)

    # noinspection PyTypeChecker
    x = np.asarray(img).copy()
    black_mask = (x[:, :, :3] <= thresh).all(axis=2)
    x[black_mask, :3] = rgb

    return Image.fromarray(x)
