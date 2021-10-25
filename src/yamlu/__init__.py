import sys

from yamlu.img import read_img, plot_img, plot_imgs, plot_img_paths, plot_anns
from yamlu.misc import flatten
from yamlu.np_utils import bin_stats
from yamlu.path import ls, glob

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

all = [
    "read_img", "plot_img", "plot_imgs", "plot_img_paths", "plot_anns",
    "flatten",
    "bin_stats",
    "ls", "glob",
]

try:
    from yamlu.pytorch import isin

    all.append("isin")
except ImportError:
    pass

__all__ = all
