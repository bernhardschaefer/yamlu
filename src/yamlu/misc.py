import functools
import itertools
import logging
import os
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List

_logger = logging.getLogger(__name__)


def flatten(collection):
    return list(itertools.chain(*collection))


# copied from maskrcnn-benchmark
class HidePrints:
    """Context Manager that mutes sys.stdout so that print() statements have no effect"""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def set_gpu_ids(gpu_ids: List[int]):
    gpu_ids_str = ",".join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str


def pickle_cache(cache_path: Path):
    """
    Decorator to cache the result of a function in a pickle file.
    :param cache_path: the path where to cache the file
    """

    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if cache_path.is_file():
                with cache_path.open("rb") as f:
                    return pickle.load(f)
            res = func(*args, **kwargs)
            _logger.info("Caching object type %s to %s", type(res), cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(res, f)
            return res

        return inner

    return decorator


@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


@contextmanager
def change_log_level(logger: logging.Logger, level):
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)
