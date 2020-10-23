import functools
import itertools
import logging
import os
import pickle
import random
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch

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


# copied from detectron2
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/env.py
def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big")
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)


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
