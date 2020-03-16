import logging
import os
from contextlib import contextmanager
from typing import List


def set_gpu_ids(gpu_ids: List[int]):
    gpu_ids_str = ",".join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str


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
