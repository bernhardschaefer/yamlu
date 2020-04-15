import datetime as dt
import inspect
import logging
import sys
from functools import wraps, partial


# copied from scikit-lego (to avoid the dependencies of scikit-lego)
# https://github.com/koaning/scikit-lego/blob/master/sklego/pandas_utils.py#L14
def log_step(func=None, *, level=logging.INFO):
    """
    Decorates a function that transforms a pandas dataframe to add automated logging statements
    :Example:
    >>> @log_step
    ... def remove_outliers(df, min_obs=5):
    ...     pass
    >>> @log_step(level=logging.INFO)
    ... def remove_outliers(df, min_obs=5):
    ...     pass
    """
    if func is None:
        return partial(log_step, level=level)

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(sys.modules[func.__module__].__name__)

        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = "".join(
            ", {} = {!r}".format(*item) for item in list(func_args.items())[1:]
        )
        logger.log(
            level,
            f"[{func.__name__}(df{func_args_str})] "
            f"n_obs={result.shape[0]} n_col={result.shape[1]} time={time_taken}",
        )

        return result

    return wrapper
