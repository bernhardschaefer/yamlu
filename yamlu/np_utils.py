import numpy as np


def bin_stats(arr) -> str:
    """Return a string basic statistics for a binary array of either {0,1} or {True,False}"""
    n = len(arr)
    k = sum(arr)
    p = 100. * k / n if n != 0 else 0.
    return f"{k}/{n} ({p:.2f}%)"


def to_python_type(v, ndigits: int):
    if v is None or isinstance(v, str) or isinstance(v, int):
        return v
    if isinstance(v, float):
        v = round(v, ndigits)
        return int(v) if v.is_integer() else v

    if isinstance(v, list):
        return [to_python_type(x, ndigits) for x in v]
    if isinstance(v, np.ndarray):
        return v.round(ndigits).tolist()
    if isinstance(v, np.number):
        return to_python_type(v.item(), ndigits)

    raise ValueError(f"Unknown type for {v}: {type(v)}")
