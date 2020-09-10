def bin_stats(arr) -> str:
    """Return a string basic statistics for a binary array of either {0,1} or {True,False}"""
    n = len(arr)
    k = sum(arr)
    p = 100. * k / n
    return f"{k}/{n} ({p:.2f}%)"
