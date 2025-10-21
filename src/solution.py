import numpy as np
import pandas as pd
from numba import njit

#(cache=True, fastmath=True)
@njit(cache=True, fastmath=True)
def _rolling_percentile_rank(values: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the rolling percentile rank of the last observation in each window.

    Parameters
    ----------
    values : np.ndarray
        Input series for a single symbol.
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        Rolling percentile ranks for `values`.
    """
    length = values.shape[0]
    out = np.empty(length, dtype=np.float32)

    for idx in range(length):
        current = values[idx]
        if np.isnan(current):
            out[idx] = np.nan
            continue

        start = idx - window + 1
        if start < 0:
            start = 0

        count = 0
        total = 0
        for j in range(start, idx + 1):
            val = values[j]
            if np.isnan(val):
                continue
            total += 1
            if val <= current:
                count += 1

        if total == 0:
            out[idx] = np.nan
        else:
            out[idx] = count / total

    return out


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be a positive integer")

    df = pd.read_parquet(input_path, columns=["symbol", "Close"])

    close = df["Close"].to_numpy().astype(np.float64, copy=False)
    codes, _ = pd.factorize(df["symbol"], sort=False)

    order = np.argsort(codes, kind="mergesort")
    sorted_codes = codes[order]
    sorted_close = close[order]

    sorted_result = np.empty(sorted_close.shape[0], dtype=np.float32)

    start = 0
    total = sorted_close.shape[0]
    while start < total:
        code = sorted_codes[start]
        end = start + 1
        while end < total and sorted_codes[end] == code:
            end += 1

        segment = np.ascontiguousarray(sorted_close[start:end])
        sorted_result[start:end] = _rolling_percentile_rank(segment, window)
        start = end

    result = np.empty_like(sorted_result)
    result[order] = sorted_result

    return result.reshape(-1, 1)
