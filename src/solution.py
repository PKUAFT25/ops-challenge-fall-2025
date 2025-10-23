from bisect import bisect_left, bisect_right, insort
from collections import deque
from typing import List

import numpy as np
import pandas as pd


def _rolling_percentile_rank(values: np.ndarray, window: int) -> np.ndarray:
    """Return percentile rank for each point against the trailing window."""
    n = values.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float32)

    result = np.empty(n, dtype=np.float32)
    sorted_window: List[float] = []
    window_queue: deque[float] = deque()

    for idx in range(n):
        val = float(values[idx])
        window_queue.append(val)
        insort(sorted_window, val)

        if len(window_queue) > window:
            to_remove = window_queue.popleft()
            remove_idx = bisect_left(sorted_window, to_remove)
            sorted_window.pop(remove_idx)

        rank = bisect_right(sorted_window, val)
        result[idx] = rank / len(sorted_window)

    return result


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be a positive integer")

    df = pd.read_parquet(input_path)
    output = np.empty(len(df), dtype=np.float32)

    for _, group in df.groupby("symbol", sort=False):
        ranks = _rolling_percentile_rank(group["Close"].to_numpy(copy=False), window)
        output[group.index.to_numpy()] = ranks

    return output[:, None]


