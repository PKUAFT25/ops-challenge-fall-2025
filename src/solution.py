import numpy as np
import pandas as pd
from numba import njit


@njit
def _rolling_rank_numba(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling percentile rank for a 1D array."""
    n = values.shape[0]
    result = np.empty(n, dtype=np.float32)

    for i in range(n):
        current = values[i]
        start = i - window + 1
        if start < 0:
            start = 0

        span = i - start + 1

        if np.isnan(current):
            result[i] = 0.0
            continue

        count = 0
        for j in range(start, i + 1):
            val = values[j]
            if not np.isnan(val) and val <= current:
                count += 1

        result[i] = count / span

    return result


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be a positive integer")

    df = pd.read_parquet(input_path)
    if df.empty:
        return np.empty((0, 1), dtype=np.float32)

    window = int(window)

    close_values = df["Close"].to_numpy(dtype=np.float64, copy=False)
    output = np.empty(close_values.shape[0], dtype=np.float32)

    grouped_indices = df.groupby("symbol", sort=False).indices
    for idx in grouped_indices.values():
        group_values = np.ascontiguousarray(close_values[idx])
        output[idx] = _rolling_rank_numba(group_values, window)

    return output.reshape(-1, 1)


