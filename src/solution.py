import numpy as np
import pandas as pd
from numba import njit, prange


@njit(parallel=True)
def _rolling_rank_all(
    values: np.ndarray,
    group_offsets: np.ndarray,
    group_lengths: np.ndarray,
    window: int,
) -> np.ndarray:
    total = values.shape[0]
    ranks = np.empty(total, dtype=np.float32)

    for group_idx in prange(group_offsets.shape[0]):
        start = group_offsets[group_idx]
        length = group_lengths[group_idx]
        end = start + length

        for idx in range(start, end):
            window_start = idx - window + 1
            if window_start < start:
                window_start = start

            current = values[idx]

            if np.isnan(current):
                ranks[idx] = 0.0
                continue

            count = 0
            denom = idx - window_start + 1

            for j in range(window_start, idx + 1):
                candidate = values[j]
                if not np.isnan(candidate) and candidate <= current:
                    count += 1

            ranks[idx] = count / denom

    return ranks


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be a positive integer")

    df = pd.read_parquet(input_path, columns=["symbol", "Close"])
    if df.empty:
        return np.empty((0, 1), dtype=np.float32)

    # Factorize symbols to group contiguous rows for JIT processing.
    symbols = df["symbol"].fillna("__NA_SYMBOL__")
    codes, uniques = pd.factorize(symbols, sort=False)

    if codes.size == 0:
        return np.empty((0, 1), dtype=np.float32)

    order = np.argsort(codes, kind="mergesort")
    sorted_values = df["Close"].to_numpy(dtype=np.float64, copy=False)[order]
    sorted_codes = codes[order]

    group_lengths = np.bincount(sorted_codes, minlength=len(uniques)).astype(np.int64)
    group_offsets = np.empty_like(group_lengths)
    group_offsets[0] = 0
    if group_offsets.shape[0] > 1:
        np.cumsum(group_lengths[:-1], out=group_offsets[1:])

    ranked_sorted = _rolling_rank_all(sorted_values, group_offsets, group_lengths, window)

    ranks = np.empty_like(ranked_sorted)
    ranks[order] = ranked_sorted

    return ranks.reshape(-1, 1)
