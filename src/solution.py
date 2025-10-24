import os
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True, nogil=True)
def _fenwick_update(tree: np.ndarray, index: int, delta: int) -> None:
    """In-place Fenwick tree point update."""
    length = tree.shape[0]
    while index < length:
        tree[index] += delta
        index += index & -index


@njit(cache=True, nogil=True)
def _fenwick_prefix_sum(tree: np.ndarray, index: int) -> int:
    """Prefix sum query on a Fenwick tree."""
    total = 0
    while index > 0:
        total += tree[index]
        index -= index & -index
    return total


@njit(cache=True, nogil=True)
def _rolling_percentile_rank_codes(codes: np.ndarray, window: int, unique_count: int) -> np.ndarray:
    """Compute rolling percentile ranks from pre-encoded integer codes."""
    n = codes.shape[0]
    result = np.empty(n, dtype=np.float32)

    if n == 0:
        return result

    tree = np.zeros(unique_count + 1, dtype=np.int64)
    size = 0

    for i in range(n):
        code = codes[i] + 1  # Fenwick trees are 1-indexed
        _fenwick_update(tree, code, 1)
        size += 1

        if size > window:
            old_code = codes[i - window] + 1
            _fenwick_update(tree, old_code, -1)
            size -= 1

        count = _fenwick_prefix_sum(tree, code)
        result[i] = count / size

    return result


def _process_group(view: np.ndarray, indices: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    group_values = np.ascontiguousarray(view[indices], dtype=np.float64)
    if group_values.size == 0:
        return indices, np.empty(0, dtype=np.float32)

    unique_vals, inverse = np.unique(group_values, return_inverse=True)
    codes = inverse.astype(np.int32, copy=False)
    ranks = _rolling_percentile_rank_codes(codes, window, int(unique_vals.shape[0]))
    return indices, ranks


def _resolve_worker_count(num_groups: int) -> int:
    if num_groups <= 1:
        return 1

    env_value = os.environ.get("OPS_MAX_THREADS")
    if env_value:
        try:
            configured = int(env_value)
        except ValueError:
            configured = 0
        if configured > 0:
            return max(1, min(num_groups, configured))

    cpu_count = os.cpu_count() or 1
    return max(1, min(num_groups, cpu_count))


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be a positive integer")
    window = int(window)

    df = pd.read_parquet(input_path)
    close_values = df["Close"].to_numpy(dtype=np.float64, copy=False)

    result = np.empty(close_values.shape[0], dtype=np.float32)
    group_indices = df.groupby("symbol", sort=False).indices

    group_arrays = [np.asarray(positions, dtype=np.int64) for positions in group_indices.values()]
    max_workers = _resolve_worker_count(len(group_arrays))

    if max_workers == 1:
        for idx in group_arrays:
            _, ranks = _process_group(close_values, idx, window)
            result[idx] = ranks
        return result[:, None]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, ranks in executor.map(_process_group, repeat(close_values), group_arrays, repeat(window)):
            result[idx] = ranks

    return result[:, None]  # must be [N, 1]


