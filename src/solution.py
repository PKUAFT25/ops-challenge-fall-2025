import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper


try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


Segment = Tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]


@njit
def _bit_update(tree: np.ndarray, idx: int, delta: int) -> None:
    """Classic Fenwick tree point update (1-indexed)."""
    size = tree.size
    while idx < size:
        tree[idx] += delta
        idx += idx & -idx


@njit
def _bit_prefix_sum(tree: np.ndarray, idx: int) -> int:
    """Prefix sum query on a Fenwick tree (1-indexed)."""
    total = 0
    while idx > 0:
        total += tree[idx]
        idx -= idx & -idx
    return total


@njit
def _rolling_rank_indices_int32(
    indices: np.ndarray,
    valid_mask: np.ndarray,
    window: int,
    bit_size: int,
) -> np.ndarray:
    """Compute rolling percentile ranks using a Fenwick tree backed by int32."""
    n = indices.size
    result = np.empty(n, dtype=np.float32)
    bit = np.zeros(bit_size, dtype=np.int32)

    for i in range(n):
        if valid_mask[i]:
            idx_val = int(indices[i])
            _bit_update(bit, idx_val, 1)

        if i >= window:
            j = i - window
            if valid_mask[j]:
                idx_val = int(indices[j])
                _bit_update(bit, idx_val, -1)
            window_len = window
        else:
            window_len = i + 1

        if valid_mask[i]:
            idx_val = int(indices[i])
            count_le = _bit_prefix_sum(bit, idx_val)
            result[i] = count_le / window_len
        else:
            result[i] = 0.0

    return result


@njit
def _rolling_rank_indices_int16(
    indices: np.ndarray,
    valid_mask: np.ndarray,
    window: int,
    bit_size: int,
) -> np.ndarray:
    """Compute rolling percentile ranks using a Fenwick tree backed by int16."""
    n = indices.size
    result = np.empty(n, dtype=np.float32)
    bit = np.zeros(bit_size, dtype=np.int16)

    for i in range(n):
        if valid_mask[i]:
            idx_val = int(indices[i])
            _bit_update(bit, idx_val, 1)

        if i >= window:
            j = i - window
            if valid_mask[j]:
                idx_val = int(indices[j])
                _bit_update(bit, idx_val, -1)
            window_len = window
        else:
            window_len = i + 1

        if valid_mask[i]:
            idx_val = int(indices[i])
            count_le = _bit_prefix_sum(bit, idx_val)
            result[i] = count_le / window_len
        else:
            result[i] = 0.0

    return result


def _prepare_segments_with_polars(
    input_path: str,
    window: int,
) -> Tuple[int, List[Segment], List[np.ndarray]]:
    df = pl.read_parquet(input_path, columns=["symbol", "Close"]).with_row_count("row_idx")
    grouped = df.groupby("symbol", maintain_order=True).agg(
        [pl.col("row_idx").alias("row_idx"), pl.col("Close").alias("Close")]
    )

    total_rows = df.height
    segments: List[Segment] = []
    zero_blocks: List[np.ndarray] = []

    row_idx_col = grouped["row_idx"]
    close_col = grouped["Close"]

    for row_idx_list, close_list in zip(row_idx_col, close_col):
        idx = np.asarray(row_idx_list, dtype=np.int64)
        values = np.ascontiguousarray(np.asarray(close_list, dtype=np.float64))
        mask = ~np.isnan(values)

        if not mask.any():
            zero_blocks.append(idx)
            continue

        valid_values = values[mask]
        unique_vals, inverse = np.unique(valid_values, return_inverse=True)
        bit_size = int(unique_vals.size + 2)
        use_int16 = bit_size <= 32768 and window <= 32767

        if use_int16:
            compressed = np.zeros(values.size, dtype=np.int16)
            compressed[mask] = inverse.astype(np.int16) + 1
        else:
            compressed = np.zeros(values.size, dtype=np.int32)
            compressed[mask] = inverse.astype(np.int32) + 1

        segments.append((idx, compressed, mask, bit_size, use_int16))

    return total_rows, segments, zero_blocks


def _prepare_segments_with_pandas(
    input_path: str,
    window: int,
) -> Tuple[int, List[Segment], List[np.ndarray]]:
    df = pd.read_parquet(input_path, columns=["symbol", "Close"])
    total_rows = len(df)
    segments: List[Segment] = []
    zero_blocks: List[np.ndarray] = []

    for _, group in df.groupby("symbol", sort=False):
        idx = group.index.to_numpy(dtype=np.int64, copy=False)
        values = np.ascontiguousarray(group["Close"].to_numpy(dtype=np.float64, copy=False))
        mask = ~np.isnan(values)

        if not mask.any():
            zero_blocks.append(idx)
            continue

        valid_values = values[mask]
        unique_vals, inverse = np.unique(valid_values, return_inverse=True)
        bit_size = int(unique_vals.size + 2)
        use_int16 = bit_size <= 32768 and window <= 32767

        if use_int16:
            compressed = np.zeros(values.size, dtype=np.int16)
            compressed[mask] = inverse.astype(np.int16) + 1
        else:
            compressed = np.zeros(values.size, dtype=np.int32)
            compressed[mask] = inverse.astype(np.int32) + 1

        segments.append((idx, compressed, mask, bit_size, use_int16))

    return total_rows, segments, zero_blocks


def _compute_segment(payload: Tuple[int, Segment]) -> Tuple[np.ndarray, np.ndarray]:
    window, segment = payload
    idx, compressed, mask, bit_size, use_int16 = segment

    if use_int16:
        ranks = _rolling_rank_indices_int16(compressed, mask, window, bit_size)
    else:
        ranks = _rolling_rank_indices_int32(compressed, mask, window, bit_size)
    return idx, ranks


def ops_rolling_rank(input_path: str, window: int = 20, max_threads: Optional[int] = None) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be a positive integer")

    total_rows = -1
    segments: List[Segment] = []
    zero_blocks: List[np.ndarray] = []

    if pl is not None:
        try:
            total_rows, segments, zero_blocks = _prepare_segments_with_polars(input_path, window)
        except Exception:  # pragma: no cover - fallback to pandas on polars failure
            total_rows = -1

    if total_rows == -1:
        total_rows, segments, zero_blocks = _prepare_segments_with_pandas(input_path, window)

    if total_rows == 0:
        return np.empty((0, 1), dtype=np.float32)

    output = np.empty(total_rows, dtype=np.float32)

    if zero_blocks:
        output[np.concatenate(zero_blocks)] = 0.0

    segment_count = len(segments)
    if segment_count == 0:
        return output.reshape(-1, 1)

    if max_threads is None:
        cpu_count = os.cpu_count() or 1
        max_threads = max(1, min(cpu_count * 2, segment_count))
    else:
        max_threads = max(1, min(max_threads, segment_count))

    segments.sort(key=lambda seg: seg[1].size, reverse=True)

    if max_threads == 1:
        for segment in segments:
            idx, ranks = _compute_segment((window, segment))
            output[idx] = ranks
    else:
        payloads = ((window, segment) for segment in segments)
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            for idx, ranks in executor.map(_compute_segment, payloads):
                output[idx] = ranks

    return output.reshape(-1, 1)
