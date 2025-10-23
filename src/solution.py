# src/solution.py
import math
import numpy as np
import pandas as pd
from numba import njit

@njit(cache=True, fastmath=True)
def _rolling_rank_numba(values: np.ndarray, window: int) -> np.ndarray:
    """
    对单个一维序列计算 rolling 百分位秩：
      rank[i] = #(x in window_i | x <= values[i]) / |window_i|
    其中 |window_i| = i - start + 1（ramp-up），之后固定为 window。
    NaN 规则：当前值为 NaN → rank=0；窗口内遇到 NaN 时跳过不计数。
    """
    n = values.shape[0]
    result = np.empty(n, dtype=np.float32)

    for i in range(n):
        cur = values[i]
        start = i - window + 1
        if start < 0:
            start = 0

        span = i - start + 1  # 分母
        if math.isnan(cur):
            result[i] = 0.0
            continue

        cnt = 0
        # 小窗口 O(window) 计数
        for j in range(start, i + 1):
            v = values[j]
            if not math.isnan(v) and v <= cur:
                cnt += 1

        result[i] = cnt / span

    return result


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    """
    返回 shape (N, 1), dtype float32 的 rolling 百分位秩（按 symbol 独立计算）。
    与评测脚本 verify.py 的接口与容差要求完全一致。
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")

    # 只读必要列，减少 IO；Close 用 float32 以降低带宽
    df = pd.read_parquet(input_path, columns=["symbol", "Close"])
    if df.empty:
        return np.empty((0, 1), dtype=np.float32)

    window = int(window)
    close_values = df["Close"].to_numpy(dtype=np.float32, copy=False)
    out = np.empty(close_values.shape[0], dtype=np.float32)

    # 保持原行序、避免全局排序：对每个 symbol 的位置索引做一次 njit 调用
    # pandas groupby(..., sort=False).indices 的开销 << 全局 argsort(几千万行)
    groups = df.groupby("symbol", sort=False).indices
    for idx in groups.values():
        # idx 是一组的整型位置索引；确保 values 连续内存，以便 numba 高效访问
        gvals = np.ascontiguousarray(close_values[idx])
        out[idx] = _rolling_rank_numba(gvals, window)

    return out.reshape(-1, 1)
