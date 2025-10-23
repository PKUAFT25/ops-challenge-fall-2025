# src/solution.py
import math
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ================ Optional: numba JIT =================
try:
    import numba as _nb
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


# ---------------- NumPy fallback: per-segment, low-peak-memory ----------------
def _rolling_percentile_numpy_segment(segment: np.ndarray, window: int, out: np.ndarray, block_rows: int = 65536) -> None:
    """
    简洁低峰值内存版：适合作为后备路径（无 numba 时）。
    """
    n = segment.size
    if n == 0:
        return

    # ramp-up 阶段
    tail = min(window - 1, n)
    for i in range(tail):
        x = segment[i]
        if np.isnan(x):
            out[i] = 0.0
        else:
            out[i] = np.count_nonzero(segment[: i + 1] <= x) / float(i + 1)

    # 固定窗口阶段
    if n >= window:
        windows = sliding_window_view(segment, window_shape=window)
        m = windows.shape[0]
        start = 0
        while start < m:
            end = min(start + block_rows, m)
            blk = windows[start:end]
            cur = blk[:, -1]
            counts = np.count_nonzero(blk <= cur[:, None], axis=1)
            out[window - 1 + start : window - 1 + end] = counts.astype(np.float32) / float(window)
            start = end


# ------------------------- Numba helpers (row-wise) ---------------------------
if _NUMBA_AVAILABLE:
    @_nb.njit(cache=True, fastmath=False)
    def _upper_bound_row(buf_row: np.ndarray, valid_len: int, x: float) -> int:
        """
        返回 buf_row[:valid_len] 中 "第一个 > x 的位置"（升序），即 #<=x 的数量。
        """
        lo = 0
        hi = valid_len
        while lo < hi:
            mid = (lo + hi) // 2
            if buf_row[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo  # #<=x

    @_nb.njit(cache=True, fastmath=False)
    def _insert_sorted_row(buf_row: np.ndarray, valid_len: int, x: float) -> int:
        """
        将 x 插入 buf_row[:valid_len] 的有序位置，返回新长度。
        使用 upper_bound 以匹配 ≤ 的秩定义。
        """
        pos = _upper_bound_row(buf_row, valid_len, x)
        # 右移腾位
        for i in range(valid_len, pos, -1):
            buf_row[i] = buf_row[i - 1]
        buf_row[pos] = x
        return valid_len + 1

    @_nb.njit(cache=True, fastmath=False)
    def _erase_one_row(buf_row: np.ndarray, valid_len: int, x: float) -> int:
        """
        从 buf_row[:valid_len] 中删除一个等于 x 的元素（若存在），返回新长度。
        """
        # lower_bound 找到第一个 >= x 的位置
        lo = 0
        hi = valid_len
        while lo < hi:
            mid = (lo + hi) // 2
            if buf_row[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        # 若找到且相等，则删除并左移
        if lo < valid_len and buf_row[lo] == x:
            for i in range(lo, valid_len - 1):
                buf_row[i] = buf_row[i + 1]
            return valid_len - 1
        return valid_len

    @_nb.njit(cache=True, fastmath=False)
    def _streaming_ssw_multi_symbol(
        codes: np.ndarray,        # (N,) int32/64, 每行的 symbol code（factorize 得到）
        closes: np.ndarray,       # (N,) float32
        window: int,
        S: int,                   # 符号总数 = codes.max()+1
        ranks: np.ndarray         # (N,) float32, 输出
    ) -> None:
        """
        逐行扫描（保持原行序），每个 symbol 维护：
          - win_vals[S, window]: 最近 window 个“原始值”的循环队列（含 NaN）
          - q_head[S], q_len[S]: 循环队列头指针和长度（<=window）
          - buf[S, window]: 有序缓冲（仅非 NaN 值参与）
          - buf_len[S]: 有序缓冲的有效长度
          - seen[S]: 该 symbol 已处理的样本数（含 NaN），用于确定分母（ramp-up/固定窗口）

        对于当前值 x：
          1) seen[s] += 1, win_len = min(seen[s], window)
          2) 若队列已满，从队首取出旧值 y，若 y 非 NaN，则从 buf 删除 y
          3) 将 x 写入循环队列（覆盖或追加）
          4) 若 x 非 NaN，将 x 插入 buf 的有序位置
          5) rank = 0（若 x 为 NaN），否则 rank = (#<=x in buf) / win_len
        """
        # 各 symbol 的状态数组
        win_vals = np.empty((S, window), dtype=np.float32)
        q_head = np.zeros(S, dtype=np.int32)
        q_len = np.zeros(S, dtype=np.int32)
        buf = np.empty((S, window), dtype=np.float32)
        buf_len = np.zeros(S, dtype=np.int32)
        seen = np.zeros(S, dtype=np.int32)

        N = codes.size
        for i in range(N):
            s = int(codes[i])
            x = closes[i]

            # 1) 更新 seen、计算分母
            seen[s] += 1
            win_len = window if seen[s] >= window else seen[s]

            # 2) 若窗口满了，移除最旧值
            if q_len[s] == window:
                y = win_vals[s, q_head[s]]
                # 循环队列头右移
                q_head[s] += 1
                if q_head[s] == window:
                    q_head[s] = 0
                # 从有序缓冲删掉 y（若非 NaN）
                if not math.isnan(y):
                    buf_len[s] = _erase_one_row(buf[s], buf_len[s], y)
                # 覆盖写入新值位置（稍后写）

                # 3) 在覆盖位置写入 x
                idx = (q_head[s] + q_len[s] - 1) % window  # 队列已满，此处等价于旧尾部位置
                win_vals[s, idx] = x
            else:
                # 窗口未满：在尾部追加
                idx = (q_head[s] + q_len[s]) % window
                win_vals[s, idx] = x
                q_len[s] += 1

            # 4) 若 x 非 NaN，插入有序缓冲
            if not math.isnan(x):
                if buf_len[s] == 0:
                    buf[s, 0] = x
                    buf_len[s] = 1
                else:
                    buf_len[s] = _insert_sorted_row(buf[s], buf_len[s], x)

                # 5) 计算 rank：#<=x / win_len
                cnt_le = _upper_bound_row(buf[s], buf_len[s], x)
                ranks[i] = float(cnt_le) / float(win_len)
            else:
                # 当前值 NaN：定义为 0.0
                ranks[i] = 0.0


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    """
    返回 shape (N, 1) 的 float32：
      rank[i] = # {x in window_i | x <= 当前值} / |window_i|
    其中 |window_i| = min(该 symbol 已处理样本数, window)（即 ramp-up 后固定为 window）
    NaN：当前值为 NaN 时 rank=0；NaN 不进入计数缓冲。
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")

    # 只读必须列；Close 用 float32 以利缓存
    df = pd.read_parquet(input_path, columns=["symbol", "Close"])
    closes = df["Close"].to_numpy(dtype=np.float32, copy=False)

    # factorize：获得每行的 symbol code（不排序！）
    # 这是 O(N) 的 C 级哈希映射，比全局 argsort 便宜得多。
    codes, _ = pd.factorize(df["symbol"], sort=False)
    codes = codes.astype(np.int32, copy=False)

    n = closes.size
    out = np.empty(n, dtype=np.float32)

    if n == 0:
        return out.reshape(-1, 1)

    if _NUMBA_AVAILABLE:
        S = int(codes.max()) + 1
        _streaming_ssw_multi_symbol(codes, closes, int(window), S, out)
    else:
        # 无 numba：回退为“按 symbol 分段”的 NumPy 版
        # 为了避免全局排序，这里用 pandas groupby 的顺序分组（哈希），不会改变原有行序。
        # 我们逐段取出索引并在段内计算，再写回 out 的对应位置。
        # （注意：这个路径在 4400 万行下会明显慢于 numba，但语义一致）
        # 组内仍使用低峰值分块算法
        for _, idx in df.groupby("symbol").groups.items():
            idx = np.fromiter(idx, dtype=np.int64)
            seg = closes[idx]
            tmp = np.empty_like(seg, dtype=np.float32)
            _rolling_percentile_numpy_segment(seg, window, tmp)
            out[idx] = tmp

    return out.reshape(-1, 1)
