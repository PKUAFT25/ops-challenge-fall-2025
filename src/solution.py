import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    """Vectorized rolling percentile rank per-symbol.

    For each row, compute fraction of values in the previous up-to-`window` values
    (including current) that are <= current value. Returns shape [N,1] float32.
    """

    df = pd.read_parquet(input_path)
    n = len(df)
    out = np.empty(n, dtype=np.float32)

    # process each symbol group and write results back to the output array
    for _, grp in df.groupby('symbol', sort=False):
        idx = grp.index.to_numpy()
        arr = grp['Close'].to_numpy()
        m = arr.shape[0]

        if m == 0:
            continue

        ranks = np.empty(m, dtype=np.float64)

        # prefix where window size grows from 1..window-1
        small_end = min(m, window - 1)
        for t in range(small_end):
            win = arr[: t + 1]
            cur = arr[t]
            ranks[t] = np.sum(win <= cur) / (t + 1)

        # full-window case using vectorized sliding windows
        if m >= window:
            windows = sliding_window_view(arr, window)
            cur_vals = arr[window - 1 :]
            counts = np.sum(windows <= cur_vals[:, None], axis=1)
            ranks[window - 1 :] = counts / float(window)

        out[idx] = ranks.astype(np.float32)

    return out[:, None]


