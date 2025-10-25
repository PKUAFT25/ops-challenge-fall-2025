import polars as pl
import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, nogil=True)
def _rolling_rank_numba(values, window, out):
    n = len(values)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        window_size = end_idx - start_idx
        if window_size == 0:
            out[i] = np.nan
        else:
            current_val = values[i]
            rank_sum = 0.0
            for j in range(start_idx, end_idx):
                if values[j] <= current_val:
                    rank_sum += 1.0
            out[i] = rank_sum / window_size
    return out


class ops:
    @staticmethod
    def rolling_rank(col_or_expr, window: int) -> pl.Expr:
        def rolling_rank(s: pl.Series) -> pl.Series:
            values = s.to_numpy()
            result = np.empty(values.shape[0], dtype=np.float32)
            _rolling_rank_numba(values, window, result)
            return result

        if isinstance(col_or_expr, str):
            expr = pl.col(col_or_expr)
        else:
            expr = col_or_expr
        return expr.map_batches(rolling_rank)
    

def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    res = (
        pl.scan_parquet(input_path)
        .with_columns(pl.col("Close").cast(pl.Float32))
        .select(
            ops.rolling_rank("Close", window).over("symbol")
        )
    ).collect()
    return res.to_numpy()
