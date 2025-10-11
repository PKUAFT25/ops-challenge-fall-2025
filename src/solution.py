import polars as pl
import numpy as np 

from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True)
def _rolling_rank_numba(values, window):
    n = len(values)
    result = np.empty(n, dtype=np.float32)

    for i in prange(n):
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        window_size = end_idx - start_idx

        if window_size == 0:
            result[i] = np.nan
        else:
            current_val = values[i]
            rank_sum = 0.0
            for j in range(start_idx, end_idx):
                if values[j] <= current_val:
                    rank_sum += 1.0
            result[i] = rank_sum / window_size

    return result


class ops:
    @staticmethod
    def rolling_rank(col_or_expr, window: int) -> pl.Expr:
        def rolling_rank(s: pl.Series) -> pl.Series:
            values = s.to_numpy()
            result = _rolling_rank_numba(values, window)
            return pl.Series(result, dtype=pl.Float32)

        if isinstance(col_or_expr, str):
            expr = pl.col(col_or_expr)
        else:
            expr = col_or_expr

        return expr.map_batches(rolling_rank)

def ops_rolling_rank(
    input_path: str, 
    window: int = 20) -> np.ndarray:
    """
    template
    """
    df = pl.read_parquet(input_path)
    res = df.select(ops.rolling_rank("Close", window).over("symbol"))
    return res.to_numpy()


