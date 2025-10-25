import numpy as np
import polars as pl
from typing import Union

# --- Polars 算子包裝類 ---
class ops:
    """
    為 Polars DataFrame 封裝原生高性能算子的類。
    """
    @staticmethod
    def rolling_rank(
        col_or_expr: Union[str, pl.Expr],
        window: int
    ) -> pl.Expr:
        """
        創建 Polars 表達式，使用原生函數計算滾動百分位排名。
        此實現旨在匹配原始 Numba 函數的行為：
        rank = (count <= current) / window_slice_length

        - 分子: 使用 rolling_rank(method='max') 計算 count <= current。
        - 分母: 使用 pl.min_horizontal(pl.arange(0, pl.len()) + 1, window) 計算窗口切片總長度。
        - 處理 NaN: 如果當前值為 NaN，結果強制為 NaN。
        """
        if isinstance(col_or_expr, str):
            expr = pl.col(col_or_expr)
        else:
            expr = col_or_expr

        # 1. 計算窗口切片的實際總長度 (分母)
        #    等同於 Numba 中的 end_idx - start_idx
        denominator_expr = pl.min_horizontal(
            pl.arange(0, pl.len()) + 1, # row_index + 1
            pl.lit(window)              # window_size
        ).cast(pl.Float32)

        # 2. 計算滾動排名 (分子)
        #    method='max' 匹配 Numba 的 "count <= current" 邏輯
        #    使用 min_samples=1 保持滾動的起始行為
        rank_expr = expr.rolling_rank(
            window_size=window,
            method='max',
            min_samples=1 # 從第一個有效樣本開始計算
        )

        # 3. 計算基礎的百分位排名
        #    rank / 窗口總長度
        base_rank_pct = (rank_expr.cast(pl.Float32) / denominator_expr).cast(pl.Float32)

        # 4. 處理當前值為 NaN 的情況
        #    如果 expr 本身是 NaN，則結果強制為 NaN
        result_expr = pl.when(expr.is_nan()).then(pl.lit(np.nan)).otherwise(base_rank_pct)

        return result_expr

# --- 完整的執行函數（示範） ---
def ops_rolling_rank(
    input_path: str,
    window: int = 20
) -> np.ndarray:
    """
    完整演示：讀取 Parquet 文件，對每個 'symbol' 分組計算滾動排名。
    使用 Polars 原生函數實現，其行為與原始 Numba 函數對齊。

    Args:
        input_path: Parquet 文件路徑。
        window: 滾動窗口大小。

    Returns:
        計算出的滾動排名結果（[N, 1] 的 NumPy 數組）。
    """
    try:
        # 1. 讀取 Parquet 文件
        df = pl.read_parquet(input_path)
    except FileNotFoundError:
        print(f"錯誤：未找到文件 {input_path}。請確保路徑正確或使用 pl.DataFrame(...) 創建一個示例數據。")
        # 假設我們無法讀取文件，返回空數組
        return np.array([[]], dtype=np.float32)

    # 2. 使用 Polars 表達式進行分組和計算
    #    .over("symbol") 會對每個 'symbol' 組獨立執行滾動排名操作
    res = df.select(
        # 調用 ops 類中的 rolling_rank 靜態方法
        ops.rolling_rank("Close", window).over("symbol").alias("RollingRank")
    )

    # 3. 返回結果的 NumPy 數組
    #    [:, None] 用於確保輸出是 [N, 1] 形狀的二維數組
    return res["RollingRank"].to_numpy().astype(np.float32)[:, None]

# --- 示範如何調用 (需要您提供一個 parquet 文件路徑) ---
# if __name__ == "__main__":
#     # 替換為您的 Parquet 文件路徑
#     parquet_file = "your_data.parquet"
#
#     # 創建示例 Parquet 文件 (如果需要)
#     try:
#         pl.read_parquet(parquet_file)
#     except FileNotFoundError:
#         print(f"創建示例文件: {parquet_file}")
#         example_df = pl.DataFrame({
#             "symbol": ["A"] * 50 + ["B"] * 50,
#             "Close": np.random.rand(100) * 100
#         })
#         example_df.write_parquet(parquet_file)
#
#     # 執行計算
#     rolling_ranks = ops_rolling_rank(parquet_file, window=10)
#     print("\n計算完成，結果數組形狀:", rolling_ranks.shape)
#     # print("部分結果:")
#     # print(rolling_ranks[:5])
#     # print("...")
#     # print(rolling_ranks[50:55])