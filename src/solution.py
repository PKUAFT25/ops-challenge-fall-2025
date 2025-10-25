import pandas as pd
import numpy as np
from numba import njit


@njit
def rolling_rank_numba(values, window):
    n = len(values)
    result = np.empty(n, dtype=np.float32)
    
    for i in range(n):
        start = max(0, i - window + 1)
        window_data = values[start:i+1]
        current_val = values[i]
        
        rank_sum = 0
        for j in range(len(window_data)):
            if window_data[j] <= current_val:
                rank_sum += 1
        
        result[i] = rank_sum / len(window_data)
    
    return result


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    df = pd.read_parquet(
        input_path, 
        columns=['symbol', 'Close'],
        engine='pyarrow',
        use_threads=True
    )
    
    if df['Close'].dtype == np.float64:
        df['Close'] = df['Close'].astype(np.float32)
    
    if df['symbol'].dtype == 'object':
        df['symbol'] = df['symbol'].astype('category')
    
    result = np.empty(len(df), dtype=np.float32)
    
    grouped = df.groupby('symbol', sort=False, observed=True)
    
    indices_list = grouped.indices
    closes_array = df['Close'].values
    
    for symbol, idx_array in indices_list.items():
        values = closes_array[idx_array]
        ranks = rolling_rank_numba(values, window)
        result[idx_array] = ranks
    
    return result[:, None]
