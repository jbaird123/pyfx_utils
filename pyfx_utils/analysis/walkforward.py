
from __future__ import annotations
import pandas as pd
from typing import List, Tuple

def walk_forward_ranges(
    df: pd.DataFrame,
    n_splits: int = 6,
    min_train: int = 2000,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    idx = df.index
    if len(idx) < (min_train + 10):
        return []
    step = len(idx) // (n_splits + 1)
    ranges = []
    start = 0
    for _ in range(n_splits):
        tr_start = start
        tr_end   = start + step
        te_end   = tr_end + step
        if (tr_end - tr_start) < min_train or te_end > len(idx):
            break
        ranges.append((idx[tr_start], idx[tr_end - 1], idx[tr_end], idx[te_end - 1]))
        start += step
    return ranges
