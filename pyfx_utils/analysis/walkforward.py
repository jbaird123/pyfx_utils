
from __future__ import annotations
import pandas as pd
import numpy as np
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

def summarize_walk_forward(wf: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize a list of walk-forward fold dicts. Each fold may contain:
      - 'train_total_pips' : float
      - 'test_total_pips'  : float
    Returns folds count, % positive tests, and medians of train/test totals.
    """
    if not isinstance(wf, list) or not wf:
        return {"folds": 0, "pct_positive_tests": 0.0,
                "median_test_total_pips": 0.0, "median_train_total_pips": 0.0}

    test_totals  = np.array([float(w.get("test_total_pips", 0.0)) for w in wf], dtype="float64")
    train_totals = np.array([float(w.get("train_total_pips", 0.0)) for w in wf], dtype="float64")

    return {
        "folds": int(len(wf)),
        "pct_positive_tests": float((test_totals > 0).mean() * 100.0),
        "median_test_total_pips": float(np.median(test_totals)) if test_totals.size else 0.0,
        "median_train_total_pips": float(np.median(train_totals)) if train_totals.size else 0.0,
    }
