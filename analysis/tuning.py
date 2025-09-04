# pyfx_utils/analysis/tuning.py
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type
import random
import pandas as pd
import os
from datetime import datetime

from pyfx_utils.backtests.core import backtest
from pyfx_utils.utils.stats import cumulative_pips
import numpy as np

def _py_scalar(v):
    # cast numpy scalars -> Python scalars; make 10.0 -> 10
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return int(v) if float(v).is_integer() else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return pd.Timestamp(v)  # already fine
    # strings or other objects pass through
    return v


# ---------- Types ----------
ObjectiveFn = Callable[[pd.DataFrame, pd.Index], float]  # (trades, index) -> score

@dataclass
class TrialResult:
    params: Dict[str, Any]
    trades: int
    total_pips: float
    max_dd_pips: float
    objective: float

# ---------- Utilities ----------
def _param_combos(param_grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    vals = [list(v) for v in param_grid.values()]
    return [{k: v for k, v in zip(keys, combo)} for combo in product(*vals)]

def _max_drawdown_pips(cum: pd.Series) -> float:
    if cum.empty:
        return 0.0
    dd = cum - cum.cummax()
    return float(dd.min())  # negative or 0

# ---------- Built-in Objectives (pips-only) ----------
def obj_total_pips(trades: pd.DataFrame, index: pd.Index) -> float:
    """Maximize final cumulative pips."""
    if trades is None or trades.empty:
        return 0.0
    cum = cumulative_pips(trades, index, when="exit")
    return float(cum.iloc[-1])

def obj_total_minus_half_dd(trades: pd.DataFrame, index: pd.Index) -> float:
    """Total pips penalized by drawdown: total + 0.5 * (negative DD)."""
    if trades is None or trades.empty:
        return 0.0
    cum = cumulative_pips(trades, index, when="exit")
    total = float(cum.iloc[-1])
    mdd = _max_drawdown_pips(cum)  # <= 0
    return total + 0.5 * mdd

def obj_pips_over_dd(trades: pd.DataFrame, index: pd.Index) -> float:
    """Risk-adjusted: total_pips / abs(max_drawdown_pips). Higher is better."""
    if trades is None or trades.empty:
        return 0.0
    cum = cumulative_pips(trades, index, when="exit")
    total = float(cum.iloc[-1])
    mdd = abs(_max_drawdown_pips(cum))  # >= 0
    return total / mdd if mdd > 0 else 0.0

def obj_win_rate_bias(trades: pd.DataFrame, index: pd.Index, alpha: float = 100.0) -> float:
    """
    Example composite: total_pips + alpha * win_rate
    Adjust alpha to trade off scale difference.
    """
    if trades is None or trades.empty:
        return 0.0
    total = float(trades["pips"].sum())
    win_rate = float((trades["pips"] > 0).mean()) if len(trades) else 0.0
    return total + alpha * win_rate

def obj_many_trades_low_dd(trades, index):
    """
    Favor many trades but penalize drawdown.
    """
    if trades is None or trades.empty:
        return 0.0
    cum = cumulative_pips(trades, index, when="exit")
    total = float(cum.iloc[-1])
    mdd = abs((cum - cum.cummax()).min())
    n = len(trades)
    return total + 0.1 * n - 0.5 * mdd


# ---------- Search Runners ----------
def grid_search(
    df: pd.DataFrame,
    strategy_cls: Type,                    # e.g., SMACrossoverStrategy
    param_grid: Dict[str, Iterable[Any]],  # {"fast":[5,10], "slow":[20,50]}
    pip: float,
    objective: ObjectiveFn = obj_total_pips,
    when: str = "exit",
) -> pd.DataFrame:
    """
    Generic, pips-only grid search for any Strategy that works with your core backtest.
    Returns a DataFrame sorted by 'objective' (desc).
    """
    results: List[TrialResult] = []
    for params in _param_combos(param_grid):
        strat = strategy_cls(**params)
        trades = backtest(df, strat, pip=pip)

        cum = cumulative_pips(trades, df.index, when=when)
        total = float(cum.iloc[-1]) if not cum.empty else 0.0
        mdd   = _max_drawdown_pips(cum)
        obj   = float(objective(trades, df.index))

        results.append(TrialResult(params, int(len(trades)), total, mdd, obj))

    rows = []
    for r in results:
        row = {f"param_{k}": _py_scalar(v) for k, v in r.params.items()}
        row.update(dict(trades=r.trades, total_pips=r.total_pips, max_dd_pips=r.max_dd_pips, objective=r.objective))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    return out

def random_search(
    df: pd.DataFrame,
    strategy_cls: Type,
    param_grid: Dict[str, Sequence[Any]],  # sequences so we can sample
    pip: float,
    n_trials: int,
    seed: Optional[int] = None,
    objective: ObjectiveFn = obj_total_pips,
    when: str = "exit",
) -> pd.DataFrame:
    """
    Randomly sample N parameter combinations from the grid (with replacement).
    Useful when the full grid is large.
    """
    rng = random.Random(seed)
    keys = list(param_grid.keys())
    values = [list(v) for v in param_grid.values()]

    results: List[TrialResult] = []
    for _ in range(n_trials):
        params = {k: rng.choice(vlist) for k, vlist in zip(keys, values)}
        strat = strategy_cls(**params)
        trades = backtest(df, strat, pip=pip)

        cum = cumulative_pips(trades, df.index, when=when)
        total = float(cum.iloc[-1]) if not cum.empty else 0.0
        mdd   = _max_drawdown_pips(cum)
        obj   = float(objective(trades, df.index))

        results.append(TrialResult(params, int(len(trades)), total, mdd, obj))

    rows = []
    for r in results:
        row = {f"param_{k}": _py_scalar(v) for k, v in r.params.items()}
        row.update(dict(trades=r.trades, total_pips=r.total_pips, max_dd_pips=r.max_dd_pips, objective=r.objective))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    return out

# ---------- Convenience ----------
def best_params(results: pd.DataFrame) -> Dict[str, Any]:
    """Extract the top row's parameter dict from results of grid_search/random_search."""
    if results is None or results.empty:
        return {}
    param_cols = [c for c in results.columns if c.startswith("param_")]
    best = results.iloc[0]
    # Coerce to clean Python scalars
    return {c.replace("param_", ""): _py_scalar(best[c]) for c in param_cols}

def top_k(results: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Return the top-k rows by 'objective' (already sorted in our runners, but
    this works even if caller re-sorted). Includes only param_* and key metrics.
    """
    if results is None or results.empty:
        return results
    cols = [c for c in results.columns if c.startswith("param_")] + [
        "trades", "total_pips", "max_dd_pips", "objective"
    ]
    cols = [c for c in cols if c in results.columns]
    return results.sort_values("objective", ascending=False).head(k)[cols].reset_index(drop=True)

def refit_best(
    df: pd.DataFrame,
    strategy_cls: Type,
    results: pd.DataFrame,
    pip: float,
    when: str = "exit",) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Refit the best params from a results table, run the backtest, and return:
      trades_df, params_dict
    """
    params = best_params(results)
    if not params:
        return pd.DataFrame(), {}
    strat = strategy_cls(**params)
    trades = backtest(df, strat, pip=pip)
    return trades, params

def walk_forward(
    df, strategy_cls, param_grid, pip, objective=obj_total_pips,
    n_splits=4, when="exit"
):
    """
    Split by time into n_splits folds; for each fold:
      - tune on fold i (grid_search)
      - refit best on fold i
      - evaluate on fold i+1 (out-of-sample)
    Returns: list of dicts per fold with in/out metrics and params.
    """
    import numpy as np
    from .tuning import grid_search, best_params
    from pyfx_utils.backtests.core import backtest
    from pyfx_utils.utils.stats import cumulative_pips

    n = len(df)
    edges = np.linspace(0, n, n_splits+1, dtype=int)
    reports = []
    for i in range(n_splits-1):
        train = df.iloc[edges[i]:edges[i+1]]
        test  = df.iloc[edges[i+1]:edges[i+2]]

        res   = grid_search(train, strategy_cls, param_grid, pip, objective=objective, when=when)
        params = best_params(res)
        strat  = strategy_cls(**params)

        tr_train = backtest(train, strat, pip=pip)
        tr_test  = backtest(test,  strat, pip=pip)

        cum_train = cumulative_pips(tr_train, train.index, when=when)
        cum_test  = cumulative_pips(tr_test,  test.index,  when=when)

        reports.append({
            "fold": i+1,
            "params": params,
            "train_total_pips": float(cum_train.iloc[-1]) if len(cum_train) else 0.0,
            "test_total_pips":  float(cum_test.iloc[-1])  if len(cum_test) else 0.0,
            "train_max_dd": float((cum_train - cum_train.cummax()).min()) if len(cum_train) else 0.0,
            "test_max_dd":  float((cum_test  - cum_test.cummax()).min())  if len(cum_test)  else 0.0,
            "train_trades": int(len(tr_train)),
            "test_trades":  int(len(tr_test)),
        })
    return pd.DataFrame(reports)

DEFAULT_OUTDIR = os.getenv("PYFX_OUTDIR", "")  # set this in the notebook if you want
def save_results(df: pd.DataFrame, path: str, fmt: str = "csv", add_timestamp: bool = True) -> str:
    """
    Save a DataFrame of tuning results to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Results to save.
    path : str
        Target file path. Extension is optional; will be inferred from `fmt`.
    fmt : {"csv","parquet"}
        File format.
    add_timestamp : bool
        If True, append _YYYYMMDD_HHMMSS before extension to avoid overwriting.

    Returns
    -------
    str : final file path actually written.
    """
    fmt = fmt.lower()
    if fmt not in ("csv", "parquet"):
        raise ValueError("fmt must be 'csv' or 'parquet'")

    base, ext = os.path.splitext(path)
    if not ext:
        ext = f".{fmt}"
    if DEFAULT_OUTDIR and not os.path.isabs(base):
        os.makedirs(DEFAULT_OUTDIR, exist_ok=True)
        base = os.path.join(DEFAULT_OUTDIR, base)
    if add_timestamp:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{base}_{stamp}{ext}"
    else:
        path = f"{base}{ext}"

    if fmt == "csv":
        df.to_csv(path, index=False)
    else:  # parquet
        df.to_parquet(path, index=False)

    return path

def load_results(path: str) -> pd.DataFrame:
    """
    Load results DataFrame from a saved CSV or Parquet file.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

