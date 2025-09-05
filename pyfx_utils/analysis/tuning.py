from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type
import random
import pandas as pd
import numpy as np
import os
import math
from datetime import datetime

from pyfx_utils.backtests.core import backtest
from pyfx_utils.utils.stats import cumulative_pips


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


# --- Keep the sanitizer we discussed (paste if not present yet) ---
def sanitize_tuning_block(tuning: dict) -> dict:
    """Ensure *_value is None whenever the corresponding *_type is None."""
    if not isinstance(tuning, dict):
        return tuning

    out = dict(tuning)

    # Fix list of top rows
    top = out.get("top")
    if isinstance(top, list):
        fixed = []
        for row in top:
            r = dict(row)
            if r.get("param_stop_type") is None:
                r["param_stop_value"] = None
            if r.get("param_tp_type") is None:
                r["param_tp_value"] = None
            if r.get("param_trail_type") is None:
                r["param_trail_value"] = None
            fixed.append(r)
        out["top"] = fixed

    # Fix best_params map
    bp = out.get("best_params")
    if isinstance(bp, dict):
        bp = dict(bp)
        if bp.get("stop_type") is None:
            bp["stop_value"] = None
        if bp.get("tp_type") is None:
            bp["tp_value"] = None
        if bp.get("trail_type") is None:
            bp["trail_value"] = None
        out["best_params"] = bp

    return out


# --- Core: run_param_search ----------------------------------------------------

@dataclass
class ParamSearchResult:
    """Internal holder for a single parameter combination evaluation."""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    n_trades: int
    # Optional: keep a pointer if you want (not returned by default)
    # trades: Optional[pd.DataFrame] = None


def _expand_param_grid(param_grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """
    Expand a scikit-like param_grid (each value is an iterable) into a list of
    concrete parameter dicts. Handles values including None.
    Example:
        {"fast": [10, 20], "slow": [50], "stop_type": [None, "atr"], "stop_value": [None, 2]}
    """
    keys = list(param_grid.keys())
    values = [list(v) for v in param_grid.values()]
    combos = []
    for tpl in product(*values):
        p = {k: v for k, v in zip(keys, tpl)}
        combos.append(p)
    return combos


def _default_score(metrics: Dict[str, float]) -> float:
    """
    Ranking score if the caller doesn’t provide one.
    Priority: higher total pips, then higher mean pips, then less drawdown.
    """
    total = metrics.get("total_pips", 0.0)
    mean_ = metrics.get("mean_pips", 0.0)
    mdd  = metrics.get("max_dd_pips", 0.0)  # typically negative
    return (total, mean_, -mdd)  # tuple ordering works with Python sort


def run_param_search(
    *,
    data: pd.DataFrame,
    # A callable that, given (data, params), returns a trades DataFrame
    # with at least a float "pips" column; times optional but helpful.
    strategy_runner: Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame],
    strict: bool = True,  #fails loudly if there are duplicates present.
    # Dict[str, Iterable] just like sklearn's GridSearchCV param_grid
    param_grid: Dict[str, Iterable[Any]],
    # Optional scoring function that maps the per-combo metrics -> sortable key
    score_fn: Optional[Callable[[Dict[str, float]], Any]] = None,
    # Limit the number of top rows kept
    top_n: int = 20,
    # When True, keeps only metric columns + params in "top"
    include_extra_cols: bool = False,
) -> Dict[str, Any]:
    """
    Run a simple, robust grid search over a strategy's parameter space.

    Parameters
    ----------
    data : DataFrame
        OHLC(V) price data used by `strategy_runner`.
    strategy_runner : Callable
        Function that executes the strategy and returns a trades ledger (DataFrame)
        with a 'pips' column. E.g. `lambda df, p: run_sma(df, p)`.
    param_grid : dict[str, Iterable]
        Dict mapping param name -> iterable of values (including None allowed).
    score_fn : Callable(metrics) -> sort key
        Optional ranking function. Defaults to a reasonable pip-centric tuple.
    top_n : int
        Keep this many best rows in the "top" list.
    include_extra_cols : bool
        If True, include additional computed columns in each top row.

    Returns
    -------
    tuning : dict
        {
          "searched_params": {...},     # the grid you searched
          "n_evals": int,
          "top": [ { "param_*": ..., metrics... }, ... ],   # up to top_n best
          "best_params": {...},         # the parameters of the best row
          "best_metrics": {...}         # metrics of the best row
        }
        (Sanitized so *_value is None when *_type is None.)
    """
    score_fn = score_fn or _default_score

    # --- Expand raw Cartesian grid
    combos_raw = _expand_param_grid(param_grid)

    # --- Canonicalize & DEDUPE BEFORE evaluation (avoid functional duplicates)
    def _canonicalize_params_for_effect(p: dict) -> dict:
        q = dict(p)
        if q.get("stop_type")  is None: q["stop_value"]  = None
        if q.get("tp_type")    is None: q["tp_value"]    = None
        if q.get("trail_type") is None: q["trail_value"] = None
        return q

    def _hashable_key(d: dict):
        def _norm(v):
            return None if (isinstance(v, float) and math.isnan(v)) else v
        return tuple(sorted((k, _norm(v)) for k, v in d.items()))

    seen = set()
    combos: List[Dict[str, Any]] = []
    for p in combos_raw:
        eff = _canonicalize_params_for_effect(p)
        key = _hashable_key(eff)
        if key not in seen:
            seen.add(key)
            combos.append(eff)
    def _is_valid_combo(p: dict) -> bool:
        # If type is None, paired value must be None
        if p.get("stop_type") is None and p.get("stop_value") is not None:
            return False
        if p.get("tp_type") is None and p.get("tp_value") is not None:
            return False
        if p.get("trail_type") is None and p.get("trail_value") is not None:
            return False

        # If type is set, value must be a real number (not None/NaN)
        import math
        if p.get("stop_type") is not None:
            v = p.get("stop_value")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return False
        if p.get("tp_type") is not None:
            v = p.get("tp_value")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return False
        if p.get("trail_type") is not None:
            v = p.get("trail_value")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return False
        return True

    invalid = [p for p in combos if not _is_valid_combo(p)]
    if invalid:
        raise RuntimeError(
            "Invalid parameter combinations in grid (type/value mismatch). "
            "First few examples: " + repr(invalid[:5])
        )

    rows: List[ParamSearchResult] = []

    for params in combos:
        try:
            trades = strategy_runner(data, params)

            # Basic checks
            if trades is None or not isinstance(trades, pd.DataFrame) or "pips" not in trades.columns:
                # Treat as zero-result
                metrics = {
                    "n_trades": 0, "total_pips": 0.0, "mean_pips": 0.0,
                    "median_pips": 0.0, "max_dd_pips": 0.0, "win_rate": 0.0
                }
                rows.append(ParamSearchResult(params=params, metrics=metrics, n_trades=0))
                continue

            t = trades.copy()
            t["pips"] = pd.to_numeric(t["pips"], errors="coerce")
            t = t.dropna(subset=["pips"])

            n_tr = int(len(t))
            total = float(t["pips"].sum()) if n_tr else 0.0
            mean_ = float(t["pips"].mean()) if n_tr else 0.0
            median_ = float(t["pips"].median()) if n_tr else 0.0

            # trade-sequence MDD in pips (no time needed)
            cp = t["pips"].cumsum()
            dd = cp - cp.cummax()
            mdd = float(dd.min()) if len(dd) else 0.0

            wins = (t["pips"] > 0)
            win_rate = float(wins.mean()) if n_tr else 0.0

            metrics = {
                "n_trades": n_tr,
                "total_pips": total,
                "mean_pips": mean_,
                "median_pips": median_,
                "max_dd_pips": mdd,
                "win_rate": win_rate,
            }

            rows.append(ParamSearchResult(params=params, metrics=metrics, n_trades=n_tr))
        except Exception:
            # Fail-soft: count as a zero row; keep search robust
            metrics = {
                "n_trades": 0, "total_pips": 0.0, "mean_pips": 0.0,
                "median_pips": 0.0, "max_dd_pips": 0.0, "win_rate": 0.0
            }
            rows.append(ParamSearchResult(params=params, metrics=metrics, n_trades=0))

    # Build a DataFrame to sort and select top rows
    recs = []
    for r in rows:
        rec: Dict[str, Any] = {}
        # Flatten params into "param_*" columns, mirroring sklearn style
        for k, v in r.params.items():
            rec[f"param_{k}"] = v
        rec.update(r.metrics)
        # Optional derived score (not kept unless include_extra_cols=True)
        rec["_score"] = score_fn(r.metrics)
        recs.append(rec)

    if len(recs) == 0:
        tuning = {
            "searched_params": {k: list(v) for k, v in param_grid.items()},
            "n_evals": 0,
            "top": [],
            "best_params": {},
            "best_metrics": {},
        }
        return sanitize_tuning_block(tuning)

    df = pd.DataFrame(recs)
    param_cols = [c for c in df.columns if c.startswith("param_")]
    if not df.empty and df[param_cols].isnull().any().any():
        bad = [c for c in param_cols if df[c].isnull().any()]
        raise RuntimeError(
            f"Param columns contain NaN after evaluation: {bad}. "
            "Likely a type/value mismatch or Pandas coercion from mixed dtypes. "
            "Ensure that when *_type is not None, the paired *_value is numeric; "
            "and when *_type is None, the paired *_value is None."
        )

    # --- STRICT RESULT CHECK: no duplicate *effective* params allowed
    def _effective_key_from_row(row: pd.Series):
        eff = {k.replace("param_", ""): row[k] for k in row.index if k.startswith("param_")}
        eff = _canonicalize_params_for_effect(eff)
        return _hashable_key(eff)

    if strict and not df.empty:
        keys = df.apply(_effective_key_from_row, axis=1)
        dup_mask = keys.duplicated(keep=False)
        if dup_mask.any():
            offenders = df.loc[
                dup_mask,
                [c for c in df.columns if c.startswith("param_")] + ["n_trades","total_pips","mean_pips","max_dd_pips"]
            ]
            examples = offenders.head(6).to_dict(orient="records")
            raise RuntimeError(
                "Duplicate effective parameter sets encountered in results. "
                "Fix your grid (or pre-eval dedupe). Examples (truncated): "
                f"{examples}"
            )

    # Sort by our score (tuple works), descending (best first)
    # Pandas can't sort by arbitrary python tuples directly, so map to rank key
    # Build a rank array by applying score_fn again (cheap)
    sort_keys = df.apply(lambda row: score_fn({
        "total_pips": row.get("total_pips", 0.0),
        "mean_pips": row.get("mean_pips", 0.0),
        "max_dd_pips": row.get("max_dd_pips", 0.0),
        "n_trades": row.get("n_trades", 0),
        "win_rate": row.get("win_rate", 0.0),
    }), axis=1)

    # Convert sort_keys (tuples) into a DataFrame to sort by multiple columns
    # This supports the default triple (total, mean, -mdd)
    sort_cols = pd.DataFrame(sort_keys.tolist(), index=df.index)
    # Default: larger is better for col0, col1; for col2 we already flipped sign
    order = [False] * sort_cols.shape[1]  # descending
    df = df.join(sort_cols.rename(columns=lambda i: f"_k{i+1}"))
    df = df.sort_values(by=[c for c in df.columns if c.startswith("_k")], ascending=order)

    # Build "top" rows
    cols = [c for c in df.columns if c.startswith("param_")] + [
        "n_trades", "total_pips", "mean_pips", "median_pips", "max_dd_pips", "win_rate"
    ]
    if include_extra_cols:
        cols = cols + [c for c in df.columns if c.startswith("_k")] + (["_score"] if "_score" in df.columns else [])

    top_rows = df[cols].head(int(top_n)).to_dict(orient="records")

    # Best row
    best = top_rows[0] if len(top_rows) else {}
    best_params = {k.replace("param_", ""): v for k, v in best.items() if k.startswith("param_")}
    best_metrics = {k: best[k] for k in ("n_trades", "total_pips", "mean_pips", "median_pips", "max_dd_pips", "win_rate") if k in best}

    tuning = {
        "searched_params": {k: list(v) for k, v in param_grid.items()},
        "n_evals": int(len(df)),
        "top": top_rows,
        "best_params": best_params,
        "best_metrics": best_metrics,
    }
    return sanitize_tuning_block(tuning)
