from __future__ import annotations
import inspect
import math
import pandas as pd

from typing import Any, Dict, Sequence, Type, Optional

from ..backtests.signal import BTConfig
from ..backtests.core import backtest
from ..utils.stats import cumulative_pips
from .regime import perf_by_regime, perf_by_regime_pips, build_regime_features, kmeans_regimes

# Canonical imports 
from .tuning import (
    grid_search,
    random_search,
    walk_forward,
    top_k,
    best_params,
    ObjectiveFn,
    obj_total_pips,
    _py_scalar
)

def adf_test(series: pd.Series, autolag: str = "AIC") -> Dict[str, Any]:
    """
    Augmented Dickey–Fuller test (optional dependency).
    Returns a dict with keys: available, statistic, pvalue, lags, nobs, critical_1%, critical_5%, critical_10%, icbest.
    If statsmodels isn't available or data is too short, returns {'available': False, 'error': '...'}.
    """
    try:
        # Lazy import so the rest of the package doesn't require statsmodels
        from statsmodels.tsa.stattools import adfuller  # type: ignore
    except Exception as e:
        return {"available": False, "error": f"statsmodels not available: {e}"}

    s = pd.Series(series).dropna().astype(float)
    if len(s) < 20:
        return {"available": False, "error": "too few points for ADF (need >= 20)"}

    try:
        stat, pval, lags, nobs, crit, icbest = adfuller(s, autolag=autolag)
        return {
            "available": True,
            "statistic": float(stat),
            "pvalue": float(pval),
            "lags": int(lags),
            "nobs": int(nobs),
            "critical_1%": float(crit.get("1%")),
            "critical_5%": float(crit.get("5%")),
            "critical_10%": float(crit.get("10%")),
            "icbest": float(icbest),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def _grid_size(space: Dict[str, Sequence[Any]]) -> Optional[int]:
    """
    Compute the total number of grid combinations if space values are finite sequences.
    Returns None if any dimension is not a finite sized sequence.
    """
    try:
        total = 1
        for v in space.values():
            # Treat strings as atomic (len would count characters)
            if isinstance(v, str):
                return None
            n = len(v)  # may raise if not sized
            if n <= 0:
                return None
            total *= n
            if total > 1_000_000:  # guard against runaway sizes
                return None
        return total
    except Exception:
        return None


def extend_brief_with_analyses(
    df: pd.DataFrame,
    strategy_cls: Type,                       # e.g., SMACrossoverStrategy
    params_space: Dict[str, Sequence[Any]],   # grid or candidate sequences
    pip: float,
    *,
    when: str = "exit",
    n_splits: int = 4,
    n_trials: int = 80,
    objective: ObjectiveFn = obj_total_pips,
) -> Dict[str, Any]:
    """
    Produce additional analysis artifacts to merge into your AI brief:
      - Parameter search (grid if feasible, otherwise random sampling)
      - Walk-forward (auto-tune on each train fold -> evaluate next fold)
      - Stationarity (ADF) on close and on simple returns
      - Regime analysis: k-means regimes over features + performance by regime

    Returns a dict with keys:
      {
        'param_search': {'top': [...], 'best_params': {...}},
        'walk_forward': [...],
        'stationarity': {'close': {...}, 'returns': {...}},
        'regimes': <dict or {'error': '...'}>
      }
    """
    # ---------------------------
    # 1) Parameter search
    # ---------------------------
    # Decide grid vs random: if total grid size is reasonable, run full grid; else random
    total_grid = _grid_size(params_space)
    if total_grid is not None and total_grid <= 1_000:
        gs = grid_search(
            df=df,
            strategy_cls=strategy_cls,
            param_grid=params_space,
            pip=pip,
            objective=objective,
            when=when,
        )
    else:
        gs = random_search(
            df=df,
            strategy_cls=strategy_cls,
            param_grid=params_space,
            pip=pip,
            n_trials=n_trials,
            objective=objective,
            when=when,
        )

    top_records = top_k(gs, k=min(5, len(gs))).to_dict(orient="records")
    top = [{k: _py_scalar(v) for k, v in rec.items()} for rec in top_records]
    best = best_params(gs) if len(gs) else {}

    # ---------------------------
    # 2) Walk-forward
    # ---------------------------
    wf = walk_forward(
        df=df,
        strategy_cls=strategy_cls,
        param_grid=params_space,
        pip=pip,
        objective=objective,
        n_splits=n_splits,
        when=when,
    )
    wf_rows = wf.to_dict(orient="records") if hasattr(wf, "to_dict") else []
    for rec in wf_rows:
      if isinstance(rec.get("params"), dict):
        rec["params"] = {k: _py_scalar(v) for k, v in rec["params"].items()}

    # ---------------------------
    # 3) Stationarity
    # ---------------------------
    st_close = adf_test(df["close"])
    # Simple close-to-close returns; user can customize to log returns externally
    ret = df["close"].pct_change()
    st_ret = adf_test(ret)

    # ---------------------------
    # 4) Regimes + performance by regime
    # ---------------------------
    regimes_block: Dict[str, Any]
    try:
      feats = build_regime_features(df)
      labs = kmeans_regimes(feats, k=3)   # returns pd.Series or None
      if labs is None:
          regimes_block = {"available": False, "error": "kmeans unavailable (scikit-learn missing?)"}
      else:
          # Backtest with best params (if any) to get trades
          strat = strategy_cls(**best) if best else strategy_cls()
          trades = backtest(df, strat, pip=pip)

          # Build per-bar pips series from trades, then per-bar pips increments
          pips_curve = cumulative_pips_series(trades, df.index, when=when)  # cumulative pips
          pips_per_bar = pips_curve.diff().fillna(0.0)                      # per-bar pips increments

          # Performance by regime (pips)
          from .perf import perf_by_regime_pips
          perf_r = perf_by_regime_pips(pips_per_bar, labs)

          regimes_block = {
              "available": True,
              "k": 3,
              "performance_by_regime_pips": perf_r.to_dict(orient="records"),
          }
    except Exception as e:
      regimes_block = {"available": False, "error": str(e)}
    
    obj_name = getattr(objective, "__name__", type(objective).__name__)
    obj_doc  = (inspect.getdoc(objective) or "").strip()
    objective_meta = {"name": obj_name, "doc": obj_doc}

    return {
        "objective": objective_meta,
        "param_search": {"top": top, "best_params": best},
        "walk_forward": wf_rows,
        "stationarity": {"close": st_close, "returns": st_ret},
        "regimes": regimes_block,
    }
# -----------------------------
# Brief builder (pips-only)
# -----------------------------

def build_pips_brief(payload: StrategyRunPayload) -> Dict[str, Any]:
    """
    Create a compact, JSON-ready brief for LLM/ML consumption. Pips-only.
    """
    validate_trades(payload.trades)
    t = payload.trades.copy()

    # Normalize side for consistent downstream reads (does not mutate caller df)
    t["side_norm"] = normalize_side(t["side"])

    overall = {
        "n_trades": int(len(t)),
        "mean_pips": float(t["pips"].mean()) if len(t) else 0.0,
        "median_pips": float(t["pips"].median()) if len(t) else 0.0,
        "total_pips": float(t["pips"].sum()),
        "by_side_total_pips": {
            "long": float(t.loc[t["side_norm"] == "long", "pips"].sum()) if len(t) else 0.0,
            "short": float(t.loc[t["side_norm"] == "short", "pips"].sum()) if len(t) else 0.0,
        },
    }

    # Include any indicator snapshot columns that happen to be present
    snapshot_cols = [c for c in OPTIONAL_TRADE_COLS if c in t.columns]

    # Keep small sample for token control
    base_cols = ["entry_time", "exit_time", "side", "pips"]
    base_cols = [c for c in base_cols if c in t.columns]
    sample_cols = base_cols + snapshot_cols
    samples = t[sample_cols].head(25) if sample_cols else pd.DataFrame()

    brief: Dict[str, Any] = {
        "meta": asdict(payload.meta),
        "params": payload.params,
        "overall_pips": overall,
        "columns_present": list(t.columns),
        "pips_quantiles": t["pips"].quantile([.1,.25,.5,.75,.9]).round(2).to_dict() if len(t) else {},
        "samples": samples.to_dict(orient="records") if not samples.empty else [],
    }

    # Optional: include a tiny spark of the cumulative curve if bar_index is given
    if payload.bar_index is not None and len(payload.bar_index) > 0:
        try:
            cum = cumulative_pips_series(t, payload.bar_index, when="exit")
            # compress: take ~50 evenly spaced points to control tokens
            if len(cum) > 50:
                idx = (pd.Series(range(len(cum))) * (len(cum) - 1) / 49).round().astype(int).unique()
                cum_small = cum.iloc[idx].rename_axis("time").reset_index()
            else:
                cum_small = cum.rename_axis("time").reset_index()
            brief["cumulative_pips_preview"] = [
                {"time": str(row["time"]), "cum_pips": float(row[0])} for _, row in cum_small.iterrows()
            ]
        except Exception:
            # Don't fail the brief if cumulative calculation can't be constructed
            pass

    return brief


def annotate_trades_with_indicators(
    trades: pd.DataFrame,
    features: pd.DataFrame,
    at: str = "entry",
    cols: Iterable[str] | None = None,
    suffix: str | None = None,
) -> pd.DataFrame:
    if cols is None:
        cols = list(features.columns)
    ts_col = f"{at}_time"
    key = pd.to_datetime(trades[ts_col])
    feat = features.copy()
    feat.index = pd.to_datetime(feat.index)
    take = feat.reindex(key, method="nearest")
    add = take[cols].reset_index(drop=True)
    if suffix:
        add = add.add_suffix(suffix)
    out = trades.reset_index(drop=True).join(add)
    return out
