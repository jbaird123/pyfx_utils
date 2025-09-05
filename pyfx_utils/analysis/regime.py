from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd 
import numpy as np
try:
    from sklearn.cluster import KMeans  # optional
except Exception:
    KMeans = None


def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight regime features from prices.
    Expects 'close' column present. Generates:
      - vol_pips_20: rolling std of close-to-close pips (20)
      - trend_50_200: EMA50-EMA200 in price units
      - bb_width_20: (upper-lower)/middle from Bollinger(20,2)
    """
    out = pd.DataFrame(index=df.index)
    if 'close' not in df.columns:
        return out
    # Approximate pips from close diffs (user may replace with their own precise pips)
    # For a robust notebook, you can pass a prepared pips_per_bar separately; here we just build a feature.
    close = df['close'].astype(float)
    pip_factor = 10000.0  # override per instrument if needed
    pips = close.diff().fillna(0) * pip_factor
    out['vol_pips_20'] = pips.rolling(20).std()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    out['trend_50_200'] = (ema50 - ema200)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2*std
    lower = mid - 2*std
    out['bb_width_20'] = (upper - lower) / (mid.replace(0, np.nan))
    return out.dropna()


def kmeans_regimes(feats: pd.DataFrame, k: int = 3, seed: int = 42) -> pd.Series:
    if feats.empty or KMeans is None:
        return pd.Series(index=feats.index, dtype='float64')
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labs = km.fit_predict(feats.values)
    return pd.Series(labs, index=feats.index, name='regime')


def perf_by_regime(ret_pips: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    if ret_pips.empty or regimes.empty:
        return pd.DataFrame()
    df = pd.concat({'ret_pips': ret_pips, 'regime': regimes}, axis=1).dropna()
    grp = df.groupby('regime')['ret_pips']
    return pd.DataFrame({
        'total_pips': grp.sum(),
        'mean_pips': grp.mean(),
        'n_bars': grp.size(),
    }).sort_index()

def perf_by_regime(pnl: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """
    Compute performance metrics grouped by regime labels.

    Parameters
    ----------
    pnl : pd.Series
        Per-bar returns (e.g. pct changes or strategy returns).
        Index should be aligned with labels.
    labels : pd.Series
        Cluster/regime labels (e.g. from kmeans_regimes).
        Index should align with pnl.

    Returns
    -------
    pd.DataFrame
        One row per regime with:
        - regime (label value)
        - bars (# of samples)
        - total_return ((1+ret).prod()-1)
        - mean (average per-bar return)
        - std (stdev per-bar return)
        - sharpe_like (mean/std, unannualized)
    """
    s = pd.Series(pnl).dropna()
    labs = pd.Series(labels).reindex(s.index).dropna()
    s = s.reindex(labs.index)

    rows = []
    for k, grp in s.groupby(labs):
        rr = (1.0 + grp).prod() - 1.0
        mu = grp.mean()
        sd = grp.std(ddof=0)
        sharpe = mu / (sd + 1e-12)
        rows.append({
            "regime": k,
            "bars": int(grp.size),
            "total_return": float(rr),
            "mean": float(mu),
            "std": float(sd),
            "sharpe_like": float(sharpe),
        })

    return pd.DataFrame(rows).sort_values("total_return", ascending=False).reset_index(drop=True)

def perf_by_regime_pips(pips_per_bar: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """
    Group performance by regime, using *pips per bar* (additive) rather than multiplicative returns.
    Returns a table with total/mean/std pips per regime.

    Parameters
    ----------
    pips_per_bar : pd.Series (float)
        Per-bar pips increments (e.g., diff of a cumulative pips curve).
    labels : pd.Series
        Regime labels indexed by the same DatetimeIndex.

    Returns
    -------
    pd.DataFrame with columns:
      - regime
      - bars
      - total_pips
      - mean_pips
      - std_pips
    """
    s = pd.Series(pips_per_bar).astype(float)
    labs = pd.Series(labels).reindex(s.index)

    mask = labs.notna()
    s = s[mask]
    labs = labs[mask]

    rows = []
    for reg, grp in s.groupby(labs):
        rows.append({
            "regime": reg,
            "bars": int(grp.size),
            "total_pips": float(grp.sum()),
            "mean_pips": float(grp.mean()),
            "std_pips": float(grp.std(ddof=0)),
        })
    return pd.DataFrame(rows).sort_values("total_pips", ascending=False).reset_index(drop=True)

def evaluate_regime_filter(
    trades: pd.DataFrame,
    regime_series: pd.Series,
    active_regime: int,
    *,
    regimes_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate 'trade only when regime == active_regime' vs. unfiltered.
    Aligns regime labels to trade entry times via forward-fill reindex.

    Returns:
      {
        'k': <clusters if available>,
        'active_regime': int,
        'filtered_total_pips': float,
        'unfiltered_total_pips': float,
        'delta': float,
        'kept_trades': int,
        'total_trades': int
      }
    """
    out = {
        "k": int(regimes_meta.get("k", 0)) if isinstance(regimes_meta, dict) else 0,
        "active_regime": int(active_regime),
        "filtered_total_pips": 0.0,
        "unfiltered_total_pips": 0.0,
        "delta": 0.0,
        "kept_trades": 0,
        "total_trades": int(len(trades) if trades is not None else 0),
    }
    if trades is None or trades.empty or regime_series is None or regime_series.empty:
        return out

    # align labels at each entry_time
    entry_times = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    labs = regime_series.reindex(entry_times, method="ffill")

    mask = (labs == active_regime)
    filtered_total = float(trades.loc[mask.fillna(False), "pips"].sum())
    unfiltered_total = float(trades["pips"].sum())

    out.update({
        "filtered_total_pips": filtered_total,
        "unfiltered_total_pips": unfiltered_total,
        "delta": float(filtered_total - unfiltered_total),
        "kept_trades": int(mask.fillna(False).sum()),
    })
    return out
