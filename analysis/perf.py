
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Iterable

def infer_bars_per_year(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 252
    dt = (index[1] - index[0]).total_seconds()
    if dt <= 60:      # 1-min
        return int(252 * 6.5 * 60)
    if dt <= 300:     # 5-min
        return int(252 * 6.5 * 12)
    if dt <= 3600:    # 1-hour
        return int(252 * 6.5)
    if dt <= 86400:   # daily
        return 252
    return 252

def _turnover(sig: pd.Series) -> pd.Series:
    return (sig != sig.shift()).astype(int).fillna(0)

def equity_curve_from_signal(
    df: pd.DataFrame,
    signal: pd.Series,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    signal_lag: int = 1,
    price_col: str = "close",
) -> pd.DataFrame:
    px = df[price_col].astype(float)
    sig = signal.shift(signal_lag).fillna(0.0)
    ret = px.pct_change().fillna(0.0)
    gross = sig * ret
    turns = _turnover(sig)
    cost = turns * (fee_bps + slippage_bps) / 10000.0
    pnl = gross - cost
    eq = (1.0 + pnl).cumprod()
    return pd.DataFrame({"ret": pnl, "eq": eq, "sig": sig}, index=df.index)

def equity_curve_from_trades(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    price_col: str = "close",
    long_label: str = "long",
    short_label: str = "short",
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    idx = df.index
    pos = pd.Series(0.0, index=idx)
    for _, r in trades.iterrows():
        side = str(r.get("side", "")).lower()
        t0 = r["entry_time"]; t1 = r["exit_time"]
        if pd.isna(t0) or pd.isna(t1):
            continue
        try:
            start = idx.get_indexer([t0], method="nearest")[0]
            end   = idx.get_indexer([t1], method="nearest")[0]
        except Exception:
            continue
        if end < start:
            start, end = end, start
        val = 1.0 if long_label in side else (-1.0 if short_label in side else 0.0)
        if val != 0.0:
            pos.iloc[start:end+1] = val
    return equity_curve_from_signal(df, pos, fee_bps=fee_bps, slippage_bps=slippage_bps, signal_lag=0, price_col=price_col)

def metrics(pnl: pd.Series, index: pd.DatetimeIndex) -> Dict[str, float]:
    pnl = pnl.fillna(0.0).astype(float)
    bpy = infer_bars_per_year(index)
    mu = pnl.mean()
    sd = pnl.std(ddof=0)
    ann_ret = (1.0 + mu) ** bpy - 1.0
    ann_vol = sd * np.sqrt(bpy)
    sharpe = ann_ret / (ann_vol + 1e-12)
    downside = pnl.clip(upper=0.0)
    dd = (1.0 + downside).var(ddof=0) ** 0.5 * np.sqrt(bpy)
    sortino = ann_ret / (dd + 1e-12)
    eq = (1.0 + pnl).cumprod()
    peaks = eq.cummax()
    mdd = ((eq / peaks) - 1.0).min()
    calmar = ann_ret / (abs(mdd) + 1e-12)
    return {
        "AnnReturn": float(ann_ret),
        "AnnVol": float(ann_vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "MaxDD": float(mdd),
        "Calmar": float(calmar),
    }

def metrics_by_period(pnl: pd.Series, freq: str = "A") -> pd.DataFrame:
    grp = pnl.fillna(0.0).groupby(pd.Grouper(freq=freq))
    out = []
    for g, s in grp:
        if s.empty:
            continue
        eq_end = float((1.0 + s).prod() - 1.0)
        ann = (1.0 + s.mean()) ** len(s) - 1.0
        vol = s.std(ddof=0) * np.sqrt(len(s))
        sharpe_like = ann / (vol + 1e-12)
        out.append({"period": g, "Return": eq_end, "Sharpe_like": sharpe_like})
    return pd.DataFrame(out).set_index("period") if out else pd.DataFrame(columns=["Return","Sharpe_like"])

def trade_pnls(pnl: pd.Series, sig: pd.Series) -> list[float]:
    tp = []
    run = 0.0
    had_pos = False
    for r, s in zip(pnl.values, sig.values):
        if s != 0:
            run += r
            had_pos = True
        else:
            if had_pos:
                tp.append(run)
                run = 0.0
                had_pos = False
    if had_pos:
        tp.append(run)
    return tp


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

import pandas as pd
import numpy as np

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
