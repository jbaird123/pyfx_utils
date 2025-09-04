from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

# ----------------------------
# Public config
# ----------------------------
@dataclass
class BTConfig:
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    signal_lag: int = 1
    price_col: str = "close"

# ----------------------------
# Internal helpers (single source of truth)
# ----------------------------
def _prep_signal(signal: pd.Series, lag: int) -> pd.Series:
    """Lag the signal and fill initial NA with flat (0)."""
    return signal.shift(lag).fillna(0.0)

def _turnover(sig: pd.Series) -> pd.Series:
    """1 on bars where position changes, else 0."""
    return (sig != sig.shift()).astype(int).fillna(0)

def _equity_from_signal(px: pd.Series, sig: pd.Series, fee_bps: float, slippage_bps: float) -> pd.DataFrame:
    """
    The single canonical computation:
    - ret = px.pct_change()
    - gross pnl = sig * ret
    - cost = turnover * (fee + slippage) in decimal
    - pnl = gross - cost
    - eq = (1 + pnl).cumprod()
    Returns DataFrame[ret(eq bar pnl), eq, sig]
    """
    px = px.astype(float)
    ret = px.pct_change().fillna(0.0)
    gross = sig * ret
    turns = _turnover(sig)
    # Turn costs from bps to decimal and apply only on turnover bars
    cost = turns * (fee_bps + slippage_bps) / 10000.0
    pnl = gross - cost
    eq = (1.0 + pnl).cumprod()
    return pd.DataFrame({"ret": pnl, "eq": eq, "sig": sig}, index=px.index)

# ----------------------------
# Public API (thin wrappers)
# ----------------------------
def backtest_signal(df: pd.DataFrame, signal: pd.Series, cfg: BTConfig) -> pd.DataFrame:
    """
    Primary engine: given a signal Series aligned to df.index, computes pnl/eq.
    This is the *single source of truth* for equity math.
    """
    sig = _prep_signal(signal, cfg.signal_lag)
    px = df[cfg.price_col]
    return _equity_from_signal(px, sig, cfg.fee_bps, cfg.slippage_bps)

def equity_curve_from_signal(
    df: pd.DataFrame,
    signal: pd.Series,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    signal_lag: int = 1,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Convenience wrapper that delegates to backtest_signal (no duplicate math).
    """
    cfg = BTConfig(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        signal_lag=signal_lag,
        price_col=price_col,
    )
    return backtest_signal(df, signal, cfg)

def equity_curve_from_trades(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    price_col: str = "close",
    long_label: str = "long",
    short_label: str = "short",
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Builds a position series from a trade ledger, then reuses the *same* equity math.
    Note: we apply no lag here (entries/exits already encode timing); signal_lag=0.
    """
    idx = df.index
    pos = pd.Series(0.0, index=idx)

    # Map trade intervals to index positions, then paint positions.
    # (Kept close to your original logic; still O(N trades).)
    for _, r in trades.iterrows():
        side = str(r.get("side", "")).lower()
        t0, t1 = r.get("entry_time"), r.get("exit_time")
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

    # Delegate to the same core pipeline (no duplicate math)
    cfg = BTConfig(fee_bps=fee_bps, slippage_bps=slippage_bps, signal_lag=0, price_col=price_col)
    return backtest_signal(df, pos, cfg)
