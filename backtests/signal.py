from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class BTConfig:
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    signal_lag: int = 1
    price_col: str = "close"

def backtest_signal(df: pd.DataFrame, signal: pd.Series, cfg: BTConfig) -> pd.DataFrame:
    px = df[cfg.price_col].astype(float)
    sig = signal.shift(cfg.signal_lag).fillna(0.0)
    ret = px.pct_change().fillna(0.0)
    gross = sig * ret
    turns = (sig != sig.shift()).astype(int).fillna(0)
    cost = turns * (cfg.fee_bps + cfg.slippage_bps) / 10000.0
    pnl = gross - cost
    eq = (1.0 + pnl).cumprod()
    return pd.DataFrame({"ret": pnl, "eq": eq, "sig": sig}, index=df.index)
