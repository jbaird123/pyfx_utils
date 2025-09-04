
from __future__ import annotations
import pandas as pd
import numpy as np

# ---- Simple moving averages ----
def sma(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length, min_periods=length).mean()

def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False, min_periods=length).mean()

# ---- RSI (Wilder) ----
def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

# ---- ATR (Wilder) ----
def _true_range(high, low, close):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

# ---- ADX (Wilder) ----
def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = _true_range(high, low, close)

    atr_w = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    plus_di  = 100 * (plus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean() / atr_w.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean() / atr_w.replace(0, np.nan))
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
    adx_val = dx.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

    return pd.DataFrame({
        f"+DI_{length}": plus_di,
        f"-DI_{length}": minus_di,
        f"ADX_{length}": adx_val
    })

# ---- Bollinger Bands ----
def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    basis = sma(close, length)
    dev = close.rolling(length, min_periods=length).std()
    upper = basis + mult * dev
    lower = basis - mult * dev
    width = (upper - lower) / basis
    pct_b = (close - lower) / (upper - lower)
    return pd.DataFrame({
        f"BB_basis_{length}": basis,
        f"BB_upper_{length}_{mult:g}": upper,
        f"BB_lower_{length}_{mult:g}": lower,
        f"BB_width_{length}_{mult:g}": width,
        f"BB_pctB_{length}_{mult:g}": pct_b
    })

# ---- Convenience: add a set of indicators to a DF ----
def add_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add indicators in-place based on a config dict.
    Example config:
      {
        "sma": [20, 50],
        "ema": [20],
        "rsi": [14],
        "atr": [14],
        "adx": [14],
        "bb":  [(20, 2.0)]
      }
    """
    out = df.copy()
    if "sma" in config:
        for n in config["sma"]:
            out[f"SMA_{n}"] = sma(out["close"], n)
    if "ema" in config:
        for n in config["ema"]:
            out[f"EMA_{n}"] = ema(out["close"], n)
    if "rsi" in config:
        for n in config["rsi"]:
            out[f"RSI_{n}"] = rsi(out["close"], n)
    if "atr" in config:
        for n in config["atr"]:
            out[f"ATR_{n}"] = atr(out["high"], out["low"], out["close"], n)
    if "adx" in config:
        for n in config["adx"]:
            adx_df = adx(out["high"], out["low"], out["close"], n)
            out = out.join(adx_df)
    if "bb" in config:
        for (n, mult) in config["bb"]:
            bb_df = bollinger(out["close"], n, mult)
            out = out.join(bb_df)
    return out
