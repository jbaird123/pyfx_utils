# pyfx_utils/indicators/features.py
from __future__ import annotations
import pandas as pd
import numpy as np

# Use ONLY the external 'ta' library
try:
    from ta.momentum import RSIIndicator
    from ta.trend import ADXIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
except Exception as e:
    raise ImportError(
        "The 'ta' package is required for add_indicators(). "
        "Install it with: pip install ta"
    ) from e


def add_indicators(
    df: pd.DataFrame,
    *,
    pip: float,
    rsi_len: int = 14,
    adx_len: int = 14,
    bb_len: int = 20,
    bb_k: float = 2.0,
    atr_len: int = 14,
    include: tuple[str, ...] = ("rsi", "adx", "bb_width", "atr_pips"),
) -> pd.DataFrame:
    """
    Minimal wrapper around the external 'ta' library that adds exactly the
    columns your pipeline expects:

      - 'rsi'       (RSIIndicator)
      - 'adx'       (ADXIndicator)
      - 'bb_width'  = (upper - lower) / middle (BollingerBands)
      - 'atr_pips'  = ATR / pip (AverageTrueRange)

    Parameters
    ----------
    df : DataFrame with columns ['open','high','low','close', ...]
    pip : float  pip size for symbol, e.g. 0.0001 for EURUSD
    rsi_len, adx_len, bb_len, bb_k, atr_len : indicator settings
    include : which canonical columns to emit

    Returns
    -------
    DataFrame: original df with added indicator columns.
    """
    out = df.copy()

    # Sanity checks
    for col in ("high", "low", "close"):
        if col not in out.columns:
            raise KeyError(f"add_indicators() requires column '{col}' in df")

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    if "rsi" in include:
        out["rsi"] = RSIIndicator(close=close, window=rsi_len).rsi()

    if "adx" in include:
        adx = ADXIndicator(high=high, low=low, close=close, window=adx_len)
        out["adx"] = adx.adx()

    if "bb_width" in include:
        bb = BollingerBands(close=close, window=bb_len, window_dev=bb_k)
        mid = bb.bollinger_mavg()
        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()
        out["bb_width"] = (upper - lower) / mid

    if "atr_pips" in include:
        if not pip:
            raise ValueError("add_indicators(...): 'pip' must be provided and non-zero to compute 'atr_pips'.")
        atr = AverageTrueRange(high=high, low=low, close=close, window=atr_len).average_true_range()
        out["atr_pips"] = atr / float(pip)

    return out
