# pyfx_utils/indicators/features.py
from __future__ import annotations
import numpy as np
import pandas as pd
from .ta import sma, ema, rsi, atr, adx as _adx, bollinger as _bb

# ---------- extra indicators (pure pandas) ----------
def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    fast_ = ema(close, fast); slow_ = ema(close, slow)
    macd_ = fast_ - slow_
    sig  = ema(macd_, signal)
    hist = macd_ - sig
    return pd.DataFrame({f"MACD_{fast}_{slow}": macd_, f"MACDsig_{signal}": sig, "MACDhist": hist})

def stoch(high, low, close, k=14, d=3) -> pd.DataFrame:
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    pct_k = 100 * (close - ll) / (hh - ll)
    pct_d = pct_k.rolling(d, min_periods=d).mean()
    return pd.DataFrame({f"StochK_{k}": pct_k, f"StochD_{d}": pct_d})

def cci(high, low, close, length=20, c=0.015) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(length, min_periods=length).mean()
    md = (tp - ma).abs().rolling(length, min_periods=length).mean()
    return (tp - ma) / (c * md)

def williams_r(high, low, close, length=14) -> pd.Series:
    ll = low.rolling(length, min_periods=length).min()
    hh = high.rolling(length, min_periods=length).max()
    return -100 * (hh - close) / (hh - ll)

def donchian(high, low, length=20) -> pd.DataFrame:
    up = high.rolling(length, min_periods=length).max()
    dn = low.rolling(length, min_periods=length).min()
    mid = (up + dn) / 2
    width = (up - dn) / mid
    return pd.DataFrame({f"DonchUp_{length}": up, f"DonchDn_{length}": dn, f"DonchMid_{length}": mid, f"DonchW_{length}": width})

def keltner(high, low, close, ema_len=20, atr_len=10, mult=2.0) -> pd.DataFrame:
    mid = ema(close, ema_len)
    rng = atr(high, low, close, atr_len)
    up = mid + mult * rng
    dn = mid - mult * rng
    return pd.DataFrame({f"KeltMid_{ema_len}": mid, f"KeltUp_{ema_len}_{atr_len}_{mult:g}": up, f"KeltDn_{ema_len}_{atr_len}_{mult:g}": dn})

def roc(close, length=10) -> pd.Series:
    return close.pct_change(length)

def zscore(s: pd.Series, window=20) -> pd.Series:
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std()
    return (s - m) / sd

def percent_rank(s: pd.Series, window=20) -> pd.Series:
    # rank of last value within window
    return s.rolling(window, min_periods=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

def logret(close, periods=1) -> pd.Series:
    return np.log(close / close.shift(periods))

def rolling_sharpe(returns: pd.Series, window=30, ann_factor: float | None = None) -> pd.Series:
    mu = returns.rolling(window, min_periods=window).mean()
    sd = returns.rolling(window, min_periods=window).std()
    sh = mu / sd
    if ann_factor:
        sh = sh * np.sqrt(ann_factor)
    return sh

# ---------- presets ----------
_PRESETS = {
    # sensible daily-FX starter pack
    "core_fx": {
        "sma": [20, 50, 200],
        "ema": [21],
        "rsi": [14],
        "atr": [14],
        "adx": [14],
        "bb":  [(20, 2.0)],
        "macd": [(12, 26, 9)],
        "stoch": [(14, 3)],
        "donchian": [20],
        "keltner": [(20, 10, 2.0)],
        "roc": [5, 20],
        "zscore": [20],
        "percent_rank": [20],
    }
}

# ---------- main builder ----------
def add_indicators(
    df: pd.DataFrame,
    config: dict | None = None,
    preset: str | None = "core_fx",
    lags: list[int] | tuple[int, ...] | None = None,
    targets: list[int] | tuple[int, ...] | None = None,
    ann_factor: float | None = None,
    dropna: bool = False,
) -> pd.DataFrame:
    """
    Build a wide feature set efficiently (batch concat to avoid fragmentation).
    """
    base_cols = list(df.columns)
    frames = [df]                 # we’ll concat once at the end
    added_names: list[str] = []   # track feature columns (for lags)

    # --- merge preset + config (preset first; config overrides) ---
    cfg = {}
    if preset:
        cfg.update(_PRESETS.get(preset, {}))
    if config:
        for k, v in config.items():
            cfg[k] = v

    # --- primitives (collect frames, don’t assign columns one-by-one) ---
    tmp = []

    if "sma" in cfg:
        for n in cfg["sma"]:
            s = sma(df["close"], n).rename(f"SMA_{n}")
            tmp.append(s)

    if "ema" in cfg:
        for n in cfg["ema"]:
            s = ema(df["close"], n).rename(f"EMA_{n}")
            tmp.append(s)

    if "rsi" in cfg:
        for n in cfg["rsi"]:
            s = rsi(df["close"], n).rename(f"RSI_{n}")
            tmp.append(s)

    if "atr" in cfg:
        for n in cfg["atr"]:
            s = atr(df["high"], df["low"], df["close"], n).rename(f"ATR_{n}")
            tmp.append(s)

    if "adx" in cfg:
        for n in cfg["adx"]:
            adx_df = _adx(df["high"], df["low"], df["close"], n)
            tmp.append(adx_df)

    if "bb" in cfg:
        for (n, mult) in cfg["bb"]:
            bb_df = _bb(df["close"], n, mult)
            tmp.append(bb_df)

    # extras
    if "macd" in cfg:
        for (f, s, sig) in cfg["macd"]:
            tmp.append(macd(df["close"], f, s, sig))

    if "stoch" in cfg:
        for (k, d) in cfg["stoch"]:
            tmp.append(stoch(df["high"], df["low"], df["close"], k, d))

    if "donchian" in cfg:
        for n in cfg["donchian"]:
            tmp.append(donchian(df["high"], df["low"], n))

    if "keltner" in cfg:
        for (ema_len, atr_len, mult) in cfg["keltner"]:
            tmp.append(keltner(df["high"], df["low"], df["close"], ema_len, atr_len, mult))

    if "roc" in cfg:
        for n in cfg["roc"]:
            tmp.append(roc(df["close"], n).rename(f"ROC_{n}"))

    if "zscore" in cfg:
        for w in cfg["zscore"]:
            tmp.append(zscore(df["close"], w).rename(f"Z_{w}"))

    if "percent_rank" in cfg:
        for w in cfg["percent_rank"]:
            tmp.append(percent_rank(df["close"], w).rename(f"PctRank_{w}"))

    if tmp:
        feats = pd.concat(tmp, axis=1)
        frames.append(feats)
        added_names.extend(feats.columns.tolist())

    # base returns + rolling Sharpe
    base_tmp = []
    base_tmp.append(df["close"].pct_change().rename("ret1"))
    base_tmp.append((np.log(df["close"] / df["close"].shift(1))).rename("logret1"))
    sh = rolling_sharpe(base_tmp[-1], window=30, ann_factor=ann_factor).rename("Sharpe_30")
    base_tmp.append(sh)
    base_block = pd.concat(base_tmp, axis=1)
    frames.append(base_block)
    added_names.extend(base_block.columns.tolist())

    # --- lags (batch per L, not column-by-column) ---
    if lags:
        lag_blocks = []
        # features are everything except the original OHLCV base columns
        # (we also include returns/sharpe)
        # Ensure we use the names we tracked:
        feat_cols = added_names.copy()
        for L in lags:
            lag_df = pd.concat(frames[1:], axis=1)  # concat features we added so far
            lag_df = lag_df[feat_cols].shift(L)
            lag_df.columns = [f"{c}_lag{L}" for c in feat_cols]
            lag_blocks.append(lag_df)
        if lag_blocks:
            frames.append(pd.concat(lag_blocks, axis=1))

    # --- targets (batch build) ---
    if targets:
        targ_blocks = []
        for h in targets:
            fr = pd.DataFrame({
                f"fwd_ret_{h}": df["close"].pct_change(-h)
            })
            fr[f"label_up_{h}"] = (fr[f"fwd_ret_{h}"] > 0).astype("int8")
            targ_blocks.append(fr)
        frames.append(pd.concat(targ_blocks, axis=1))

    # --- final concat (single allocation), optional dropna, defragment copy ---
    out = pd.concat(frames, axis=1)
    if dropna:
        out = out.dropna()
    return out.copy()

