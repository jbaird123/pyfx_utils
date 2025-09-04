# /MyDrive/code/pyfx_utils/backtests/core.py
import numpy as np
import pandas as pd
from typing import Protocol,Optional

class StrategyProtocol(Protocol):
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame: ...
    # Optional attrs read by execution:
    # stop_type: 'pips'|'atr'|None
    # stop_value: float|None
    # tp_type: 'pips'|'atr'|None
    # tp_value: float|None
    # trail_type: 'pips'|'atr'|None
    # trail_value: float|None
    # trail_mode: 'chandelier'|'close_based'
    # atr_length: int
    # tie_break: 'stop_then_tp'|'tp_then_stop'

def _atr_wilder(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder's ATR in price units (requires high/low/close)."""
    high = df["high"].astype(float).to_numpy()
    low  = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([
        np.abs(high - low),
        np.abs(high - prev_close),
        np.abs(low  - prev_close)
    ])
    atr = pd.Series(tr).ewm(alpha=1.0/float(n), adjust=False).mean()
    return pd.Series(atr, index=df.index)

def _fixed_price_from_dist(side: int, entry_price: float, dist_price: float, is_stop: bool) -> float:
    """Convert a signed distance in *price units* into an absolute stop/tp price."""
    if side == 1:   # long
        return entry_price - dist_price if is_stop else entry_price + dist_price
    else:           # short
        return entry_price + dist_price if is_stop else entry_price - dist_price

def _dist_price(
    side: int,
    kind: Optional[str],
    value: Optional[float],
    pip: float,
    atr_val: Optional[float],
) -> float:   # may return np.nan
    """
    Return a positive distance in *price units* for the chosen kind:
      - 'pips':  value * pip
      - 'atr':   value * atr_val
      - None:    np.nan
    """
    if kind is None or value is None:
        return np.nan
    if kind == "pips":
        return float(value) * float(pip)
    if kind == "atr":
        if atr_val is None or np.isnan(atr_val):
            return np.nan
        return float(value) * float(atr_val)
    # unknown kind → disabled
    return np.nan

def backtest(data: pd.DataFrame, strategy: StrategyProtocol, pip: float) -> pd.DataFrame:
    """
    Single-position engine (flat → long/short → flat) with:
      - Fixed stop/TP: distance in pips or ATR multiples
      - Trailing stop: pips or ATR; 'chandelier' (since-entry H/L) or 'close_based'
      - Intrabar check using H/L; tie-break when both stop & TP are plausible
      - Signal-based exits at close when no protective exits hit
    Returns columns: side, entry_time, entry_price, exit_time, exit_price, pips
    """
    df = data.copy()
    sig = strategy.generate_signals(df)

    # Arrays for speed/clarity
    el = sig["entry_long"].to_numpy(bool)
    es = sig["entry_short"].to_numpy(bool)
    xl = sig["exit_long"].to_numpy(bool)
    xs = sig["exit_short"].to_numpy(bool)

    close = df["close"].astype(float).to_numpy()
    high  = df["high"].astype(float).to_numpy() if "high" in df else close
    low   = df["low"].astype(float).to_numpy()  if "low"  in df else close
    idx   = df.index.to_numpy()

    # Execution params
    stop_type   = getattr(strategy, "stop_type", None)
    stop_value  = getattr(strategy, "stop_value", None)
    tp_type     = getattr(strategy, "tp_type", None)
    tp_value    = getattr(strategy, "tp_value", None)
    trail_type  = getattr(strategy, "trail_type", None)
    trail_value = getattr(strategy, "trail_value", None)
    trail_mode  = str(getattr(strategy, "trail_mode", "chandelier"))
    tie_break   = str(getattr(strategy, "tie_break", "stop_then_tp"))
    atr_len     = int(getattr(strategy, "atr_length", 14))

    use_stop  = (stop_type is not None and stop_value is not None)
    use_tp    = (tp_type   is not None and tp_value   is not None)
    use_trail = (trail_type is not None and trail_value is not None)

    if (use_stop or use_tp or use_trail) and not {"high","low","close"}.issubset(df.columns):
        raise ValueError("Protective exits require high/low/close columns.")

    atr = _atr_wilder(df, atr_len).to_numpy() if ((stop_type == "atr") or (tp_type == "atr") or (trail_type == "atr")) else None

    # State
    state = 0       # 0=flat, 1=long, -1=short
    ei = -1
    ep = np.nan
    stop_px = np.nan       # fixed stop from entry
    tp_px   = np.nan       # fixed tp from entry
    trail_px = np.nan      # trailing stop
    hh_since_entry = -np.inf
    ll_since_entry =  np.inf
    out = []

    def close_trade(exit_i: int, price: float):
        nonlocal state, ei, ep, stop_px, tp_px, trail_px, hh_since_entry, ll_since_entry
        side = "long" if state == 1 else "short"
        pips_val = (price - ep)/pip if state == 1 else (ep - price)/pip
        out.append({
            "side": side,
            "entry_time": pd.Timestamp(idx[ei]),
            "entry_price": float(ep),
            "exit_time": pd.Timestamp(idx[exit_i]),
            "exit_price": float(price),
            "pips": float(pips_val),
        })
        state = 0
        ei = -1
        ep = stop_px = tp_px = trail_px = np.nan
        hh_since_entry = -np.inf
        ll_since_entry =  np.inf

    n = len(df)
    for i in range(n):
        # 1) Update trailing stop if in a position
        if state != 0 and use_trail:
            base_dist = _dist_price(state, trail_type, trail_value, pip, atr[i] if atr is not None else None)
            if not np.isnan(base_dist):
                if trail_mode == "chandelier":
                    if state == 1:  # long
                        hh_since_entry = max(hh_since_entry, high[i])
                        candidate = hh_since_entry - base_dist
                    else:           # short
                        ll_since_entry = min(ll_since_entry, low[i])
                        candidate = ll_since_entry + base_dist
                else:  # 'close_based'
                    candidate = (close[i] - base_dist) if state == 1 else (close[i] + base_dist)
                # Never loosen the trailing stop (tighten only)
                if np.isnan(trail_px):
                    trail_px = candidate
                else:
                    if state == 1:
                        trail_px = max(trail_px, candidate)
                    else:
                        trail_px = min(trail_px, candidate)

        # 2) Exits first when in a position
        if state != 0:
            # Effective stop = tightest of fixed stop and trailing stop
            eff_stop = np.nan
            if state == 1:   # long
                candidates = [x for x in (stop_px, trail_px) if not np.isnan(x)]
                eff_stop = max(candidates) if candidates else np.nan
                hit_stop = (not np.isnan(eff_stop)) and (low[i] <= eff_stop)
                hit_tp   = (not np.isnan(tp_px))     and (high[i] >= tp_px)
            else:            # short
                candidates = [x for x in (stop_px, trail_px) if not np.isnan(x)]
                eff_stop = min(candidates) if candidates else np.nan
                hit_stop = (not np.isnan(eff_stop)) and (high[i] >= eff_stop)
                hit_tp   = (not np.isnan(tp_px))     and (low[i]  <= tp_px)

            if hit_stop or hit_tp:
                # deterministic tie-break
                if tie_break == "stop_then_tp":
                    price = eff_stop if hit_stop else tp_px
                else:  # 'tp_then_stop'
                    price = tp_px if hit_tp else eff_stop
                close_trade(i, float(price))
                continue

            # Signal exits if no protective exit hit
            if (state == 1 and xl[i]) or (state == -1 and xs[i]):
                close_trade(i, close[i])
                continue

        # 3) Entries when flat
        if state == 0:
            if el[i] or es[i]:
                state = 1 if el[i] else -1
                ei = i
                ep = close[i]
                # initialize since-entry extremes
                hh_since_entry = high[i]
                ll_since_entry = low[i]
                trail_px = np.nan

                # Fixed stop/tp at entry (if configured)
                dist_stop = _dist_price(state, stop_type, stop_value, pip, atr[i] if atr is not None else None)
                dist_tp   = _dist_price(state, tp_type,   tp_value,   pip, atr[i] if atr is not None else None)
                stop_px = _fixed_price_from_dist(state, ep, dist_stop, is_stop=True)  if not np.isnan(dist_stop) else np.nan
                tp_px   = _fixed_price_from_dist(state, ep, dist_tp,   is_stop=False) if not np.isnan(dist_tp)   else np.nan

    return pd.DataFrame(out, columns=["side","entry_time","entry_price","exit_time","exit_price","pips"])
