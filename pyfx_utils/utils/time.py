
from __future__ import annotations
import pandas as pd
import numpy as np

def to_naive(x):
    """Return timezone-naive datetimes from Series or DatetimeIndex.
    Keeps the UTC *clock time* but removes tz info, so merging with tz-naive charts is easy.
    """
    y = pd.to_datetime(x, errors="coerce", utc=True)
    if isinstance(y, pd.Series):
        return y.dt.tz_localize(None)
    else:
        # DatetimeIndex / DatetimeArray
        return y.tz_localize(None)

def snap_to_bars(ts: pd.Series, bar_index: pd.DatetimeIndex, tolerance="18h") -> pd.Series:
    """Snap a series of timestamps to nearest bar in bar_index within tolerance.
    Returns a Series of snapped datetimes (tz-naive).
    """
    t = to_naive(ts)
    bars_dt = pd.Series(bar_index).dropna().drop_duplicates().sort_values().rename("bar_time")
    s = pd.DataFrame({"t": t}).sort_values("t")
    merged = pd.merge_asof(
        s, bars_dt.to_frame(), left_on="t", right_on="bar_time",
        tolerance=pd.Timedelta(tolerance), direction="nearest"
    )
    merged.index = s.index
    return merged["bar_time"]

def infer_bars_per_year(index: pd.DatetimeIndex, *, market: str = "fx") -> float:
    """
    Estimate bars/year from a time index.
    - Intraday: use market-specific trading hours/week
      - fx:     24/5
      - equities (RTH): 6.5h/day * 252d/yr
      - crypto: 24/7
    - Daily or higher: return 252 if weekends mostly excluded, else 365.
    """
    if len(index) < 2:
        return 252.0

    # Robust spacing (seconds) via median diff
    deltas = pd.Series(index).diff().dropna()
    sec = deltas.dt.total_seconds().median()

    if not np.isfinite(sec) or sec <= 0:
        return 252.0

    # If >= ~12h between bars, treat as daily-or-higher
    if sec >= 12 * 3600:
        # Heuristic: inspect presence of weekend dates in the index
        weekdays = pd.Index(index.weekday)  # 0=Mon ... 6=Sun
        has_weekend = (weekdays == 5).any() or (weekdays == 6).any()
        return 365.0 if has_weekend else 252.0

    # Intraday: compute bars/year from trading seconds per year
    if market == "equities":
        # Regular Trading Hours (US): 6.5h/day * 252d/yr
        trading_seconds_per_year = 6.5 * 3600 * 252
    elif market == "crypto":
        # 24/7
        trading_seconds_per_year = 365.25 * 24 * 3600
    else:
        # Default FX: 24 hours/day, 5 days/week
        trading_seconds_per_year = 52 * 5 * 24 * 3600

    return float(trading_seconds_per_year / sec)
