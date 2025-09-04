
from __future__ import annotations
import pandas as pd

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