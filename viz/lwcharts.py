
from __future__ import annotations
import pandas as pd
from lightweight_charts import JupyterChart
from ..utils.time import to_naive, snap_to_bars

# ---------- Core builders ----------

def build_ohlc(df: pd.DataFrame, time_col: str = None) -> pd.DataFrame:
    """Return a DataFrame ready for JupyterChart.set(...)
    with columns: time (string '%Y-%m-%d %H:%M:%S'), open, high, low, close, (optional volume).
    If time_col is None, the df index is used as the time.
    """
    if time_col is None:
        t = to_naive(df.index)
    else:
        t = to_naive(df[time_col])
    timestr = pd.to_datetime(t).strftime('%Y-%m-%d %H:%M:%S')
    out_cols = ["open","high","low","close"]
    out = pd.DataFrame({"time": timestr})
    for c in out_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' for OHLC.")
        out[c] = df[c].values
    if "volume" in df.columns:
        out["volume"] = df["volume"].values
    return out

def new_chart(width: int = 1400, height: int = 800, toolbox: bool = True) -> JupyterChart:
    """Create a JupyterChart with sensible defaults and a quiet legend."""
    chart = JupyterChart(width=width, height=height, toolbox=toolbox)
    # Keep legend for OHLC, hide it for per-trade lines to avoid clutter
    try:
        chart.legend(visible=True, ohlc=True, lines=False)
    except Exception:
        pass
    return chart

# ---------- Overlays & helpers ----------

def add_sma(chart: JupyterChart, df_like: pd.DataFrame, length: int = 20,
            source: str = "close", name: str | None = None,
            color: str = "#f39c12", width: int | None = None):
    """Add a Simple Moving Average overlay to chart. df_like must have 'time' or 'time_str'."""
    if "time" in df_like.columns:
        times = df_like["time"]
    elif "time_str" in df_like.columns:
        times = df_like["time_str"]
    else:
        raise ValueError("df_like must contain 'time' (string) or 'time_str'.")

    if pd.api.types.is_datetime64_any_dtype(times):
        times = pd.to_datetime(times).dt.strftime('%Y-%m-%d %H:%M:%S')

    if source not in df_like.columns:
        raise ValueError(f"Column '{source}' not found.")

    sma_vals = df_like[source].rolling(length, min_periods=length).mean()
    series_name = name or f"SMA{length}"
    payload = pd.DataFrame({"time": times, series_name: sma_vals}).dropna()

    line = chart.create_line(
        name=series_name,
        color=color,
        price_line=False,
        price_label=False
    )
    line.set(payload)

    if width is not None:
        try:
            line.apply_options({"lineWidth": int(width)})
        except Exception:
            pass
    return line

# ---------- Trades (lines + markers) ----------

def prepare_trades_for_chart(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of trades with snapped times and entry/exit values computed from df.
    Expects trades columns: side, entry_time, exit_time, (optional entry_price, exit_price), pips.
    """
    out = trades.copy()
    # normalize types
    for col in ["entry_time","exit_time"]:
        out[col] = to_naive(out[col])

    # build bar index from df for snapping
    bar_index = to_naive(df.index)
    out["entry_time_snapped"] = snap_to_bars(out["entry_time"], bar_index)
    out["exit_time_snapped"]  = snap_to_bars(out["exit_time"],  bar_index)

    out["entry_time_str"] = pd.to_datetime(out["entry_time_snapped"]).dt.strftime('%Y-%m-%d %H:%M:%S')
    out["exit_time_str"]  = pd.to_datetime(out["exit_time_snapped"]).dt.strftime('%Y-%m-%d %H:%M:%S')

    # restrict to keys that exist
    valid = out["entry_time_snapped"].notna() & out["exit_time_snapped"].notna()
    out = out[valid].copy()

    # fallback prices from df closes if not provided
    df_naive = df.copy()
    df_naive.index = to_naive(df_naive.index)

    def _price_or_close(row, which):
        col = f"{which}_price"
        if col in out.columns and pd.notna(row.get(col)):
            return float(row[col])
        snap_col = f"{which}_time_snapped"
        return float(df_naive.loc[row[snap_col], "close"])

    out["entry_val"] = out.apply(lambda r: _price_or_close(r, "entry"), axis=1)
    out["exit_val"]  = out.apply(lambda r: _price_or_close(r, "exit"), axis=1)
    return out

def add_trades(chart: JupyterChart, prepared_trades: pd.DataFrame,
               wins_color: str = "#26a65b", loss_color: str = "#ef5350",
               show_markers: bool = True):
    """Add one line per trade (entry->exit). Use prepare_trades_for_chart() first."""
    for i, r in prepared_trades.reset_index(drop=True).iterrows():
        side = str(r["side"]).upper()
        pips = float(r["pips"])
        color = wins_color if pips >= 0 else loss_color
        series_name = f"{side} {pips:+.0f}p"

        line = chart.create_line(
            name=series_name,
            color=color,
            price_line=False,
            price_label=False
        )
        line_df = pd.DataFrame({
            "time": [r["entry_time_str"], r["exit_time_str"]],
            series_name: [r["entry_val"], r["exit_val"]]
        })
        line.set(line_df)

        if show_markers:
            # put a single hoverable marker at midpoint (below bar to avoid clipping)
            t0 = pd.to_datetime(r["entry_time_str"])
            t1 = pd.to_datetime(r["exit_time_str"])
            mid = (t0 + (t1 - t0) / 2).strftime('%Y-%m-%d %H:%M:%S')
            tip = f"{side} {pips:+.0f}p\nEntry: {r['entry_val']:.5f}\nExit:  {r['exit_val']:.5f}"
            try:
                chart.marker(time=mid, position="belowBar", shape="circle", color=color, title=tip)
            except TypeError:
                chart.marker(time=mid, position="belowBar", shape="circle", color=color)

def add_y_padding(chart: JupyterChart, ohlc: pd.DataFrame, top: float = 0.18, bottom: float = 0.12):
    """Create invisible padding series to give the chart vertical margins (works on all builds)."""
    lo = float(ohlc["low"].min()); hi = float(ohlc["high"].max())
    rng = hi - lo if hi > lo else max(hi, 1.0) * 1e-6
    pad_lo = lo - rng * bottom; pad_hi = hi + rng * top
    t0, t1 = ohlc["time"].iloc[0], ohlc["time"].iloc[-1]

    for name, y in (("_pad_lo", pad_lo), ("_pad_hi", pad_hi)):
        s = chart.create_line(name=name, color="rgba(0,0,0,0)", price_line=False, price_label=False)
        s.set(pd.DataFrame({"time": [t0, t1], name: [y, y]}))
