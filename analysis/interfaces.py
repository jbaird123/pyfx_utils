"""
interfaces.py — pips-only strategy output interface (LLM/ML ready)

Usage:
  from interfaces import (
      StrategyRunMeta, StrategyRunPayload, validate_trades,
      build_pips_brief, normalize_side, REQUIRED_TRADE_COLS, OPTIONAL_TRADE_COLS,
      StrategyResultProvider
  )

Design:
  - Universal, pips-only schema for a strategy "run"
  - Minimal required trades columns
  - Dataclasses for metadata + payload
  - Validators + side normalization
  - Brief builder for LLM/ML (JSON-ready)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Literal, Optional, Protocol
import pandas as pd

# -----------------------------
# Core schema (pips-only)
# -----------------------------

Side = Literal["long", "short", 1, -1]

REQUIRED_TRADE_COLS: List[str] = [
    "entry_time", "exit_time", "side",
    "entry_price", "exit_price", "pips",
]

# Optional but recommended columns we will consume if present.
# All are scalars per trade (no money metrics). Add freely as needed.
OPTIONAL_TRADE_COLS: List[str] = [
    "symbol", "setup_id", "signal_name",
    "atr@entry", "rsi@entry", "adx@entry", "bb_width@entry",
    "mfe_pips", "mae_pips"
]

@dataclass
class StrategyRunMeta:
    strategy_name: str
    timeframe: str                      # e.g., "M15", "H1", "D1"
    universe: List[str] | str           # list of symbols or single symbol string
    version: str                        # your strategy/runner version (e.g., "1.2.3")
    run_id: str                         # unique id for this run (uuid or hash ok)
    timestamp: str                      # ISO8601 UTC string

@dataclass
class StrategyRunPayload:
    meta: StrategyRunMeta
    params: Dict[str, Any]
    trades: pd.DataFrame                # must satisfy REQUIRED_TRADE_COLS schema
    bar_index: Optional[pd.DatetimeIndex] = None  # optional, for per-bar cumulative pips

# -----------------------------
# Utilities & Validators
# -----------------------------

def normalize_side(series: pd.Series) -> pd.Series:
    """
    Normalize 'side' to {'long','short'} while accepting {1,-1} or strings.
    """
    def _norm(x):
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("long", "buy", "+1", "1"):
                return "long"
            if s in ("short", "sell", "-1"):
                return "short"
        if x in (1, +1):
            return "long"
        if x in (-1, -1):
            return "short"
        raise ValueError(f"Unrecognized side value: {x!r}")
    return series.map(_norm)

def validate_trades(df: pd.DataFrame) -> None:
    """
    Ensure required columns & basic dtypes are present for pips-only analysis.
    """
    missing = [c for c in REQUIRED_TRADE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Trades missing required columns: {missing}")

    # datetime checks
    if not pd.api.types.is_datetime64_any_dtype(df["entry_time"]):
        raise TypeError("entry_time must be datetime64")
    if not pd.api.types.is_datetime64_any_dtype(df["exit_time"]):
        raise TypeError("exit_time must be datetime64")

    # side normalization hint (accept 'long'/'short' or +/-1)
    try:
        _ = normalize_side(df["side"])
    except Exception as e:
        raise ValueError("side must be 'long'/'short' or 1/-1") from e

    # pips numeric
    if not pd.api.types.is_numeric_dtype(df["pips"]):
        raise TypeError("pips must be numeric (float preferred)")

def cumulative_pips_series(trades: pd.DataFrame, index: pd.DatetimeIndex, when: str = "exit") -> pd.Series:
    """
    Build a per-bar cumulative pips series from a trade ledger.
    - 'when' is 'exit' or 'entry' to choose when to book the pips.
    """
    assert when in ("exit", "entry")
    tcol = f"{when}_time"
    pips_per_bar = pd.Series(0.0, index=index, dtype="float64")
    if trades.empty:
        return pips_per_bar
    increments = trades.groupby(tcol)["pips"].sum()
    increments = increments.reindex(index, fill_value=0.0)
    return increments.cumsum()

# -----------------------------
# Brief builder (pips-only)
# -----------------------------

def build_pips_brief(payload: StrategyRunPayload) -> Dict[str, Any]:
    """
    Create a compact, JSON-ready brief for LLM/ML consumption. Pips-only.
    """
    validate_trades(payload.trades)
    t = payload.trades.copy()

    # Normalize side for consistent downstream reads (does not mutate caller df)
    t["side_norm"] = normalize_side(t["side"])

    overall = {
        "n_trades": int(len(t)),
        "mean_pips": float(t["pips"].mean()) if len(t) else 0.0,
        "median_pips": float(t["pips"].median()) if len(t) else 0.0,
        "total_pips": float(t["pips"].sum()),
        "by_side_total_pips": {
            "long": float(t.loc[t["side_norm"] == "long", "pips"].sum()) if len(t) else 0.0,
            "short": float(t.loc[t["side_norm"] == "short", "pips"].sum()) if len(t) else 0.0,
        },
    }

    # Include any indicator snapshot columns that happen to be present
    snapshot_cols = [c for c in OPTIONAL_TRADE_COLS if c in t.columns]

    # Keep small sample for token control
    base_cols = ["entry_time", "exit_time", "side", "pips"]
    base_cols = [c for c in base_cols if c in t.columns]
    sample_cols = base_cols + snapshot_cols
    samples = t[sample_cols].head(25) if sample_cols else pd.DataFrame()

    brief: Dict[str, Any] = {
        "meta": asdict(payload.meta),
        "params": payload.params,
        "overall_pips": overall,
        "columns_present": list(t.columns),
        "pips_quantiles": t["pips"].quantile([.1,.25,.5,.75,.9]).round(2).to_dict() if len(t) else {},
        "samples": samples.to_dict(orient="records") if not samples.empty else [],
    }

    # Optional: include a tiny spark of the cumulative curve if bar_index is given
    if payload.bar_index is not None and len(payload.bar_index) > 0:
        try:
            cum = cumulative_pips_series(t, payload.bar_index, when="exit")
            # compress: take ~50 evenly spaced points to control tokens
            if len(cum) > 50:
                idx = (pd.Series(range(len(cum))) * (len(cum) - 1) / 49).round().astype(int).unique()
                cum_small = cum.iloc[idx].rename_axis("time").reset_index()
            else:
                cum_small = cum.rename_axis("time").reset_index()
            brief["cumulative_pips_preview"] = [
                {"time": str(row["time"]), "cum_pips": float(row[0])} for _, row in cum_small.iterrows()
            ]
        except Exception:
            # Don't fail the brief if cumulative calculation can't be constructed
            pass

    return brief

# -----------------------------
# Adapter protocol (plug any strategy runner)
# -----------------------------

class StrategyResultProvider(Protocol):
    def to_payload(self) -> StrategyRunPayload:
        ...

__all__ = [
    "StrategyRunMeta",
    "StrategyRunPayload",
    "REQUIRED_TRADE_COLS",
    "OPTIONAL_TRADE_COLS",
    "validate_trades",
    "normalize_side",
    "cumulative_pips_series",
    "build_pips_brief",
    "StrategyResultProvider",
]
