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
