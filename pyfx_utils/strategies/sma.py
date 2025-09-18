from dataclasses import dataclass
import pandas as pd
from pyfx_utils.analysis.interfaces import StrategyRunMeta, StrategyRunPayload

class Strategy:
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

@dataclass
class SMACrossoverStrategy(Strategy):
    fast: int = 10
    slow: int = 20
    
    # Stop-loss
    stop_type: str | None = None      # 'pips', 'atr', or None
    stop_value: float | None = None   # distance in pips (if 'pips') or ATR-mults (if 'atr')

    # Take-profit
    tp_type: str | None = None        # 'pips', 'atr', or None
    tp_value: float | None = None

    # Trailing stop
    trail_type: str | None = None     # 'pips', 'atr', or None
    trail_value: float | None = None
    trail_mode: str = "chandelier"    # 'chandelier' (since-entry high/low) or 'close_based'

    # ATR config (if you use 'atr' types)
    atr_length: int = 14

    # Exit tie-break when both stop & TP plausible inside same bar
    tie_break: str = "stop_then_tp"   # or 'tp_then_stop'

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        sma_fast = df["close"].rolling(self.fast, min_periods=self.fast).mean()
        sma_slow = df["close"].rolling(self.slow, min_periods=self.slow).mean()

        # Require valid values today AND yesterday to test a crossover
        valid = (
            sma_fast.notna() & sma_slow.notna() &
            sma_fast.shift(1).notna() & sma_slow.shift(1).notna()
        )

        # Cross UP: was fast <= slow yesterday AND fast > slow today
        cross_up = valid & (sma_fast.shift(1) <= sma_slow.shift(1)) & (sma_fast > sma_slow)

        # Cross DOWN: was fast >= slow yesterday AND fast < slow today
        cross_down = valid & (sma_fast.shift(1) >= sma_slow.shift(1)) & (sma_fast < sma_slow)

        out["entry_long"]  = cross_up
        out["exit_long"]   = cross_down
        out["entry_short"] = cross_down
        out["exit_short"]  = cross_up
        return out

@dataclass
class SMACrossoverAdapter:
    """
    Adapter that wraps SMA crossover trades into the generic analysis interface.
    Produces a StrategyRunPayload so downstream AI/ML can consume the results.
    """
    trades_df: pd.DataFrame
    params: dict
    meta: StrategyRunMeta
    bar_index: pd.DatetimeIndex | None = None

    def to_payload(self) -> StrategyRunPayload:
        return StrategyRunPayload(
            meta=self.meta,
            params=self.params,
            trades=self.trades_df,
            bar_index=self.bar_index,
        )

"""
How to use (examples)

Fixed distances in pips (no ATR):

strat = SMACrossoverStrategy(
    fast=10, slow=50,
    stop_type="pips", stop_value=100,   # 100 pips SL
    tp_type="pips",   tp_value=200,     # 200 pips TP
    trail_type=None
)


ATR-based stops/targets:

strat = SMACrossoverStrategy(
    fast=10, slow=50,
    stop_type="atr", stop_value=2.0,    # 2×ATR SL
    tp_type="atr",   tp_value=4.0,      # 4×ATR TP
    atr_length=14
)


Fixed trailing stop in pips:

strat = SMACrossoverStrategy(
    fast=10, slow=50,
    stop_type=None, tp_type=None,
    trail_type="pips", trail_value=150,    # trail by 150 pips
    trail_mode="chandelier",               # or "close_based"
)


Combine fixed SL + trailing SL + TP:

strat = SMACrossoverStrategy(
    fast=10, slow=50,
    stop_type="pips", stop_value=100,      # base SL
    trail_type="atr", trail_value=2.0,     # tighten with 2×ATR trail
    tp_type="pips",   tp_value=250,
    tie_break="stop_then_tp"
)

"""