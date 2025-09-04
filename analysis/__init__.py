
from .perf import (
    infer_bars_per_year, metrics, equity_curve_from_signal, equity_curve_from_trades,
    trade_pnls, metrics_by_period, annotate_trades_with_indicators,perf_by_regime_pips
)
from .walkforward import walk_forward_ranges

from .augment import extend_brief_with_analyses, adf_test

__all__ = [
    "infer_bars_per_year", "metrics", "equity_curve_from_signal", "equity_curve_from_trades",
    "trade_pnls", "metrics_by_period", "annotate_trades_with_indicators",
    "walk_forward_ranges",
    "extend_brief_with_analyses",
    "adf_test","perf_by_regime_pips"
]
