from .signal import BTConfig, backtest_signal, equity_curve_from_signal, equity_curve_from_trades
from .metrics import metrics, metrics_by_period

# Optional: keep backtest if you truly have a trades-based engine in core.py
try:
    from .core import backtest
except Exception:
    backtest = None

__all__ = [
    "BTConfig", "backtest_signal", "equity_curve_from_signal", "equity_curve_from_trades",
    "metrics", "metrics_by_period",
    "backtest",
]
