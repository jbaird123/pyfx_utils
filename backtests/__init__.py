# Keep your trades-based backtest if present (optional)
try:
    from .core import backtest
except Exception:
    backtest = None

from .signal import BTConfig, backtest_signal

__all__ = ["backtest", "BTConfig", "backtest_signal"]
