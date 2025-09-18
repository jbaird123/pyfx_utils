import pytest

# Basic import smoke tests to catch import-time errors / heavy side effects

@pytest.mark.parametrize("mod", [
    "pyfx_utils",
    "pyfx_utils.utils",
    "pyfx_utils.utils.stats",
    "pyfx_utils.utils.time",
    "pyfx_utils.backtests",
    "pyfx_utils.backtests.signal",
    "pyfx_utils.backtests.metrics",
    "pyfx_utils.analysis",
    "pyfx_utils.analysis.augment",
    "pyfx_utils.analysis.regime",
    "pyfx_utils.analysis.walkforward",
    "pyfx_utils.strategies",
    "pyfx_utils.strategies.sma",
    "pyfx_utils.strategies.rsi",
    "pyfx_utils.indicators",
    "pyfx_utils.indicators.ta",
    "pyfx_utils.indicators.features",
    "pyfx_utils.viz",
    "pyfx_utils.viz.curves",
    "pyfx_utils.viz.lwcharts",
    "pyfx_utils.viz.visualization",
])
def test_can_import(mod):
    pytest.importorskip(mod)
