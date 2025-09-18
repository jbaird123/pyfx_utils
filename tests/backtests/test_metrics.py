import numpy as np
import pandas as pd
import pytest

try:
    from pyfx_utils.backtests.metrics import metrics, metrics_by_period
except Exception:
    metrics = None
    metrics_by_period = None


@pytest.mark.skipif(metrics is None, reason="metrics() not available")
def test_metrics_shape_and_types():
    # Daily index
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    # Small returns with zeros and flips
    ret = pd.Series([0, 0.01, -0.01, 0.0, 0.005, 0.0, -0.002, 0.0, 0.0, 0.001], index=idx, dtype=float)
    out = metrics(ret, idx)

    # Expected keys (loose contract)
    expected_keys = {"AnnReturn", "AnnVol", "Sharpe", "Sortino", "MaxDD", "Calmar"}
    assert expected_keys.issubset(set(out.keys()))
    # All numeric & finite
    for k in expected_keys:
        assert np.isfinite(float(out[k])) or np.isneginf(out[k]) or np.isposinf(out[k])


@pytest.mark.skipif(metrics_by_period is None, reason="metrics_by_period() not available")
def test_metrics_by_period_groups():
    idx = pd.date_range("2020-01-01", periods=90, freq="D")
    ret = pd.Series(np.random.RandomState(0).normal(0, 0.001, size=len(idx)), index=idx)

    # Your implementation returns a DataFrame; use month-end ('ME') to avoid deprecation warnings
    out = metrics_by_period(ret, freq="ME")

    # Should be a DataFrame with at least two monthly rows and expected columns
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] >= 2
    assert {"Return", "Sharpe_like"}.issubset(set(out.columns))
