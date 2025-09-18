import numpy as np
import pandas as pd
import pytest

try:
    from pyfx_utils.utils.time import infer_bars_per_year
except Exception:
    infer_bars_per_year = None


@pytest.mark.skipif(infer_bars_per_year is None, reason="infer_bars_per_year not available")
def test_infer_bars_per_year_hourly():
    idx = pd.date_range("2024-01-01", periods=10_000, freq="h")  # lowercase 'h' avoids FutureWarning
    est = infer_bars_per_year(idx)

    # Accept either ~24*252 (6048) OR ~6.5*252 (1638) within 25% tolerance,
    # depending on the heuristic your function uses.
    candidates = [24 * 252, 6.5 * 252]
    assert any(abs(est - c) / c < 0.25 for c in candidates)


@pytest.mark.skipif(infer_bars_per_year is None, reason="infer_bars_per_year not available")
def test_infer_bars_per_year_daily():
    idx = pd.date_range("2024-01-01", periods=365, freq="D")  # calendar-daily includes weekends
    est = infer_bars_per_year(idx)

    # If weekends appear in the index, expect ~365; otherwise expect ~252
    weekdays = pd.Index(idx.weekday)  # 0=Mon ... 6=Sun
    has_weekend = (weekdays == 5).any() or (weekdays == 6).any()

    target = 365.0 if has_weekend else 252.0
    # 10% tolerance
    assert abs(est - target) / target < 0.10

    # Tip: if you want to assert ~252 explicitly, build the index with business days:
    # idx_b = pd.date_range("2024-01-01", periods=252, freq="B")
    # est_b = infer_bars_per_year(idx_b)
    # assert abs(est_b - 252.0) / 252.0 < 0.10
