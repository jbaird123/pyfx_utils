# tests/utils/test_stats.py
import pandas as pd
import numpy as np
import pytest

try:
    from pyfx_utils.utils.stats import cumulative_pips as cumulative_pips
except Exception:
    cumulative_pips = None

try:
    from pyfx_utils.utils.stats import trades_summary
except Exception:
    trades_summary = None


@pytest.mark.skipif(cumulative_pips is None, reason="cumulative_pips not available")
def test_cumulative_pips_exit_and_entry():
    # Build a simple bar index
    idx = pd.date_range("2020-01-01", periods=6, freq="D")

    # Two trades:
    #  - Trade1: +10 pips, entry t1, exit t3
    #  - Trade2: -5  pips, entry t4, exit t5
    trades = pd.DataFrame(
        {
            "entry_time": [idx[1], idx[4]],
            "exit_time":  [idx[3], idx[5]],
            "pips":       [10.0, -5.0],
        }
    )

    # Book at EXIT
    exit_curve = cumulative_pips(trades, idx, when="exit")
    # Increments at t3: +10, at t5: -5
    expected_exit_incr = pd.Series(0.0, index=idx)
    expected_exit_incr.iloc[3] = 10.0
    expected_exit_incr.iloc[5] = -5.0
    expected_exit = expected_exit_incr.cumsum().rename("pips")  # function returns name 'pips'
    pd.testing.assert_series_equal(exit_curve, expected_exit)

    # Book at ENTRY
    entry_curve = cumulative_pips(trades, idx, when="entry")
    # Increments at t1: +10, at t4: -5
    expected_entry_incr = pd.Series(0.0, index=idx)
    expected_entry_incr.iloc[1] = 10.0
    expected_entry_incr.iloc[4] = -5.0
    expected_entry = expected_entry_incr.cumsum().rename("pips")
    pd.testing.assert_series_equal(entry_curve, expected_entry)


@pytest.mark.skipif(trades_summary is None, reason="trades_summary not available")
def test_trades_summary_keys_and_consistency():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    trades = pd.DataFrame(
        {
            "entry_time": [idx[0], idx[1], idx[2]],
            "exit_time":  [idx[1], idx[2], idx[3]],
            "pips":       [10.0, -4.0, 2.0],
        }
    )
    summ = trades_summary(trades)

    # Flexible key names to match your implementation
    # count / n_trades
    n = summ.get("n_trades", summ.get("count"))
    if n is not None:
        assert int(n) == 3

    # total / sum
    total = summ.get("total_pips", summ.get("sum_pips", summ.get("pips_sum")))
    if total is not None:
        assert abs(float(total) - 8.0) < 1e-9

    # mean / avg
    mean_val = summ.get("mean_pips", summ.get("avg_pips"))
    if mean_val is not None:
        assert abs(float(mean_val) - (8.0 / 3.0)) < 1e-9

    # median should exist and be correct
    assert "median_pips" in summ
    assert abs(float(summ["median_pips"]) - 2.0) < 1e-9

    # Optional sanity: max/min if present
    if "max_win" in summ:
        assert abs(float(summ["max_win"]) - 10.0) < 1e-9
    if "max_loss" in summ:
        assert abs(float(summ["max_loss"]) - (-4.0)) < 1e-9
