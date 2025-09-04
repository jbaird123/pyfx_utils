import pandas as pd
import numpy as np
import pytest

from pyfx_utils.backtests.signal import (
    BTConfig,
    _prep_signal,
    _turnover,
    _equity_from_signal,
    backtest_signal,
    equity_curve_from_signal,
    equity_curve_from_trades,
)

# ---------- helpers ----------

def mk_df(prices, freq="D"):
    idx = pd.date_range("2020-01-01", periods=len(prices), freq=freq)
    return pd.DataFrame({"close": np.asarray(prices, dtype=float)}, index=idx)

def pct_change(a):
    a = np.asarray(a, dtype=float)
    out = np.zeros_like(a, dtype=float)
    out[1:] = (a[1:] - a[:-1]) / a[:-1]
    return out

# ---------- BTConfig ----------

def test_btconfig_defaults():
    cfg = BTConfig()
    assert cfg.fee_bps == 0.0
    assert cfg.slippage_bps == 0.0
    assert cfg.signal_lag == 1
    assert cfg.price_col == "close"

# ---------- _prep_signal ----------

def test_prep_signal_lag_and_fill():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    sig = pd.Series([0, 1, 1, 0], index=idx, dtype=float)

    out0 = _prep_signal(sig, 0)
    pd.testing.assert_series_equal(out0, sig)

    out1 = _prep_signal(sig, 1)
    # shifted right by 1, with leading 0
    exp = pd.Series([0.0, 0.0, 1.0, 1.0], index=idx, dtype=float)
    pd.testing.assert_series_equal(out1, exp)

# ---------- _turnover ----------

def test_turnover_counts_changes():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    sig = pd.Series([0, 1, 1, -1, -1, 0], index=idx, dtype=float)
    # changes occur at 1 (0->1), 3 (1->-1), 5 (-1->0)
    expected = pd.Series([0, 1, 0, 1, 0, 1], index=idx, dtype=int)
    turns = _turnover(sig)
    pd.testing.assert_series_equal(turns.astype(int), expected)

# ---------- _equity_from_signal ----------

def test_equity_from_signal_math_no_costs():
    prices = [100, 101, 100, 102]
    df = mk_df(prices)
    idx = df.index
    # long from t1..t2, flat at ends: after lag is handled in wrappers;
    # here we pass a *final* signal
    sig = pd.Series([0.0, 1.0, 1.0, 0.0], index=idx)
    ret = pct_change(prices)  # [0, +1/100, -1/101, +2/100]
    gross = sig * ret
    # no costs
    pnl = gross
    eq = (1.0 + pnl).cumprod()

    got = _equity_from_signal(df["close"], sig, fee_bps=0.0, slippage_bps=0.0)
    # ret column stores pnl-per-bar
    np.testing.assert_allclose(got["ret"].values, pnl, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["eq"].values, eq, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(got["sig"], sig)

def test_equity_from_signal_applies_costs_on_turnover_only():
    prices = [100, 101, 102, 103]
    df = mk_df(prices)
    idx = df.index
    sig = pd.Series([0.0, 1.0, 1.0, 0.0], index=idx)

    # two turnovers: 0->1 at t1, 1->0 at t3
    fee_bps = 10.0  # 0.10%
    slippage_bps = 5.0  # 0.05%
    per_turn_cost = (fee_bps + slippage_bps) / 10000.0  # 0.0015

    got = _equity_from_signal(df["close"], sig, fee_bps, slippage_bps)
    # costs should reduce pnl only on turnover bars (t1 and t3)
    turns = _turnover(sig).astype(bool).values
    # sanity check: turnovers at bars 1 and 3
    assert turns.tolist() == [False, True, False, True]
    # compute a reference manually
    ret = pct_change(prices)
    gross = sig.values * ret
    cost = np.zeros_like(gross)
    cost[turns] = per_turn_cost
    pnl = gross - cost
    eq = (1.0 + pnl).cumprod()
    np.testing.assert_allclose(got["ret"].values, pnl, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["eq"].values, eq, rtol=1e-12, atol=1e-12)

# ---------- backtest_signal ----------

def test_backtest_signal_delegates_and_lags():
    prices = [100, 101, 100, 102]
    df = mk_df(prices)
    idx = df.index
    raw_sig = pd.Series([0.0, 1.0, 1.0, 0.0], index=idx)

    # With lag=1, the effective signal is shifted right by 1
    cfg = BTConfig(fee_bps=0.0, slippage_bps=0.0, signal_lag=1, price_col="close")
    got = backtest_signal(df, raw_sig, cfg)

    effective = _prep_signal(raw_sig, 1)
    ref = _equity_from_signal(df["close"], effective, fee_bps=0.0, slippage_bps=0.0)
    pd.testing.assert_frame_equal(got, ref)

# ---------- equity_curve_from_signal ----------

def test_equity_curve_from_signal_matches_backtest_signal():
    prices = [100, 101, 103, 102]
    df = mk_df(prices)
    idx = df.index
    sig = pd.Series([0.0, 1.0, 0.0, -1.0], index=idx)

    got = equity_curve_from_signal(df, sig, fee_bps=7.0, slippage_bps=3.0, signal_lag=1, price_col="close")
    cfg = BTConfig(fee_bps=7.0, slippage_bps=3.0, signal_lag=1, price_col="close")
    ref = backtest_signal(df, sig, cfg)
    pd.testing.assert_frame_equal(got, ref)

# ---------- equity_curve_from_trades ----------

def test_equity_curve_from_trades_equivalence_to_signal():
    # prices monotonic up, then down, to make sign effects visible
    prices = [100, 101, 99, 100, 102, 101]
    df = mk_df(prices)
    idx = df.index

    # build trades: long from t1..t2, flat, short from t4..t5
    trades = pd.DataFrame(
        {
            "side": ["long", "short"],
            "entry_time": [idx[1], idx[4]],
            "exit_time":  [idx[2], idx[5]],
        }
    )

    # Expected position series painted over the same intervals:
    pos = pd.Series(0.0, index=idx)
    pos.iloc[1:2+1] = 1.0     # t1..t2 inclusive
    pos.iloc[4:5+1] = -1.0    # t4..t5 inclusive

    # Using trades API (which internally builds pos and calls backtest_signal)
    got = equity_curve_from_trades(
        df, trades, price_col="close",
        long_label="long", short_label="short",
        fee_bps=0.0, slippage_bps=0.0,
    )
    # Directly via the signal API with lag=0
    cfg = BTConfig(fee_bps=0.0, slippage_bps=0.0, signal_lag=0, price_col="close")
    ref = backtest_signal(df, pos, cfg)
    pd.testing.assert_frame_equal(got, ref)

def test_equity_curve_from_trades_ignores_invalid_rows():
    prices = [100, 101, 102]
    df = mk_df(prices)
    idx = df.index

    trades = pd.DataFrame(
        {
            "side": ["long", "short", "long"],
            "entry_time": [idx[0], pd.NaT, idx[1]],
            "exit_time":  [idx[1], idx[2], pd.NaT],
        }
    )

    # Only the first row is valid (long from t0..t1)
    got = equity_curve_from_trades(df, trades, fee_bps=0.0, slippage_bps=0.0)
    # Build the expected pos and compare
    pos = pd.Series(0.0, index=idx)
    pos.iloc[0:1+1] = 1.0
    cfg = BTConfig(signal_lag=0)
    ref = backtest_signal(df, pos, cfg)
    pd.testing.assert_frame_equal(got, ref)
