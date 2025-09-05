import numpy as np
import pandas as pd
import pytest

from pyfx_utils.strategies.rsi import rsi, RSIThresholdStrategy


# ---------- helpers ----------

def _mk_ohlc(n=200, seed=0):
    rng = np.random.RandomState(seed)
    closes = 100 + rng.randn(n).cumsum()
    highs = closes + rng.rand(n)
    lows = closes - rng.rand(n)
    opens = np.r_[closes[0], closes[:-1]]
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)


# ---------- rsi() tests ----------

def test_rsi_has_nan_warmup_and_bounds():
    s = pd.Series(np.arange(1.0, 101.0))
    out = rsi(s, length=14)
    assert out.iloc[:13].isna().all()
    assert not out.iloc[13:].isna().all()
    valid = out.dropna()
    assert ((valid >= 0) & (valid <= 100)).all()


def test_rsi_uptrend_near_hundred_and_downtrend_near_zero():
    n = 200
    up = pd.Series(np.arange(1.0, n + 1.0))
    down = pd.Series(np.arange(float(n), 0.0, -1.0))

    r_up = rsi(up, length=14)
    r_dn = rsi(down, length=14)

    assert r_up.iloc[-1] > 90
    assert r_dn.iloc[-1] < 10


def test_rsi_constant_all_nan_after_warmup():
    s = pd.Series(np.ones(50))
    out = rsi(s, length=14)
    assert out.iloc[14:].isna().all()


# ---------- RSIThresholdStrategy tests ----------

def test_rsi_strategy_defaults():
    strat = RSIThresholdStrategy()
    assert strat.length == 14
    assert strat.buy_below == 30.0
    assert strat.sell_above == 70.0


def test_rsi_strategy_shape_and_types():
    df = _mk_ohlc(120, seed=42)
    strat = RSIThresholdStrategy(length=10, buy_below=30, sell_above=70)
    out = strat.generate_signals(df)

    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(df.index)
    for col in ("entry_long", "exit_long", "entry_short", "exit_short"):
        assert col in out.columns
        assert out[col].dtype == bool


def test_rsi_strategy_exact_cross_logic_matches_definition():
    df = _mk_ohlc(200, seed=5)
    length = 14
    buy_below = 30.0
    sell_above = 70.0

    strat = RSIThresholdStrategy(length=length, buy_below=buy_below, sell_above=sell_above)
    out = strat.generate_signals(df)

    r = rsi(df["close"], length)
    long_entry  = ((r > buy_below) & (r.shift(1) <= buy_below)).fillna(False).rename("entry_long")
    long_exit   = ((r < sell_above) & (r.shift(1) >= sell_above)).fillna(False).rename("exit_long")
    short_entry = ((r < sell_above) & (r.shift(1) >= sell_above)).fillna(False).rename("entry_short")
    short_exit  = ((r > buy_below) & (r.shift(1) <= buy_below)).fillna(False).rename("exit_short")

    pd.testing.assert_series_equal(out["entry_long"],  long_entry)
    pd.testing.assert_series_equal(out["exit_long"],   long_exit)
    pd.testing.assert_series_equal(out["entry_short"], short_entry)
    pd.testing.assert_series_equal(out["exit_short"],  short_exit)

    assert not (out["entry_long"] & out["entry_short"]).any()


def test_rsi_strategy_no_signals_before_warmup():
    df = _mk_ohlc(10, seed=7)
    strat = RSIThresholdStrategy(length=15, buy_below=30, sell_above=70)
    out = strat.generate_signals(df)
    assert (out.sum(axis=1) == 0).all()


def test_rsi_thresholds_affect_signal_counts():
    df = _mk_ohlc(400, seed=11)

    base = RSIThresholdStrategy(length=14, buy_below=30.0, sell_above=70.0).generate_signals(df)
    tight = RSIThresholdStrategy(length=14, buy_below=40.0, sell_above=60.0).generate_signals(df)

    base_count = int(base[["entry_long", "entry_short"]].sum().sum())
    tight_count = int(tight[["entry_long", "entry_short"]].sum().sum())
    assert tight_count >= base_count
