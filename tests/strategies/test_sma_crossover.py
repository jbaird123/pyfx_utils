import numpy as np
import pandas as pd
import pytest

from pyfx_utils.strategies.sma import SMACrossoverStrategy


def _mk_ohlc(n=60, seed=0):
    rng = np.random.RandomState(seed)
    closes = 100 + rng.randn(n).cumsum()
    highs = closes + rng.rand(n)
    lows = closes - rng.rand(n)
    opens = np.r_[closes[0], closes[:-1]]
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


def test_defaults_all_fields():
    s = SMACrossoverStrategy()
    assert s.fast == 10
    assert s.slow == 20
    assert s.stop_type is None
    assert s.stop_value is None
    assert s.tp_type is None
    assert s.tp_value is None
    assert s.trail_type is None
    assert s.trail_value is None
    assert s.trail_mode == "chandelier"
    assert s.atr_length == 14
    assert s.tie_break == "stop_then_tp"


def test_override_all_fields_persists():
    s = SMACrossoverStrategy(
        fast=7,
        slow=19,
        stop_type="pips",
        stop_value=25.0,
        tp_type="atr",
        tp_value=3.0,
        trail_type="atr",
        trail_value=2.5,
        trail_mode="close_based",
        atr_length=21,
        tie_break="tp_then_stop",
    )
    assert s.fast == 7
    assert s.slow == 19
    assert s.stop_type == "pips"
    assert s.stop_value == 25.0
    assert s.tp_type == "atr"
    assert s.tp_value == 3.0
    assert s.trail_type == "atr"
    assert s.trail_value == 2.5
    assert s.trail_mode == "close_based"
    assert s.atr_length == 21
    assert s.tie_break == "tp_then_stop"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stop_type": "pips", "stop_value": 20.0},
        {"stop_type": "atr", "stop_value": 2.0, "atr_length": 10},
        {"tp_type": "pips", "tp_value": 30.0},
        {"tp_type": "atr", "tp_value": 3.0, "atr_length": 18},
        {"trail_type": "pips", "trail_value": 15.0, "trail_mode": "chandelier"},
        {"trail_type": "atr", "trail_value": 2.5, "trail_mode": "close_based", "atr_length": 21},
        {"tie_break": "tp_then_stop"},
    ],
)
def test_generate_signals_invariant_to_non_signal_params(kwargs):
    df = _mk_ohlc(80, seed=4)
    base = SMACrossoverStrategy(fast=5, slow=12).generate_signals(df)
    alt = SMACrossoverStrategy(fast=5, slow=12, **kwargs).generate_signals(df)
    pd.testing.assert_frame_equal(alt, base)


def test_generate_signals_shape_and_types():
    df = _mk_ohlc(40, seed=42)
    strat = SMACrossoverStrategy(fast=3, slow=5)
    out = strat.generate_signals(df)

    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(df.index)
    for col in ("entry_long", "exit_long", "entry_short", "exit_short"):
        assert col in out.columns
        assert out[col].dtype == bool


def test_generate_signals_crossover_logic_matches_definition():
    df = _mk_ohlc(120, seed=1)
    fast, slow = 5, 12
    strat = SMACrossoverStrategy(fast=fast, slow=slow)
    out = strat.generate_signals(df)

    sma_fast = df["close"].rolling(fast, min_periods=fast).mean()
    sma_slow = df["close"].rolling(slow, min_periods=slow).mean()
    valid = (
        sma_fast.notna() & sma_slow.notna()
        & sma_fast.shift(1).notna() & sma_slow.shift(1).notna()
    )

    cross_up = (valid & (sma_fast.shift(1) <= sma_slow.shift(1)) & (sma_fast > sma_slow)).fillna(False).rename("entry_long")
    cross_dn = (valid & (sma_fast.shift(1) >= sma_slow.shift(1)) & (sma_fast < sma_slow)).fillna(False).rename("exit_long")  # name adjusted later per column

    pd.testing.assert_series_equal(out["entry_long"], cross_up)

    # For the remaining columns, use the proper names explicitly
    pd.testing.assert_series_equal(out["exit_short"],  cross_up.rename("exit_short"))
    pd.testing.assert_series_equal(out["entry_short"], cross_dn.rename("entry_short"))
    pd.testing.assert_series_equal(out["exit_long"],   cross_dn.rename("exit_long"))

    assert not (cross_up.astype(bool) & cross_dn.astype(bool)).any()
    assert (out.loc[~valid.fillna(False)].sum(axis=1) == 0).all()


def test_generate_signals_no_signals_before_min_periods():
    df = _mk_ohlc(10, seed=7)
    strat = SMACrossoverStrategy(fast=3, slow=15)
    out = strat.generate_signals(df)
    assert (out.sum(axis=1) == 0).all()
