import inspect
import numpy as np
import pandas as pd
import pytest

def _mk_ohlc(n=50, seed=0):
    rng = np.random.RandomState(seed)
    closes = 100 + rng.randn(n).cumsum()
    highs = closes + rng.rand(n)
    lows = closes - rng.rand(n)
    opens = np.r_[closes[0], closes[:-1]]
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)

@pytest.mark.parametrize("modname", [
    "pyfx_utils.indicators.ta",
    "pyfx_utils.indicators.features",
])
def test_indicator_callables_align_with_input(modname):
    mod = pytest.importorskip(modname)
    df = _mk_ohlc()

    candidates = []
    for name in dir(mod):
        if name.startswith("_"):
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        params = list(sig.parameters.values())
        if not params:
            continue
        p0 = params[0]
        if p0.kind in (p0.POSITIONAL_ONLY, p0.POSITIONAL_OR_KEYWORD):
            candidates.append((name, fn))

    if not candidates:
        pytest.skip(f"No indicator-like callables discovered in {modname}")

    saw_aligned = False
    for name, fn in candidates:
        try:
            out = fn(df)  # try with DF only; many indicators default window params
        except Exception:
            continue
        if isinstance(out, pd.Series):
            saw_aligned = True
            assert len(out) == len(df)
            assert out.index.equals(df.index)
        elif isinstance(out, pd.DataFrame):
            saw_aligned = True
            assert len(out) == len(df)
            assert out.index.equals(df.index)

    if not saw_aligned:
        pytest.skip(f"No indicator in {modname} accepted df-only and returned aligned output")
