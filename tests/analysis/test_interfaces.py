# tests/analysis/test_interfaces.py
import pandas as pd
import pytest

mod = pytest.importorskip("pyfx_utils.analysis.interfaces")

REQUIRED_TRADE_COLS = getattr(mod, "REQUIRED_TRADE_COLS", None)
validate_trades = getattr(mod, "validate_trades", None)
normalize_side = getattr(mod, "normalize_side", None)


@pytest.mark.skipif(REQUIRED_TRADE_COLS is None or validate_trades is None,
                    reason="validate_trades/REQUIRED_TRADE_COLS not available")
def test_validate_trades_accepts_minimal_frame():
    # Build a one-row trades frame satisfying REQUIRED_TRADE_COLS
    row = {}
    for col in REQUIRED_TRADE_COLS:
        name = str(col).lower()
        if "time" in name or "date" in name:
            row[col] = pd.Timestamp("2024-01-01")
        elif "side" in name:
            row[col] = "long"
        elif "pip" in name or "points" in name:
            row[col] = 1.0
        else:
            row[col] = 0.0
    trades = pd.DataFrame([row])

    out = validate_trades(trades)
    if isinstance(out, tuple):
        assert isinstance(out[0], pd.DataFrame)
    else:
        assert out is None or bool(out) is True


@pytest.mark.skipif(normalize_side is None, reason="normalize_side not available")
def test_normalize_side_series_idempotent_for_valid_values():
    s = pd.Series(["long", "short"])
    out = normalize_side(s)
    pd.testing.assert_series_equal(out, s)


@pytest.mark.skipif(normalize_side is None, reason="normalize_side not available")
def test_normalize_side_various_inputs_normalize_correctly():
    s = pd.Series(["buy", "sell", "+1", "-1", "1", "-1", " LONG ", " Short "])
    out = normalize_side(s)
    expected = pd.Series(["long", "short", "long", "short", "long", "short", "long", "short"])
    pd.testing.assert_series_equal(out, expected)
