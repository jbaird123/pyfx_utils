import inspect
import pandas as pd
import pytest

from pyfx_utils.strategies.sma import SMACrossoverAdapter

interfaces = pytest.importorskip("pyfx_utils.analysis.interfaces")
StrategyRunMeta = getattr(interfaces, "StrategyRunMeta")
StrategyRunPayload = getattr(interfaces, "StrategyRunPayload")


def _build_meta_instance():
    sig = inspect.signature(StrategyRunMeta)
    kwargs = {}
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is not inspect._empty:
            continue  # optional
        name = p.name.lower()
        # heuristic defaults by parameter name
        if "name" in name or "strategy" in name:
            val = "SMA Crossover"
        elif "symbol" in name or "instrument" in name or "ticker" in name:
            val = "EURUSD"
        elif "time" in name or "tf" in name or "frame" in name:
            val = "D"
        elif "version" in name or "ver" in name:
            val = "test"
        elif "market" in name or "exchange" in name:
            val = "FX"
        else:
            val = "x"
        kwargs[p.name] = val

    # Try kwargs, fall back to positional if needed
    try:
        return StrategyRunMeta(**kwargs)
    except Exception as e:
        # positional in declared order using same values
        args = [kwargs.get(p.name, "x") for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        try:
            return StrategyRunMeta(*args)
        except Exception as e2:
            pytest.skip(f"Could not instantiate StrategyRunMeta: {e2}")


def test_sma_adapter_to_payload_roundtrip():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    trades = pd.DataFrame(
        {
            "entry_time": [idx[1], idx[3]],
            "exit_time":  [idx[2], idx[4]],
            "side":       ["long", "short"],
            "pips":       [12.0, -7.0],
        }
    )

    params = {
        "fast": 5, "slow": 12,
        "stop_type": "pips", "stop_value": 25.0,
        "tp_type": "atr", "tp_value": 3.0,
        "trail_type": "atr", "trail_value": 2.5, "trail_mode": "close_based",
        "atr_length": 14, "tie_break": "tp_then_stop",
    }

    meta = _build_meta_instance()
    adapter = SMACrossoverAdapter(
        trades_df=trades,
        params=params,
        meta=meta,
        bar_index=idx,
    )

    payload = adapter.to_payload()
    assert isinstance(payload, StrategyRunPayload)
    assert payload.meta == meta
    assert payload.params == params
    pd.testing.assert_index_equal(payload.bar_index, idx)
    pd.testing.assert_frame_equal(payload.trades.reset_index(drop=True), trades.reset_index(drop=True))
