# pyfx_utils/colab_boot.py
def boot(code_dir="/content/drive/MyDrive/fx/code", ensure=("lightweight-charts",)):
    import os, sys, importlib, subprocess

    # 1) Mount Drive (Colab)
    try:
        from google.colab import drive  # type: ignore
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
    except Exception:
        pass

    # 2) Put your code dir on path (ahead of site-packages)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    # 3) Hot-reload pyfx_utils to pick up local edits
    for m in list(sys.modules):
        if m.startswith("pyfx_utils"):
            del sys.modules[m]
    importlib.invalidate_caches()

    # 3b) Ensure optional deps (module names may differ from pip names)
    for p in ensure or ():
        mod = p.replace("-", "_")
        try:
            __import__(mod)
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

    # 4) Imports to expose in the notebook

    # viz (requires lightweight_charts)
    from pyfx_utils.viz import (
        build_ohlc, new_chart, add_sma,
        prepare_trades_for_chart, add_trades, add_y_padding,
    )

    # indicators
    from pyfx_utils.indicators import add_indicators

    # utils
    from pyfx_utils.utils import load_fx_csv, resample
    from pyfx_utils.utils.stats import infer_pip_size, compute_pips  # no pip_factor anymore

    # analysis (your new helpers / interfaces)
    from pyfx_utils.analysis import annotate_trades, walk_forward_ranges
    from pyfx_utils.analysis.interfaces import build_pips_brief, StrategyRunMeta

    # backtests
    from pyfx_utils.backtests import (
        BTConfig,
        backtest_signal,
        equity_curve_from_trades,
        equity_curve_from_signal,
        metrics,
        metrics_by_period,
    )
    from pyfx_utils.backtests.core import backtest, trade_pnls

    # 5) Return a namespace dict that %pyfx_boot will inject into globals()
    return dict(
        # viz
        build_ohlc=build_ohlc, new_chart=new_chart, add_sma=add_sma,
        prepare_trades_for_chart=prepare_trades_for_chart, add_trades=add_trades, add_y_padding=add_y_padding,

        # indicators
        add_indicators=add_indicators,

        # utils
        load_fx_csv=load_fx_csv, resample=resample,
        infer_pip_size=infer_pip_size, compute_pips=compute_pips,

        # analysis
        annotate_trades=annotate_trades,
        walk_forward_ranges=walk_forward_ranges,
        build_pips_brief=build_pips_brief, StrategyRunMeta=StrategyRunMeta,

        # backtests
        BTConfig=BTConfig, backtest_signal=backtest_signal,
        equity_curve_from_trades=equity_curve_from_trades,
        equity_curve_from_signal=equity_curve_from_signal,
        metrics=metrics, metrics_by_period=metrics_by_period,
        backtest=backtest, trade_pnls=trade_pnls,
    )
