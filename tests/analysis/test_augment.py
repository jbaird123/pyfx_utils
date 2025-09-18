import numpy as np
import pandas as pd
import pytest

augment = pytest.importorskip("pyfx_utils.analysis.augment")

# statsmodels is optional
sm = pytest.importorskip("statsmodels", reason="statsmodels not installed (required for ADF test)")
from pyfx_utils.analysis.augment import adf_test


def test_adf_test_stationary_vs_trending():
    # Stationary: white noise
    rng = np.random.RandomState(42)
    x_stat = pd.Series(rng.normal(0, 1, size=1000))

    # Trending: random walk (non-stationary)
    steps = rng.normal(0, 1, size=1000)
    x_rw = pd.Series(steps).cumsum()

    stat_res = adf_test(x_stat)
    rw_res = adf_test(x_rw)

    # Be flexible about return type: dict with 'pvalue' or tuple (stat, p)
    def get_pvalue(res):
        if isinstance(res, dict):
            return float(res.get("pvalue"))
        if isinstance(res, (tuple, list)) and len(res) >= 2:
            return float(res[1])
        raise AssertionError("Unexpected adf_test return type")

    p_stat = get_pvalue(stat_res)
    p_rw = get_pvalue(rw_res)

    assert p_stat < 0.2, "Stationary series should have relatively low ADF p-value"
    assert p_rw > 0.1, "Random walk should have relatively high ADF p-value"
