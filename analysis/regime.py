import pandas as pd 
import numpy as np
try:
    from sklearn.cluster import KMeans  # optional
except Exception:
    KMeans = None


def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight regime features from prices.
    Expects 'close' column present. Generates:
      - vol_pips_20: rolling std of close-to-close pips (20)
      - trend_50_200: EMA50-EMA200 in price units
      - bb_width_20: (upper-lower)/middle from Bollinger(20,2)
    """
    out = pd.DataFrame(index=df.index)
    if 'close' not in df.columns:
        return out
    # Approximate pips from close diffs (user may replace with their own precise pips)
    # For a robust notebook, you can pass a prepared pips_per_bar separately; here we just build a feature.
    close = df['close'].astype(float)
    pip_factor = 10000.0  # override per instrument if needed
    pips = close.diff().fillna(0) * pip_factor
    out['vol_pips_20'] = pips.rolling(20).std()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    out['trend_50_200'] = (ema50 - ema200)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2*std
    lower = mid - 2*std
    out['bb_width_20'] = (upper - lower) / (mid.replace(0, np.nan))
    return out.dropna()


def kmeans_regimes(feats: pd.DataFrame, k: int = 3, seed: int = 42) -> pd.Series:
    if feats.empty or KMeans is None:
        return pd.Series(index=feats.index, dtype='float64')
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labs = km.fit_predict(feats.values)
    return pd.Series(labs, index=feats.index, name='regime')


def perf_by_regime(ret_pips: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    if ret_pips.empty or regimes.empty:
        return pd.DataFrame()
    df = pd.concat({'ret_pips': ret_pips, 'regime': regimes}, axis=1).dropna()
    grp = df.groupby('regime')['ret_pips']
    return pd.DataFrame({
        'total_pips': grp.sum(),
        'mean_pips': grp.mean(),
        'n_bars': grp.size(),
    }).sort_index()

