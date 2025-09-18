# pyfx_utils/stats.py
from __future__ import annotations
import pandas as pd
import numpy as np

def cumulative_pips(trades: pd.DataFrame, index: pd.DatetimeIndex, when: str = "exit") -> pd.Series:
    """
    Build a per-bar cumulative pips series from a trade ledger.
    - 'when' is 'exit' or 'entry' to choose when to book the pips.
    """
    assert when in ("exit", "entry")
    tcol = f"{when}_time"
    pips_per_bar = pd.Series(0.0, index=index, dtype="float64")
    if trades.empty:
        return pips_per_bar
    increments = trades.groupby(tcol)["pips"].sum()
    increments = increments.reindex(index, fill_value=0.0)
    return increments.cumsum()



def trades_summary(trades: pd.DataFrame) -> dict:
    """
    Minimal trade-level summary stats in pips (no capital, no fees).
    Requires a 'pips' column in trades DataFrame.
    """
    if trades is None or len(trades) == 0:
        return {
            "trades": 0,
            "total_pips": 0.0,
            "win_rate": None,
            "avg_pips": None,
            "median_pips": None,
            "max_win": None,
            "max_loss": None,
        }

    wins = (trades["pips"] > 0).sum()
    total = int(len(trades))
    return {
        "trades": total,
        "total_pips": float(trades["pips"].sum()),
        "win_rate": wins / total if total else None,
        "avg_pips": float(trades["pips"].mean()),
        "median_pips": float(trades["pips"].median()),
        "max_win": float(trades["pips"].max()),
        "max_loss": float(trades["pips"].min()),
    }

def drawdown_pips(cum_pips):
  """Return a Series of drawdown (in pips) from running max."""
  running_max = cum_pips.cummax()
  return cum_pips - running_max

def max_drawdown_pips(cum_pips) -> float:
  """Most negative drawdown (pips)."""
  return float(drawdown_pips(cum_pips).min())

def streaks(trades):
  """Return (max_win_streak, max_loss_streak) based on pips > 0 / < 0."""
  if trades is None or trades.empty: return (0, 0)
  signs = (trades["pips"] > 0).astype(int).values
  max_w = max_l = cur_w = cur_l = 0
  for s in signs:
      if s == 1:
          cur_w += 1; max_w = max(max_w, cur_w); cur_l = 0
      else:
          cur_l += 1; max_l = max(max_l, cur_l); cur_w = 0
  return (int(max_w), int(max_l))

def _monte_carlo_paths(
    trade_pips: pd.Series | np.ndarray,
    iters: int = 2000,
    horizon: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Bootstrap trade pips to simulate equity paths of length `horizon` trades.
    Returns array shape (iters, horizon) of cumulative pips paths.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(trade_pips, dtype=float)
    if arr.size == 0:
        hor = (horizon if (horizon and horizon > 0) else 0) or 0
        return np.zeros((iters, max(hor, 0)))
    hor = arr.size if (horizon is None or horizon <= 0) else int(horizon)
    out = np.empty((iters, hor), dtype=float)
    for i in range(iters):
        sample = rng.choice(arr, size=hor, replace=True)
        out[i] = sample.cumsum()
    return out

def monte_carlo_summary(
    trade_pips: pd.Series | np.ndarray,
    iters: int = 2000,
    horizon: int | None = None,
    seed: int = 42,
    percentiles: Iterable[float] = (5, 50, 95),
) -> Dict[str, float]:
    """
    Summary statistics for Monte Carlo bootstrapped equity paths built from trade-level pips.

    Returns (JSON-friendly):
      - p05, p50, p95: percentiles of ending total pips (matches your existing names)
      - exp: mean of ending totals (expected ending pips)
      - mdd05: max drawdown of the path whose ending total is at the 5th percentile
      Additionally (non-breaking enrichments):
      - p05_mdd, p50_mdd, p95_mdd: percentiles of per-path max drawdowns
      - prob_neg_total: probability the ending total is negative
    """
    paths = _monte_carlo_paths(trade_pips, iters=iters, horizon=horizon, seed=seed)
    if paths.size == 0:
        return {
            "p05": 0.0, "p50": 0.0, "p95": 0.0, "exp": 0.0, "mdd05": 0.0,
            "p05_mdd": 0.0, "p50_mdd": 0.0, "p95_mdd": 0.0, "prob_neg_total": 0.0,
            "n_sims": int(paths.shape[0]), "horizon": int(paths.shape[1]) if paths.ndim == 2 else 0,
        }

    final = paths[:, -1]
    q = np.percentile
    p = sorted(set(percentiles))
    # Ending totals percentiles (keep your original names for 5/50/95)
    pcts = q(final, p).tolist()
    pct_map = {int(x): float(v) for x, v in zip(p, pcts)}
    p05 = pct_map.get(5, float(q(final, 5)))
    p50 = pct_map.get(50, float(q(final, 50)))
    p95 = pct_map.get(95, float(q(final, 95)))
    exp = float(final.mean())

    # Compute per-path max drawdowns
    roll_max = np.maximum.accumulate(paths, axis=1)
    dd = paths - roll_max
    mdds = dd.min(axis=1)  # negative numbers

    # mdd05: your original definition—DD of the path whose ending total is ~5th percentile
    worst_idx = int(np.clip(round(0.05 * (len(final) - 1)), 0, len(final) - 1))
    idx_sorted = np.argsort(final)  # ascending by ending total
    eq_5th = paths[idx_sorted[worst_idx]]
    rm_5th = np.maximum.accumulate(eq_5th)
    dd_5th = eq_5th - rm_5th
    mdd05 = float(dd_5th.min())

    # Extra (useful) fields—percentiles of the MDD distribution + prob of loss
    p05_mdd = float(q(mdds, 5))
    p50_mdd = float(q(mdds, 50))
    p95_mdd = float(q(mdds, 95))
    prob_neg_total = float((final < 0).mean())

    return {
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "exp": float(exp),
        "mdd05": mdd05,
        "p05_mdd": p05_mdd,
        "p50_mdd": p50_mdd,
        "p95_mdd": p95_mdd,
        "prob_neg_total": prob_neg_total,
        "n_sims": int(paths.shape[0]),
        "horizon": int(paths.shape[1]),
    }


def compute_pips(df: pd.DataFrame, symbol: str) -> pd.Series:
    """
    Compute pips for a trades DataFrame using entry/exit price and side.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: ["side", "entry_price", "exit_price"]
        side can be strings ('long'/'short'/'buy'/'sell') or numeric (+1/-1).
    symbol : str
        Used only to infer pip size (see infer_pip_size).

    Returns
    -------
    Series[float]
        Signed pips per trade.
    """
    pip = infer_pip_size(symbol)

    def side_sign(x):
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("long", "buy", "+1", "1"):
                return 1
            if s in ("short", "sell", "-1"):
                return -1
        return 1 if x in (1, +1, True) else -1

    sign = df["side"].map(side_sign)
    raw = (df["exit_price"] - df["entry_price"]) / pip
    return sign * raw
def infer_pip_size(symbol: str) -> float:
    """
    Very lightweight heuristic:
    - JPY pairs (e.g., 'USDJPY', 'EUR/JPY') use 0.01
    - Everything else defaults to 0.0001

    Notes
    -----
    - This is intentionally simple; if you maintain contract specs elsewhere,
      consider wiring that in and making this a table lookup instead.
    """
    return 0.01 if "JPY" in symbol.replace("/", "") else 0.0001

