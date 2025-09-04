def metrics(pnl: pd.Series, index: pd.DatetimeIndex) -> Dict[str, float]:
    pnl = pnl.fillna(0.0).astype(float)
    bpy = infer_bars_per_year(index)
    mu = pnl.mean()
    sd = pnl.std(ddof=0)
    ann_ret = (1.0 + mu) ** bpy - 1.0
    ann_vol = sd * np.sqrt(bpy)
    sharpe = ann_ret / (ann_vol + 1e-12)
    downside = pnl.clip(upper=0.0)
    dd = (1.0 + downside).var(ddof=0) ** 0.5 * np.sqrt(bpy)
    sortino = ann_ret / (dd + 1e-12)
    eq = (1.0 + pnl).cumprod()
    peaks = eq.cummax()
    mdd = ((eq / peaks) - 1.0).min()
    calmar = ann_ret / (abs(mdd) + 1e-12)
    return {
        "AnnReturn": float(ann_ret),
        "AnnVol": float(ann_vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "MaxDD": float(mdd),
        "Calmar": float(calmar),
    }

def metrics_by_period(pnl: pd.Series, freq: str = "A") -> pd.DataFrame:
    grp = pnl.fillna(0.0).groupby(pd.Grouper(freq=freq))
    out = []
    for g, s in grp:
        if s.empty:
            continue
        eq_end = float((1.0 + s).prod() - 1.0)
        ann = (1.0 + s.mean()) ** len(s) - 1.0
        vol = s.std(ddof=0) * np.sqrt(len(s))
        sharpe_like = ann / (vol + 1e-12)
        out.append({"period": g, "Return": eq_end, "Sharpe_like": sharpe_like})
    return pd.DataFrame(out).set_index("period") if out else pd.DataFrame(columns=["Return","Sharpe_like"])
