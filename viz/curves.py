def plot_curves(curves: dict, title="Cumulative Pips — Comparison"):
    """
    curves: dict[name -> pd.Series of cumulative pips aligned to same index]
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,5))
    for name, ser in curves.items():
        ax.plot(ser.index, ser.values, lw=1.4, label=name)
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("Pips")
    ax.grid(True, alpha=0.3); ax.legend()
    return ax
