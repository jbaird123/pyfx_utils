def stability_heatmap(results, x_param, y_param, value="objective"):
    """
    Pivot a tuning results DataFrame into a heatmap-friendly table.
    """
    import pandas as pd, numpy as np, matplotlib.pyplot as plt
    pivot = results.pivot_table(index=f"param_{y_param}",
                                columns=f"param_{x_param}",
                                values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_xlabel(x_param); ax.set_ylabel(y_param); ax.set_title(f"{value} heatmap")
    fig.colorbar(im, ax=ax)
    return pivot
