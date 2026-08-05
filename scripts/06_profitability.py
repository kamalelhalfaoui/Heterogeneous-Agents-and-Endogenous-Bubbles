"""
Figure 6 -- Which forecasting rule actually makes money.

Panels (a) and (b): the expanding-window mean profit of each rule, which is
what a trader who had followed that rule since the start would have earned on
average per period. Panel (c): the long-run average profit against the long-run
average population share.

Panel (c) is the substantive one. If evolutionary selection worked cleanly,
every point would lie on an upward-sloping line: profitable rules attract
followers. The extent to which it does not is a measure of how badly one period
of profit proxies for the quality of a rule.
"""

from _common import FIGDIR, TABDIR, announce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bh1998 import ABSParams, market, simulate
from bh1998.analysis import survival_table
from bh1998.plotting import MUTED, label_panel, save


def main() -> None:
    fig = plt.figure(figsize=(14.2, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.28)

    results = {}
    handles: dict[str, object] = {}

    for k, n_types in enumerate((6, 5)):
        ax = fig.add_subplot(gs[0, k])
        p = ABSParams(R=1.05, beta=2.0, n_steps=10_000, burn_in=1_000)
        res = simulate(p, market(n_types))
        results[f"{n_types} types"] = res

        rmp = res.running_mean_profit()
        for j, (name, colour) in enumerate(zip(res.names, res.colors)):
            (line,) = ax.plot(rmp[:, j], color=colour, lw=1.5)
            handles.setdefault(name, line)
        ax.plot(rmp.mean(axis=1), color="black", lw=1.6, ls="--")
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(2, 400)
        ax.set_xlabel("Period $t$")
        ax.set_title(f"{n_types} trader types", fontsize=10.5)
        label_panel(ax, f"({chr(97 + k)})")
        if k == 0:
            ax.set_ylabel("Expanding-window mean profit")

    # --- panel (c): profit against share ---------------------------------
    ax = fig.add_subplot(gs[0, 2])
    res = results["6 types"]
    shares = res.mean_fraction()
    profits = res.mean_profit()

    # Optimists and pessimists sit almost on top of each other, so their
    # labels are pushed in opposite directions rather than both placed above.
    offsets = {"Optimists": (-6, 13, "right"), "Pessimists": (6, -20, "left")}
    for name, colour, s, pr in zip(res.names, res.colors, shares, profits):
        dx, dy, ha = offsets.get(name, (0, 12, "center"))
        ax.scatter(s, pr, s=110, color=colour, zorder=3,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(
            name, (s, pr), textcoords="offset points", xytext=(dx, dy),
            ha=ha, fontsize=8.8, color=colour,
        )
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Long-run mean share")
    ax.set_ylabel("Long-run mean profit")
    ax.set_title("Does selection reward profit?", fontsize=10.5)
    ax.set_xlim(-0.04, 0.50)
    ax.margins(y=0.22)
    label_panel(ax, "(c)")

    corr = np.corrcoef(shares, profits)[0, 1]
    ax.text(
        0.97, 0.05, f"correlation of share and profit\n$r = {corr:+.2f}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.6),
    )

    black = plt.Line2D([], [], color="black", ls="--", lw=1.6)
    fig.legend(
        list(handles.values()) + [black],
        list(handles.keys()) + ["Cross-type average"],
        loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.10), frameon=False,
    )
    fig.suptitle(
        "Profitability of forecasting rules under evolutionary selection",
        fontsize=13, fontweight="bold", y=1.01,
    )
    announce(save(fig, FIGDIR / "fig06_profitability.png").name)

    tab = survival_table(results).round(4)
    tab.to_csv(TABDIR / "table_profitability.csv", index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
