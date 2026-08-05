"""
Figure 12 -- The ecology of strategies: who survives, and does survival pay?

Panel (a): long-run average population share of each rule across four market
configurations. Panel (b): long-run average profit for the same. Panel (c): the
survival premium, defined as a rule's average share minus the share it would
hold if the population were split evenly. A positive premium means selection
favours the rule.

The result worth noticing is that the biased rules -- optimists and pessimists,
which forecast a constant and ignore all data -- take the largest shares and
earn the highest profits, while adapters, the only rule that forms an unbiased
long-run forecast, consistently lose money.

This is not a paradox. Profit in this model is earned by holding a position
that moves with the price, and the price oscillates between the optimists' and
the pessimists' forecasts. A rule anchored at either extreme is right half the
time and spectacularly right when the market swings its way. A rule that
averages past prices is never far wrong and never usefully right, and it pays
the spread every period. Being well calibrated and being profitable are
different objectives, and the selection mechanism only rewards the second.
"""

from _common import FIGDIR, TABDIR, announce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bh1998 import ABSParams, market, simulate
from bh1998.plotting import MUTED, label_panel, save

CONFIGS = [
    ("6 types\n$\\beta=2$", 6, dict(beta=2.0)),
    ("5 types\n$\\beta=2$", 5, dict(beta=2.0)),
    ("6 types\n$\\beta=20$", 6, dict(beta=20.0)),
    ("6 types\n$\\beta=2$, $R=1.6$", 6, dict(beta=2.0, R=1.6)),
]

ORDER = [
    "Fundamentalists", "Trend followers", "Adapters",
    "Learners", "Optimists", "Pessimists",
]


def main() -> None:
    shares, profits, premia = {}, {}, {}
    colours: dict[str, str] = {}

    for label, n_types, kw in CONFIGS:
        p = ABSParams(R=kw.get("R", 1.05), beta=kw["beta"],
                      n_steps=20_000, burn_in=4_000)
        res = simulate(p, market(n_types))
        s = res.mean_fraction()
        pr = res.mean_profit()
        even = 1.0 / n_types
        shares[label] = dict(zip(res.names, s))
        profits[label] = dict(zip(res.names, pr))
        premia[label] = {n: v - even for n, v in zip(res.names, s)}
        colours.update(dict(zip(res.names, res.colors)))

    df_s = pd.DataFrame(shares).reindex(ORDER)
    df_p = pd.DataFrame(profits).reindex(ORDER)
    df_x = pd.DataFrame(premia).reindex(ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    labels = list(shares)
    xpos = np.arange(len(labels))
    width = 0.13

    def grouped(ax, df, ylabel, title, centre=None):
        for k, name in enumerate(ORDER):
            vals = df.loc[name].values.astype(float)
            ax.bar(xpos + (k - 2.5) * width, np.nan_to_num(vals), width,
                   color=colours[name], label=name,
                   edgecolor="white", linewidth=0.5)
        if centre is not None:
            ax.axhline(centre, color=MUTED, lw=0.9, ls="--")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", visible=False)

    grouped(axes[0], df_s, "Long-run mean share", "Who survives")
    label_panel(axes[0], "(a)")

    grouped(axes[1], df_p, "Long-run mean profit", "Who earns", centre=0.0)
    label_panel(axes[1], "(b)")

    grouped(axes[2], df_x, "Share minus even split",
            "Survival premium relative to an even split", centre=0.0)
    label_panel(axes[2], "(c)")

    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.suptitle(
        "Selection rewards being extreme, not being right",
        fontsize=13.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    announce(save(fig, FIGDIR / "fig12_ecology.png").name)

    out = pd.concat(
        {"share": df_s, "profit": df_p, "premium": df_x}, axis=1
    ).round(4)
    out.to_csv(TABDIR / "table_ecology.csv")
    print(df_s.round(3).to_string())
    print()
    print(df_p.round(3).to_string())


if __name__ == "__main__":
    main()
