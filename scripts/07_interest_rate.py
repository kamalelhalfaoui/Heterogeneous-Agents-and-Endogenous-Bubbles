"""
Figure 7 -- The risk-free rate as a control parameter.

A common way to ask what the risk-free rate does to profitability is to run the
model at a handful of values of R and compare. That is done here, and then
extended, because four snapshots can hide the shape of the curve between them.

Panels (a)-(d): the expanding-window mean profit at four values of R.

Panel (e): a continuous sweep of the long-run mean profit of every rule over R
in [1.01, 2.0], which shows the whole curve rather than four points on it.

Panel (f): the volatility of the deviation over the same sweep. The mechanism
is visible directly in the pricing equation R x_t = sum_h n_{h,t} f_{h,t}: a
larger R divides the same weighted forecast by a larger number, so the
deviation is compressed towards zero. Rising R therefore does not "reward
fundamentalists" so much as shrink the arena everybody is competing in.
"""

from _common import FIGDIR, TABDIR, announce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bh1998 import ABSParams, market, simulate
from bh1998.plotting import ACCENT, MUTED, label_panel, save

R_PANELS = (1.05, 1.2, 1.6, 2.0)
R_GRID = np.linspace(1.01, 2.0, 90)


def main() -> None:
    fig = plt.figure(figsize=(13.6, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.18], hspace=0.45,
                          wspace=0.30)

    handles: dict[str, object] = {}

    # --- top row: four snapshots -----------------------------------------
    for k, R in enumerate(R_PANELS):
        ax = fig.add_subplot(gs[0, k])
        res = simulate(
            ABSParams(R=R, beta=2.0, n_steps=8_000, burn_in=1_000), market(6)
        )
        rmp = res.running_mean_profit()
        for j, (name, colour) in enumerate(zip(res.names, res.colors)):
            (line,) = ax.plot(rmp[:, j], color=colour, lw=1.4)
            handles.setdefault(name, line)
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(2, 300)
        ax.set_title(f"$R = {R}$", fontsize=10.5)
        ax.set_xlabel("Period $t$")
        label_panel(ax, f"({chr(97 + k)})")
        if k == 0:
            ax.set_ylabel("Mean profit")

    # --- bottom left: profit sweep ---------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    profit_curves = np.full((len(R_GRID), 6), np.nan)
    vol = np.full(len(R_GRID), np.nan)

    for i, R in enumerate(R_GRID):
        res = simulate(
            ABSParams(R=float(R), beta=2.0, n_steps=4_000, burn_in=1_500),
            market(6),
        )
        if res.diverged:
            continue
        profit_curves[i] = res.mean_profit()
        vol[i] = float(np.std(res.trimmed().x))

    names = market(6)
    for j, rule in enumerate(names):
        ax.plot(R_GRID, profit_curves[:, j], color=rule.color, lw=1.7,
                label=rule.name)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    for R in R_PANELS:
        ax.axvline(R, color="#bbbbbb", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("Gross risk-free return $R$")
    ax.set_ylabel("Long-run mean profit")
    ax.set_title("Mean profit across the whole range of $R$", fontsize=10.5)
    label_panel(ax, "(e)")

    # --- bottom right: volatility sweep ----------------------------------
    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(R_GRID, vol, color=ACCENT, lw=2.0)
    ax.fill_between(R_GRID, 0, vol, color=ACCENT, alpha=0.12)
    for R in R_PANELS:
        ax.axvline(R, color="#bbbbbb", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("Gross risk-free return $R$")
    ax.set_ylabel(r"Std. deviation of $x_t$")
    ax.set_title(r"Discounting compresses the deviation: $\sigma_x \sim 1/R$",
                 fontsize=10.5)
    ax.set_ylim(bottom=0)
    label_panel(ax, "(f)")

    fig.legend(
        list(handles.values()), list(handles.keys()),
        loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.045), frameon=False,
    )
    fig.suptitle(
        "The risk-free rate as a control parameter",
        fontsize=13, fontweight="bold", y=0.99,
    )
    announce(save(fig, FIGDIR / "fig07_interest_rate.png").name)

    out = pd.DataFrame(profit_curves, columns=[r.name for r in names])
    out.insert(0, "R", R_GRID)
    out["std_x"] = vol
    out.round(5).to_csv(TABDIR / "table_interest_rate_sweep.csv", index=False)
    print(out.iloc[::12].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
