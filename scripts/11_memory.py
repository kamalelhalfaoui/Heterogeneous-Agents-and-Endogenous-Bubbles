"""
Figure 11 -- Memory: the two parameters the original analysis held at zero.

The baseline calibration fixes delta = 0 and w = 0, which is the assumption that
traders have no memory whatsoever: fitness is one period of profit, and the
entire population reallocates itself every period. Figure 10 showed that this
assumption is what prevents the model from producing volatility clustering.
This figure maps what the two parameters actually do.

delta is inertia in the population shares -- the fraction of traders who do not
reconsider at all in a given period. w is the decay rate of the fitness
measure, so that 1/(1-w) is roughly the effective number of periods of past
performance a rule is judged on.

Panel (a): volatility of the deviation over the (delta, w) plane.
Panel (b): clustering, measured as the mean autocorrelation of absolute returns
over the first twenty lags, over the same plane.
Panel (c): a slice through panel (b) at three values of delta.

The two surfaces do not have the same shape, which is the point. Raising memory
calms the price level while making the timing of turbulence more persistent.
"""

from _common import FIGDIR, TABDIR, announce

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bh1998 import ABSParams, market, simulate
from bh1998.analysis import stylised_facts
from bh1998.plotting import ACCENT, HILITE, label_panel, save

DELTAS = np.linspace(0.0, 0.95, 20)
WS = np.linspace(0.0, 0.99, 20)
BASE = dict(R=1.05, beta=10.0, noise=0.1, n_steps=12_000, burn_in=3_000, seed=5)
RULES = dict(alpha_opt=0.3, phi_trend=1.05)


def run(delta: float, w: float):
    p = ABSParams(delta=float(delta), w=float(w), **BASE)
    res = simulate(p, market(6, **RULES))
    if res.diverged:
        return np.nan, np.nan
    try:
        sf = stylised_facts(res, max_lag=20)
        clus = float(np.mean(sf["acf_abs_returns"]))
    except Exception:
        clus = np.nan
    return float(np.std(res.trimmed().x)), clus


def main() -> None:
    vol = np.full((WS.size, DELTAS.size), np.nan)
    clus = np.full((WS.size, DELTAS.size), np.nan)

    print("  sweeping the (delta, w) plane ...")
    for j, w in enumerate(WS):
        for i, d in enumerate(DELTAS):
            vol[j, i], clus[j, i] = run(d, w)

    fig = plt.figure(figsize=(14.6, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.45)
    extent = [DELTAS[0], DELTAS[-1], WS[0], WS[-1]]

    # --- (a) volatility ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(vol, origin="lower", aspect="auto", extent=extent,
                   cmap="YlGnBu")
    fig.colorbar(im, ax=ax, label=r"Std. deviation of $x_t$")
    ax.set_xlabel(r"Share inertia $\delta$")
    ax.set_ylabel(r"Fitness memory $w$")
    ax.set_title("Price volatility", fontsize=10.5)
    ax.grid(False)
    label_panel(ax, "(a)")

    # --- (b) clustering ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(clus, origin="lower", aspect="auto", extent=extent,
                   cmap="RdYlBu_r", vmin=-0.1, vmax=0.35)
    fig.colorbar(im, ax=ax, label=r"Mean ACF of $|r_t|$")
    ax.set_xlabel(r"Share inertia $\delta$")
    ax.set_title("Volatility clustering", fontsize=10.5)
    ax.grid(False)
    ax.plot(0.9, 0.99, marker="*", ms=15, color="white",
            markeredgecolor="black", markeredgewidth=0.8)
    ax.annotate("calibration\nused in Fig. 10", (0.9, 0.99),
                xytext=(-10, -14), textcoords="offset points",
                ha="right", va="top", fontsize=8.4, color="#1c1c1c",
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    label_panel(ax, "(b)")

    # --- (c) slices --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    for d_target, colour, ls in (
        (0.0, "#8a8a8a", "-"), (0.5, ACCENT, "-"), (0.9, HILITE, "-")
    ):
        i = int(np.argmin(np.abs(DELTAS - d_target)))
        ax.plot(WS, clus[:, i], color=colour, ls=ls, lw=1.9,
                label=rf"$\delta = {DELTAS[i]:.2f}$")
    ax.axhline(0, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel(r"Fitness memory $w$")
    ax.set_ylabel(r"Mean ACF of $|r_t|$, lags 1-20")
    ax.set_title("Clustering against fitness memory", fontsize=10.5)
    ax.legend(fontsize=9)
    label_panel(ax, "(c)")

    fig.suptitle(
        "Memory in the selection mechanism, the assumption usually set to zero",
        fontsize=13, fontweight="bold", y=1.01,
    )
    announce(save(fig, FIGDIR / "fig11_memory.png").name)

    pd.DataFrame(clus, index=np.round(WS, 3), columns=np.round(DELTAS, 3)).round(
        4
    ).to_csv(TABDIR / "table_memory_clustering.csv")
    best = np.unravel_index(np.nanargmax(clus), clus.shape)
    print(f"  strongest clustering at delta={DELTAS[best[1]]:.2f}, "
          f"w={WS[best[0]]:.2f}: {clus[best]:.3f}")


if __name__ == "__main__":
    main()
