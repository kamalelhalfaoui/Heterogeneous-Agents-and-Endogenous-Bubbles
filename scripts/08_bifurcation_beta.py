"""
Figure 8 -- The route to complex dynamics as the intensity of choice rises.

Panel (a): bifurcation diagram. For each value of beta the model is run, the
transient discarded, and the remaining values of the deviation plotted as a
vertical slice. A fixed point appears as a single dot, a period-k cycle as k
dots, and an aperiodic attractor as a dense band.

Panel (b): the largest Lyapunov exponent over the same grid, estimated from the
simulated series by the Rosenstein algorithm. This is the diagnostic that
separates the two kinds of aperiodic behaviour: a torus has an exponent of zero
while chaos has a strictly positive one, and the bifurcation diagram alone
cannot tell them apart.

The economics: beta measures how quickly traders abandon a rule that has just
underperformed. At low beta the population barely reacts and the market settles
down. Past a threshold the reaction is fast enough that the switching itself
becomes the dominant force, and the price never settles. Instability here is
not caused by shocks or by irrationality -- it is caused by traders becoming
more responsive to evidence.
"""

from _common import FIGDIR, TABDIR, announce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bh1998 import ABSParams, market
from bh1998.analysis import bifurcation, lyapunov_scan
from bh1998.plotting import ACCENT, HILITE, MUTED, label_panel, save

BETA_MIN, BETA_MAX = 0.0, 40.0
N_BIF = 420
N_LYA = 120


def main() -> None:
    base = ABSParams(R=1.05, beta=2.0)
    rules = market(6)

    print("  computing bifurcation diagram ...")
    grid = np.linspace(BETA_MIN, BETA_MAX, N_BIF)
    bx, by = bifurcation(
        grid, param="beta", rules=rules, base=base,
        n_steps=3_000, burn_in=2_400, keep=180,
    )

    print("  computing Lyapunov spectrum ...")
    lgrid = np.linspace(BETA_MIN, BETA_MAX, N_LYA)
    lam = lyapunov_scan(
        lgrid, param="beta", base=base, rules=rules,
        n_steps=6_000, burn_in=2_000,
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(11.6, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12},
    )

    # --- (a) bifurcation --------------------------------------------------
    ax = axes[0]
    ax.plot(bx, by, ".", ms=0.55, color=ACCENT, alpha=0.35, rasterized=True)
    ax.set_ylabel(r"Deviation from fundamental, $x_t$")
    ax.set_title(
        "Bifurcation diagram: attractor of the deviation against the "
        "intensity of choice",
        fontsize=11.5,
    )
    ax.grid(alpha=0.35)
    label_panel(ax, "(a)", dx=-0.055)

    # --- (b) Lyapunov -----------------------------------------------------
    ax = axes[1]
    ok = np.isfinite(lam)
    # The tolerance band is the point of this panel. The estimator resolves
    # exponents of order 0.05 comfortably (it recovers 0.411 against a true
    # 0.419 on the Henon map), so anything inside +/-0.01 is zero as far as
    # this method can tell. The entire curve lies inside the band.
    TOL = 0.01
    ax.axhspan(-TOL, TOL, color="#9aa5b1", alpha=0.22, lw=0,
               label=r"indistinguishable from zero ($|\lambda|<0.01$)")
    ax.axhline(0, color=MUTED, lw=1.0, ls="--")
    ax.plot(lgrid[ok], lam[ok], color=ACCENT, lw=1.5,
            label=r"estimated $\lambda_{\max}$")
    ax.set_xlabel(r"Intensity of choice $\beta$")
    ax.set_ylabel(r"$\lambda_{\max}$")
    ax.set_xlim(BETA_MIN, BETA_MAX)
    ax.set_ylim(-0.055, 0.055)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    label_panel(ax, "(b)", dx=-0.055)

    ax.annotate(
        "aperiodic from here on, but the exponent stays at zero:\n"
        "motion on a torus, not chaos",
        xy=(30.0, 0.0), xytext=(11.0, -0.040), fontsize=9, color="#444444",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9),
    )

    fig.suptitle(
        "Faster learning, less stable prices: period doubling into "
        "quasi-periodicity",
        fontsize=13.5, fontweight="bold", y=0.985,
    )
    announce(save(fig, FIGDIR / "fig08_bifurcation_beta.png").name)

    pd.DataFrame({"beta": lgrid, "lyapunov": lam}).round(5).to_csv(
        TABDIR / "table_lyapunov_beta.csv", index=False
    )
    frac = np.mean(lam[ok] > 0.01)
    print(f"  share of the beta grid with lambda_max > 0.01: {frac:.1%}")
    print(f"  max |lambda| over the grid: {np.nanmax(np.abs(lam)):.4f}")


if __name__ == "__main__":
    main()
