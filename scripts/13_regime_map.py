"""
Figure 13 -- A regime map of the parameter space.

Figures 8 and 9 each varied one parameter. This figure varies both at once and
maps three quantities over the (beta, phi) plane: the amplitude of the
deviation, the largest Lyapunov exponent, and the average share held by
fundamentalists. Regions where the trajectory diverges are hatched.

Panel (b) is the summary of the whole project. The chaotic region is a band,
not a corner: it requires trend extrapolation strong enough to amplify price
moves, and an intensity of choice high enough for the population to actually
reallocate towards whichever rule is currently winning. Weaken either and the
market settles into a cycle; strengthen phi much further and the price simply
explodes.

Panel (c) shows what happens to the stabilising rule inside that band. The
fundamentalist share is lowest exactly where the dynamics are wildest, because
during a bubble the fundamentalist forecast is the one losing money fastest.
The rule that would stabilise the market is selected out of it precisely when
it is most needed.
"""

from _common import FIGDIR, TABDIR, announce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bh1998 import ABSParams, market, simulate
from bh1998.analysis import lyapunov_rosenstein
from bh1998.plotting import label_panel, save

BETAS = np.linspace(0.5, 30.0, 26)
PHIS = np.linspace(0.0, 1.12, 26)
ALPHA_OPT = 0.3


def main() -> None:
    nb, npx = BETAS.size, PHIS.size
    vol = np.full((npx, nb), np.nan)
    lya = np.full((npx, nb), np.nan)
    fund = np.full((npx, nb), np.nan)
    div = np.zeros((npx, nb), dtype=bool)

    print(f"  sweeping {nb} x {npx} = {nb * npx} parameter combinations ...")
    for j, phi in enumerate(PHIS):
        for i, b in enumerate(BETAS):
            res = simulate(
                ABSParams(R=1.05, beta=float(b), n_steps=5_000, burn_in=2_000),
                market(6, alpha_opt=ALPHA_OPT, phi_trend=float(phi)),
            )
            if res.diverged:
                div[j, i] = True
                continue
            t = res.trimmed()
            vol[j, i] = float(np.std(t.x))
            lya[j, i] = lyapunov_rosenstein(t.x)
            fund[j, i] = float(t.n[:, 0].mean())

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    extent = [BETAS[0], BETAS[-1], PHIS[0], PHIS[-1]]
    common = dict(origin="lower", aspect="auto", extent=extent)

    def hatch_divergence(ax):
        if div.any():
            ax.contourf(
                BETAS, PHIS, div.astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["////"],
            )
            ax.contour(BETAS, PHIS, div.astype(float), levels=[0.5],
                       colors="#444444", linewidths=0.8)

    # --- (a) volatility ---------------------------------------------------
    ax = axes[0]
    # Below the bifurcation the trajectory converges to a fixed point and the
    # standard deviation is ~1e-7. Left unclipped that single region stretches
    # the log scale over seven decades and flattens everything else, so it is
    # floored and labelled as what it is: a steady state.
    FLOOR = 1e-3
    im = ax.imshow(np.log10(np.maximum(vol, FLOOR)), cmap="YlGnBu",
                   vmin=np.log10(FLOOR), **common)
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ std. deviation of $x_t$")
    ax.contour(BETAS, PHIS, np.nan_to_num(vol), levels=[FLOOR],
               colors="#c0392b", linewidths=1.2)
    ax.text(2.4, 0.5, "steady\nstate", fontsize=9, color="#7a5c00",
            ha="center", va="center")
    hatch_divergence(ax)
    ax.set_xlabel(r"Intensity of choice $\beta$")
    ax.set_ylabel(r"Trend extrapolation $\varphi$")
    ax.set_title("Amplitude", fontsize=10.5)
    ax.grid(False)
    label_panel(ax, "(a)")

    # --- (b) Lyapunov -----------------------------------------------------
    ax = axes[1]
    vmax = float(np.nanmax(np.abs(lya)))
    im = ax.imshow(lya, cmap="RdBu_r", vmin=-vmax, vmax=vmax, **common)
    fig.colorbar(im, ax=ax, label=r"$\lambda_{\max}$")
    ax.contour(BETAS, PHIS, np.nan_to_num(lya), levels=[0.01],
               colors="black", linewidths=1.1)
    hatch_divergence(ax)
    ax.set_xlabel(r"Intensity of choice $\beta$")
    ax.set_title(r"Chaos ($\lambda_{\max} > 0$, inside the contour)",
                 fontsize=10.5)
    ax.grid(False)
    label_panel(ax, "(b)")

    # --- (c) fundamentalist share ------------------------------------------
    ax = axes[2]
    im = ax.imshow(fund, cmap="viridis", **common)
    fig.colorbar(im, ax=ax, label="Mean fundamentalist share")
    ax.contour(BETAS, PHIS, np.nan_to_num(lya), levels=[0.01],
               colors="white", linewidths=1.1)
    hatch_divergence(ax)
    ax.set_xlabel(r"Intensity of choice $\beta$")
    ax.set_title("The stabilising rule is crowded out", fontsize=10.5)
    ax.grid(False)
    label_panel(ax, "(c)")

    fig.suptitle(
        "Regime map: chaos needs both amplification and fast switching  "
        "(hatched = explosive)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    announce(save(fig, FIGDIR / "fig13_regime_map.png").name)

    pd.DataFrame(lya, index=np.round(PHIS, 3),
                 columns=np.round(BETAS, 2)).round(4).to_csv(
        TABDIR / "table_regime_map_lyapunov.csv"
    )
    chaotic = np.nansum(lya > 0.01)
    total = np.sum(~np.isnan(lya))
    print(f"  chaotic in {chaotic}/{total} non-divergent cells "
          f"({chaotic / max(total, 1):.1%})")
    print(f"  explosive in {div.sum()}/{div.size} cells ({div.mean():.1%})")


if __name__ == "__main__":
    main()
