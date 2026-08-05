"""
Figure 9 -- Where the model really does become chaotic.

Figure 8 varied the intensity of choice and found period doubling followed by
quasi-periodicity, with a Lyapunov exponent that never left zero. This figure
varies the strength of trend extrapolation instead, at a fixed intensity of
choice, and finds something different: a genuinely positive exponent over a
wide interval.

The distinction matters because the two parameters do different economic work.
Beta governs how fast traders switch between rules; phi governs how strongly
one of those rules amplifies the last price move. Switching alone reorganises
the population without destabilising the price. Amplification is what makes the
price feed back on itself, and that is what produces chaos.

The vertical line marks phi = R, the point at which trend followers extrapolate
exactly as fast as the market discounts. Note that the complex region begins
somewhat before it: the destabilising force does not need to win outright, only
to be strong enough relative to the rules pulling the other way.
"""

from _common import FIGDIR, TABDIR, announce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bh1998 import ABSParams, market, simulate
from bh1998.analysis import bifurcation, lyapunov_rosenstein
from bh1998.plotting import ACCENT, HILITE, MUTED, label_panel, save

PHI_MIN, PHI_MAX = 0.0, 1.15
N_BIF = 380
N_LYA = 130
BETA = 10.0
ALPHA_OPT = 0.3


def main() -> None:
    base = ABSParams(R=1.05, beta=BETA)
    rk = dict(alpha_opt=ALPHA_OPT)

    print("  computing bifurcation diagram ...")
    grid = np.linspace(PHI_MIN, PHI_MAX, N_BIF)
    bx, by = bifurcation(
        grid, param="phi_trend", base=base, rules=None, rule_kwargs=rk,
        n_steps=3_000, burn_in=2_400, keep=180,
    )

    print("  computing Lyapunov spectrum ...")
    lgrid = np.linspace(PHI_MIN, PHI_MAX, N_LYA)
    lam = np.empty(N_LYA)
    diverged = np.zeros(N_LYA, dtype=bool)
    for i, v in enumerate(lgrid):
        res = simulate(
            ABSParams(R=1.05, beta=BETA, n_steps=6_000, burn_in=2_000),
            market(6, phi_trend=float(v), **rk),
        )
        diverged[i] = res.diverged
        lam[i] = np.nan if res.diverged else lyapunov_rosenstein(res.trimmed().x)

    fig, axes = plt.subplots(
        2, 1, figsize=(11.6, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12},
    )

    # --- (a) bifurcation --------------------------------------------------
    ax = axes[0]
    ax.plot(bx, by, ".", ms=0.55, color=ACCENT, alpha=0.35, rasterized=True)
    ax.axvline(1.05, color=HILITE, lw=1.1, ls="--")
    ax.text(1.05, 0.985, r"  $\varphi = R$", transform=ax.get_xaxis_transform(),
            color=HILITE, fontsize=9.5, va="top")
    # The attractor grows by two orders of magnitude across the sweep, so a
    # symmetric log scale is the only way to show the small-amplitude cycles
    # on the left and the large bubbles on the right in the same panel.
    ax.set_yscale("symlog", linthresh=0.5, linscale=0.6)
    ax.set_ylim(-2.0, 200)
    ax.set_ylabel(r"Deviation from fundamental, $x_t$  (symlog scale)")
    ax.set_title(
        r"Bifurcation diagram in the strength of trend extrapolation "
        rf"($\beta = {BETA:.0f}$, $R = 1.05$)",
        fontsize=11.5,
    )
    ax.grid(alpha=0.35)
    label_panel(ax, "(a)", dx=-0.055)

    # --- (b) Lyapunov -----------------------------------------------------
    ax = axes[1]
    ok = np.isfinite(lam)
    TOL = 0.01
    ax.axhspan(-TOL, TOL, color="#9aa5b1", alpha=0.22, lw=0,
               label=r"$|\lambda| < 0.01$: not resolvable from zero")
    ax.axhline(0, color=MUTED, lw=1.0, ls="--")
    ax.plot(lgrid[ok], lam[ok], color=ACCENT, lw=1.6,
            label=r"estimated $\lambda_{\max}$")
    ax.fill_between(
        lgrid[ok], TOL, lam[ok], where=lam[ok] > TOL,
        color=HILITE, alpha=0.35, interpolate=True, label="chaotic region",
    )
    ax.axvline(1.05, color=HILITE, lw=1.1, ls="--")

    if diverged.any():
        first = lgrid[diverged][0]
        ax.axvspan(first, PHI_MAX, color="#d0d0d0", alpha=0.55, lw=0)
        ax.text(0.5 * (first + PHI_MAX), 0.06, "explosive",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=9,
                color="#555555")

    ax.set_xlabel(r"Trend extrapolation coefficient $\varphi$")
    ax.set_ylabel(r"$\lambda_{\max}$")
    ax.set_xlim(PHI_MIN, PHI_MAX)
    ax.legend(loc="upper left", ncol=1, fontsize=8.8)
    label_panel(ax, "(b)", dx=-0.055)

    fig.suptitle(
        "Trend extrapolation, not switching speed, is what generates chaos",
        fontsize=13.5, fontweight="bold", y=0.985,
    )
    announce(save(fig, FIGDIR / "fig09_bifurcation_phi.png").name)

    pd.DataFrame(
        {"phi_trend": lgrid, "lyapunov": lam, "diverged": diverged}
    ).round(5).to_csv(TABDIR / "table_lyapunov_phi.csv", index=False)

    chaotic = lgrid[ok][lam[ok] > TOL]
    if chaotic.size:
        print(f"  chaotic for phi in [{chaotic.min():.3f}, {chaotic.max():.3f}]")
        print(f"  peak lambda_max = {np.nanmax(lam):.4f}")
    print(f"  explosive for {diverged.mean():.0%} of the grid")


if __name__ == "__main__":
    main()
